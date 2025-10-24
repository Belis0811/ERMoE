import torch, torch.nn as nn
from ViT3D import VisionTransformer3D as ViT3D

class ERMoEBrain(nn.Module):
    def __init__(
        self,
        top_k: int = 2,
        threshold: float = 0.5,
        free_experts=(0,),
        patch_size=(7, 7, 7),
        hidden_dim: int = 256,
        img_size=(91, 109, 91),
        num_layers: int = 4,
        num_heads: int = 4,
        device=None,
        gm_ckpt: str = "gm_expert.pth",
        wm_ckpt: str = "wm_expert.pth",
        csf_ckpt: str = "csf_expert.pth",
    ):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_k = top_k
        self.threshold = threshold
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_experts = 3

        self.register_buffer('_free_mask', self._build_free_mask(free_experts), persistent=False)

        self.shared_vit = ViT3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=hidden_dim,
            depth=num_layers,
            num_heads=num_heads
        ).to(self.device)

        for p in self.shared_vit.parameters():
            p.requires_grad = False
        self.shared_vit.eval()

        self.experts = nn.ModuleList([
            ViT3D(img_size=img_size, patch_size=patch_size, in_chans=1,
                  embed_dim=hidden_dim, depth=num_layers, num_heads=num_heads),
            ViT3D(img_size=img_size, patch_size=patch_size, in_chans=1,
                  embed_dim=hidden_dim, depth=num_layers, num_heads=num_heads),
            ViT3D(img_size=img_size, patch_size=patch_size, in_chans=1,
                  embed_dim=hidden_dim, depth=num_layers, num_heads=num_heads),
        ]).to(self.device)

        self.experts[0].load_state_dict(torch.load(gm_ckpt, map_location=self.device))
        self.experts[1].load_state_dict(torch.load(wm_ckpt, map_location=self.device))
        self.experts[2].load_state_dict(torch.load(csf_ckpt, map_location=self.device))

        for expert in self.experts:
            for p in expert.parameters(): p.requires_grad = False
            expert.eval()

        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])

    @staticmethod
    def _build_free_mask(free_experts):
        max_idx = max(free_experts) if free_experts else -1
        mask = torch.zeros(max_idx+1, dtype=torch.bool)
        for i in free_experts:
            if i >= 0:
                if i >= mask.size(0):
                    mask = torch.cat([mask, torch.zeros(i - mask.size(0) + 1, dtype=torch.bool)])
                mask[i] = True
        return mask

    @torch.no_grad()
    def _encode(self, model, x):
        output = model(x)  
        last_hidden = output.last_hidden_state  
        attn = output.attentions[-1]         
        cls = last_hidden[:, 0, :]            
        patches = last_hidden[:, 1:, :]        
        attn_mean = attn.mean(dim=1)          
        attn_pp = attn_mean[:, 1:, 1:]        
        return cls, patches, attn_pp

    @staticmethod
    def _eig_from_patches(patches):
        B, P, D = patches.shape
        U_list = []; L_list = []
        for b in range(B):
            X = patches[b]                   
            C = X @ X.T                     
            eigvals, eigvecs = torch.linalg.eigh(C)
            U_list.append(eigvecs)
            L_list.append(eigvals)
        return U_list, L_list

    @staticmethod
    def _diag_energy_score(U, A):
        A_proj = U.T @ A @ U                
        diag = torch.diag(A_proj)
        energy_diag = torch.sum(diag**2)
        total_energy = torch.sum(A_proj**2) + 1e-10
        return energy_diag / total_energy

    @torch.no_grad()
    def route(self, images):
        x = images.to(self.device)
        B = x.shape[0]

        scores = torch.zeros(B, len(self.experts), device=self.device)
        cache = []
        cls_shared, patches, attn_pp = self._encode(self.shared_vit, x)

        for e in range(3):
            mask = region_patch_masks[e]  
            patches_e = patches[:, mask, :]       
            attn_e = attn_pp[:, mask][:, :, mask]  
            U_list, _ = self._eig_from_patches(patches_e)
            for b in range(B):
                U = U_list[b]                      
                A = attn_e[b]                      
                scores[b, e] = self._diag_energy_score(U, A)
            cache.append((cls_shared, patches_e, attn_e))

        sel_mask = torch.zeros_like(scores, dtype=torch.bool)
        free_indices = [i for i in range(min(3, self._free_mask.numel())) 
                        if self._free_mask.numel()>0 and self._free_mask[i]]
        for b in range(B):
            above = (scores[b] >= self.threshold).nonzero(as_tuple=True)[0].tolist()
            selected = []
            if len(above) > 0:
                if len(above) > self.top_k:
                    above.sort(key=lambda idx: float(scores[b, idx]), reverse=True)
                    selected = above[:self.top_k]
                else:
                    selected = list(above)
            needed = self.top_k - len(selected)
            if needed > 0:
                if free_indices:
                    pool = [i for i in free_indices if i not in selected]
                    if pool:
                        perm = torch.randperm(len(pool), device=self.device)
                        add = [pool[i] for i in perm[:needed].tolist()]
                        selected.extend(add)
                        needed = max(0, needed - len(add))
                if needed > 0:
                    order = torch.argsort(scores[b], descending=True).tolist()
                    for idx in order:
                        if idx not in selected:
                            selected.append(idx)
                            needed -= 1
                            if needed == 0: break
            for idx in selected:
                sel_mask[b, idx] = True

        return scores, sel_mask, cache

    def forward(self, images):
        scores, sel_mask, cache = self.route(images)
        B = scores.size(0)
        pred_sum = torch.zeros(B, device=self.device)
        weight_sum = torch.zeros(B, device=self.device)

        for e in range(3):
            if sel_mask[:, e].any():
                x_e = images.to(self.device) * region_mask_tensor[e]
                cls_e, _, _ = self._encode(self.experts[e], x_e)
                preds_e = self.heads[e](cls_e).squeeze()
                weight = scores[:, e] * sel_mask[:, e].float()
                pred_sum += preds_e * weight
                weight_sum += weight

        weight_sum = weight_sum.clamp_min(1e-8)
        out = pred_sum / weight_sum
        return out
