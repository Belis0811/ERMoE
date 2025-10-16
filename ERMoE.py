import torch
import torch.nn as nn
from copy import deepcopy
from transformers import ViTModel, ViTImageProcessor

class ERMoE(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_experts=8,
        top_k=2,
        threshold=0.5,                  # New threshold parameter for Top-D selection
        free_experts=(0,),
        pretrained_name="google/vit-base-patch16-224",
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_experts = num_experts
        self.top_k = top_k
        self.threshold = threshold      # Store the confidence threshold
        self.register_buffer("_free_mask", self._build_free_mask(free_experts), persistent=False)

        # Load pre-trained ViT model as base for experts
        base_model = ViTModel.from_pretrained(
            pretrained_name,
            output_attentions=True,
            output_hidden_states=False
        )
        # Create experts as frozen copies of the base model
        experts = [deepcopy(base_model) for _ in range(num_experts)]
        self.experts = nn.ModuleList(experts)
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = False
            expert.eval()  # ensure experts are in eval mode (no dropout, etc.)

        hidden_dim = base_model.config.hidden_size  
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_experts)])

        # Image processor for preprocessing input images
        self.processor = ViTImageProcessor.from_pretrained(pretrained_name)
        self.to(self.device)

    @staticmethod
    def _build_free_mask(free_experts):
        """Create a boolean mask for free experts indices."""
        max_idx = max(free_experts) if len(free_experts) > 0 else -1
        if max_idx < 0:
            return torch.zeros(0, dtype=torch.bool)
        mask = torch.zeros(max_idx + 1, dtype=torch.bool)
        for i in free_experts:
            if i < 0:
                continue
            if i >= len(mask):
                # Extend mask if index is out of current range
                extra = torch.zeros(i - len(mask) + 1, dtype=torch.bool)
                mask = torch.cat([mask, extra], dim=0)
            mask[i] = True
        return mask

    def preprocess(self, images):
        batch = self.processor(images=images, return_tensors="pt")
        return batch["pixel_values"].to(self.device, non_blocking=True)

    @torch.no_grad()
    def _encode(self, expert, pixel_values):
        """
        Run a single expert model on the input and return:
          - cls: [B, hidden_dim] CLS token embeddings.
          - patches: [B, P, hidden_dim] patch embeddings (excluding CLS).
          - attn_pp: [B, P, P] patch-to-patch attention matrix from the last layer.
        """
        output = expert(pixel_values=pixel_values)
        last_hidden = output.last_hidden_state            
        attn = output.attentions[-1]                  
        cls = last_hidden[:, 0, :]                     
        patches = last_hidden[:, 1:, :]                   
        attn_mean = attn.mean(dim=1)                      
        attn_pp = attn_mean[:, 1:, 1:]                  
        return cls, patches, attn_pp

    @staticmethod
    def _eig_from_patches(patches):
        """
        Compute eigenvectors (U) and eigenvalues of the patch covariance for each sample in the batch.
        Returns lists of eigenvector matrices (U_list) and eigenvalues (L_list) for each batch sample.
        """
        B, P, D = patches.shape
        U_list, L_list = [], []
        for b in range(B):
            X = patches[b]                        
            C = X @ X.T                            # [P, P] covariance (Gram) matrix of patches
            eigvals, eigvecs = torch.linalg.eigh(C)  # eigen-decomposition of covariance
            U_list.append(eigvecs)                 # eigenvectors (P x P)
            L_list.append(eigvals)                 # eigenvalues  (P)
        return U_list, L_list

    @staticmethod
    def _diag_energy_score(U, A):
        """
        Compute the alignment score between attention matrix A and eigenbasis U.
        Higher score means A's energy is more aligned with the eigenbasis axes (diagonal in that basis).
        """
        # Project the attention matrix A into the eigenbasis U
        A_proj = U.T @ A @ U 
        diag_elements = torch.diag(A_proj)  # diagonal elements of A_proj
        diag_energy = torch.sum(diag_elements * diag_elements)        # sum of squared diagonal entries
        total_energy = torch.sum(A_proj * A_proj) + 1e-10             # sum of squared all entries (Frobenius norm)
        return diag_energy / total_energy

    @torch.no_grad()
    def route(self, images):
        """
        Compute confidence scores for each expert and decide which experts to activate for each input sample.
        Returns:
          - scores: Tensor of shape [B, E] with confidence scores for each expert.
          - sel_mask: Boolean mask of shape [B, E] indicating selected experts for each sample.
          - cache: List of per-expert tuples (cls, patches, attn_pp) for use in forward pass.
        """
        pixel_values = self.preprocess(images)                    
        B, E = pixel_values.shape[0], self.num_experts

        scores = torch.zeros(B, E, device=self.device)
        cache = []

        # Compute score for each expert
        for e, expert in enumerate(self.experts):
            cls, patches, attn_pp = self._encode(expert, pixel_values)
            # Compute eigenbasis for patch features using covariance
            U_list, _ = self._eig_from_patches(patches)          
            # Compute confidence score for each sample for this expert
            for b in range(B):
                U = U_list[b]    # eigenvectors (P x P) for sample b
                A = attn_pp[b]   # attention matrix (P x P) for sample b
                scores[b, e] = self._diag_energy_score(U, A)
            cache.append((cls, patches, attn_pp))               

        # Determine selected experts based on threshold (Top-D) and top_k (Top-K)
        sel_mask = torch.zeros_like(scores, dtype=torch.bool)
        # List of free expert indices (within valid range) for potential use
        free_indices = [i for i in range(min(E, len(self._free_mask))) if len(self._free_mask) > 0 and self._free_mask[i].item()]
        
        for b in range(B):
            # Experts meeting the confidence threshold
            above_thresh = (scores[b] >= self.threshold).nonzero(as_tuple=True)[0].tolist()
            selected = []  # indices of experts selected for this sample

            if len(above_thresh) > 0:
                # If more than top_k experts exceed threshold, take the top_k highest scores
                if len(above_thresh) > self.top_k:
                    # Sort the above-threshold experts by score (descending) and select top_k
                    above_thresh.sort(key=lambda idx: float(scores[b, idx]), reverse=True)
                    selected = above_thresh[:self.top_k]
                else:
                    # If threshold-qualified experts are <= top_k, select all of them
                    selected = above_thresh.copy()
            # If no expert meets the threshold (or not enough experts selected), we'll fill up using free experts and top scores
            if len(selected) < self.top_k:
                needed = self.top_k - len(selected)
                # Randomly choose from the free experts pool (those not already selected)
                if free_indices:
                    # Filter out any free experts already selected
                    free_pool = [idx for idx in free_indices if idx not in selected]
                    if free_pool:
                        # Determine how many free experts to add (cannot exceed needed or pool size)
                        num_free_to_add = min(needed, len(free_pool))
                        # Randomly select free experts from the pool
                        perm = torch.randperm(len(free_pool), device=self.device)
                        chosen_free = [free_pool[i] for i in perm[:num_free_to_add].tolist()]
                        selected.extend(chosen_free)
                        needed -= num_free_to_add
                # If still need more experts (either no free experts or not enough free experts to fill required slots),
                # select additional experts by highest remaining scores
                if needed > 0:
                    # Get all expert indices sorted by score (desc), and add those not already selected
                    score_order = torch.argsort(scores[b], descending=True).tolist()
                    for idx in score_order:
                        if idx not in selected:
                            selected.append(idx)
                            needed -= 1
                            if needed == 0:
                                break

            # Mark the selected experts in the mask for this sample
            for idx in selected:
                if 0 <= idx < E:
                    sel_mask[b, idx] = True

        return scores, sel_mask, cache

    def forward(self, images):
        """
        Forward pass: routes each image to selected experts and combines their outputs.
        Returns:
          - logits: [B, num_classes] tensor of classification scores.
          - aux: dictionary with auxiliary information (raw scores and selection mask).
        """
        scores, sel_mask, cache = self.route(images)
        B = sel_mask.shape[0]
        num_classes = self.heads[0].out_features

        # Initialize accumulators for weighted sum of logits
        logits_sum = torch.zeros(B, num_classes, device=self.device)
        weight_sum = torch.zeros(B, device=self.device)

        # Compute logits from each expert and accumulate with weight
        for e in range(self.num_experts):
            if sel_mask[:, e].any():
                cls, _, _ = cache[e]                   # CLS embeddings for all samples from expert e
                logits_e = self.heads[e](cls)          # [B, num_classes] logits from expert e
                # Use expert's score as weight for each sample (0 if not selected due to mask)
                weight = (scores[:, e] * sel_mask[:, e].float())  # [B] weight for expert e per sample
                # Add weighted logits (expand weight to shape [B, 1] for broadcasting)
                logits_sum += logits_e * weight.unsqueeze(1)
                weight_sum += weight

        # Normalize by total weight per sample to get weighted average; avoid division by zero with epsilon
        weight_sum = weight_sum.clamp_min(1e-8).unsqueeze(1)  # shape [B, 1]
        logits = logits_sum / weight_sum

        aux = {"scores": scores, "sel_mask": sel_mask}
        return logits, aux