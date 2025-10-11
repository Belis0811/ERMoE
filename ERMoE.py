import torch
import torch.nn as nn
from copy import deepcopy
from transformers import ViTModel, ViTImageProcessor
num_experts = 8                
top_k = 2                          
free_experts = [0]               
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_base = ViTModel.from_pretrained(
    "google/vit-base-patch16-224",
    output_attentions=True,
    output_hidden_states=False
).to(device)
vit_base.eval()
experts = nn.ModuleList([deepcopy(vit_base) for _ in range(num_experts)])
for expert in experts:
    for param in expert.parameters():
        param.requires_grad = False
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def extract_features_and_attention(model, images):
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state           
    attentions = outputs.attentions[-1]              

    patch_feats = last_hidden[:, 1:, :]             
    attn_mean = attentions.mean(dim=1)            
    attn_matrix = attn_mean[:, 1:, 1:]            
    return patch_feats, attn_matrix

def compute_covariance_eig(patch_feats):
    batch_size, P, D = patch_feats.shape
    U_list = []
    Lambda_list = []
    for b in range(batch_size):
        X = patch_feats[b]                     
        C = X @ X.T                               

        eigvals, eigvecs = torch.linalg.eigh(C)   
        U_list.append(eigvecs)                 
        Lambda_list.append(eigvals)          
    return U_list, Lambda_list

def compute_attention_score(U, attn):
    A_proj = U.T @ attn @ U  
    diag = torch.diag(A_proj)                  
    diag_energy = torch.sum(diag**2)           
    total_energy = torch.sum(A_proj**2)         
    score = diag_energy / (total_energy + 1e-10)
    return score

def select_topk_experts(scores, top_k, free_experts):
    batch_size, num_experts = scores.shape
    selected_mask = torch.zeros_like(scores, dtype=torch.bool)
    for b in range(batch_size):
        selected = set(free_experts)
        top_indices = torch.argsort(scores[b], descending=True)
        for idx in top_indices:
            idx = idx.item()
            if len(selected) >= top_k + len(free_experts):
                break
            if idx not in selected:
                selected.add(idx)
        for idx in selected:
            if idx < num_experts:
                selected_mask[b, idx] = True
    return selected_mask

def moE_router_forward(images):

    batch_size = len(images)
    scores_tensor = torch.zeros(batch_size, num_experts, device=device)
    

    for i, expert in enumerate(experts):
        patch_feats, attn_matrix = extract_features_and_attention(expert, images)
        U_list, _ = compute_covariance_eig(patch_feats)  
        for b in range(batch_size):
            U = U_list[b]           # [P, P]
            A = attn_matrix[b]      # [P, P]
            score = compute_attention_score(U, A)
            scores_tensor[b, i] = score
    selected_mask = select_topk_experts(scores_tensor, top_k, free_experts)
    return scores_tensor, selected_mask
