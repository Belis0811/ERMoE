import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TransformerBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        B, N, E = x.shape
        x_ln = self.norm1(x)
        qkv = self.qkv(x_ln)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ v)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, E)
        x = x + self.proj(attn_output)

        x_ln = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(x_ln)))
        return x, attn_weights


class VisionTransformer3D(nn.Module):
    def __init__(self, img_size=(91, 109, 91), patch_size=(7, 7, 7), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        H, W, D = img_size;
        pH, pW, pD = patch_size
        self.num_patches = (H // pH) * (W // pW) * (D // pD)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock3D(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x, return_attn=False):
        B = x.size(0)

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        attn_weights = None
        for blk in self.blocks:
            x, attn = blk(x)
            attn_weights = attn
        x = self.norm(x)
        if return_attn:
            attn_weights = attn_weights[:, :, 1:, 1:]
            attn_avg = attn_weights.mean(dim=1)
            return x, attn_avg
        return x