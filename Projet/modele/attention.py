import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_emb:int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.in_proj = nn.Linear(d_emb, 3*d_emb, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_emb, d_emb, bias = out_proj_bias)
        self.n_heads = n_heads
        assert d_emb % n_heads == 0
        self.d_heads = d_emb // n_heads

    def forward(self, x:torch.Tensor, apply_mask = False) -> torch.Tensor:
        #x: B, Seq_len, dim
        input_shape = x.shape
        batch_size, seq_len, d_emb = input_shape

        q,k,v = self.in_proj(x).chunk(3, dim = -1)

        #taille des matrices "coupées" en prenant en compte les têtes
        interm_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

        #B, seq_len, dim -> b, seq_len, H, dim/H -> B, H, seq_len, dim/H
        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        #B, H, seq_len, seq_len
        w = q @ k.transpose(-1,-2)

        if apply_mask:
            mask = torch.ones_like(w, dtype = torch.bool)
            mask = mask.triu(1) #.triu retourne la partie triangulaire supérieure 
            w.masked_fill_(mask, -torch.inf)

        w /= math.sqrt(self.d_heads)

        w = F.softmax(w, dim = -1)

        #B, H, seq_len, dim/H
        score = w @ v

        #B, seq_len, H, dim/H
        score = score.transpose(1, 2)

        score = score.reshape(input_shape)

        output = self.out_proj(score)

        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_emb: int, d_cross: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.q_proj   = nn.Linear(d_emb, d_emb, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_emb, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_emb, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_emb, d_emb, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_emb // n_heads
    
    def forward(self, x, y):
        # x (latent): # (B, Seq_Len_Q, Dim_Q)
        # y (context): # (B, Seq_Len_KV, Dim_KV)

        input_shape = x.shape
        batch_size, sequence_length, d_emb = input_shape
        # Forme en prenant les heads en compte
        interm_shape = (batch_size, -1, self.n_heads, self.d_heads)
        
        # (B, Seq_Len_Q, Dim_Q) -> (B, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (B, Seq_Len_KV, Dim_KV) -> (B, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (B, Seq_Len_KV, Dim_KV) -> (B, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (B, Seq_Len_Q, Dim_Q) -> (B, Seq_Len_Q, H, Dim_Q / H) -> (B, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interm_shape).transpose(1, 2) 
        # (B, Seq_Len_KV, Dim_Q) -> (B, Seq_Len_KV, H, Dim_Q / H) -> (B, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interm_shape).transpose(1, 2) 
        # (B, Seq_Len_KV, Dim_Q) -> (B, Seq_Len_KV, H, Dim_Q / H) -> (B, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interm_shape).transpose(1, 2) 
        
        # (B, H, Seq_Len_Q, Dim_Q / H) @ (B, H, Dim_Q / H, Seq_Len_KV) -> (B, H, Seq_Len_Q, Seq_Len_KV)
        w = q @ k.transpose(-1, -2)
        
        # (B, H, Seq_Len_Q, Seq_Len_KV)
        w /= math.sqrt(self.d_heads)
        
        # (B, H, Seq_Len_Q, Seq_Len_KV)
        w = F.softmax(w, dim=-1)
        
        # (B, H, Seq_Len_Q, Seq_Len_KV) @ (B, H, Seq_Len_KV, Dim_Q / H) -> (B, H, Seq_Len_Q, Dim_Q / H)
        output = w @ v
        
        # (B, H, Seq_Len_Q, Dim_Q / H) -> (B, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (B, Seq_Len_Q, H, Dim_Q / H) -> (B, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (B, Seq_Len_Q, Dim_Q) -> (B, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (B, Seq_Len_Q, Dim_Q)
        return output