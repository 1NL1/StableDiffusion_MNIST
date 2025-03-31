import torch
from torch import nn
from torch.nn import functional as F
import math

class selfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed:int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, out_proj_bias)
        self.n_heads = n_heads
        assert d_embed % n_heads == 0
        self.d_heads = d_embed // n_heads

    def forward(self, x:torch.Tensor, apply_mask = False) -> torch.Tensor:
        #x: B, Seq_len, dim
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        k,q,v = self.in_proj(x).chunk(3, dim = -1)

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
            mask *= -torch.inf
            w += mask

        w /= math.sqrt(self.d_heads)

        w = F.softmax(w, dim = -1)

        #B, H, seq_len, dim/H
        score = w @ v

        #B, seq_len, H, dim/H
        score = score.transpose(1, 2)

        score = score.reshape(input_shape)

        output = self.out_proj(score)

        return output