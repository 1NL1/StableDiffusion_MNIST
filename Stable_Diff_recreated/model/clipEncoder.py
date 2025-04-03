"""
Dans ce fichier on code le CLIP encoder, la partie qui va permettre de prendre en compte la consigne texte lors de la création.
C'est grosso-modo la partie encodeur d'un transformer
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import selfAttention

class CLIPEmbedding(nn.Module):
    #comprend l'embedding et le positionnal encoding
    def __init__(self, v_size: int, emb_size: int, seq_len: int): #seq_len = nb max de tokens (padding)
        super().__init__()
        self.token_embedding = nn.Embedding(v_size, emb_size)

        #Au lieu de la fonction sinusoidale utilisée pour les transfos, on prend pour l'embedding des poids préconçus
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, emb_size))
    
    def forward(self, tokens: torch.Tensor):
        
        #B, seq_len -> B, seq_len, emb_size
        emb_tokens = self.token_embedding(tokens)

        #B, seq_len, emb_size -> B, seq_len, emb_size
        emb_pos_tokens = emb_tokens + self.position_embedding
        
        return emb_pos_tokens


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super().__init__()
        self.attentionBlock = selfAttention(n_heads, d_model)
        self.addAndNorm = lambda x, y : nn.layer_norm(x+y)
        self.feedForward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            lambda x : x * torch.sigmoid(1.702 * x),  #quick GELU
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #B, seq_len, d_model
        y = self.attentionBlock(x, apply_mask = True)
        a = self.addAndNorm(x, y)
        b = self.feedForward(a)
        output = self.addAndNorm(a, b)
        
        return output

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(v_size = 49408, emb_size = 768, seq_len = 77)

        self.layers = nn.Module([
            CLIPLayer(n_heads = 12, d_model = 768) for _ in range(12)
        ])

        self.layerNorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        #embedding
        #B, seq_len -> B, seq_len, emb_dim
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        #B, seq_len, emb_dim
        output = self.layerNorm(state)
        return output