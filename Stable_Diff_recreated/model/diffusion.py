"""
Dans ce fichier on implémente le modèle de diffusion, ie le Unet lui-même
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import selfAttention, crossAttention

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        