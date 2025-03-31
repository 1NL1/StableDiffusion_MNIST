"""
Dans ce fichier on code le décodeur, qui va prendre les vecteurs latents et les retransformer en images. C'est le réciproque de l'encodeur.
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import selfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupNorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1)
        self.groupNorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x : B, in_channels, H, W

        residue_x = self.residual_layer(x)

        x = self.groupNorm1(x)
        x = nn.SiLU(x)
        x = self.conv1(x)
        x = self.groupNorm2(x)
        x = nn.SiLU(x)
        x = self.conv2(x)
        
        return x + residue_x

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupNorm = nn.GroupNorm(channels)
        self.attention = selfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: B, C, H, W
        residue = x

        b,c,h,w = x.shape

        #B, C, H, W -> B, C, H*W
        x = x.view(b,c,h*w)
    
        #B,C,H*W -> B,H*W, C
        x = x.transpose(-1, -2)

        #B,H*W, C -> B,H*W, C
        x = self.attention(x)

        #B,H*W, C -> B, C, H*W
        x = x.transpose(-1, -2)

        #B, C, H*W -> B, C, H, W
        x = x.view(b,c,h,w)

        return x + residue


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            #B, 512, H/8, W/8
            VAE_ResidualBlock(512, 512),

            #B, 512, H/8, W/8 -> B, 512, H/4, W/4
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            #B, 512, H/4, W/4 -> B, 512, H/2, W/2
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding = 1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            #B, 256, H/2, W/2 -> B, 256, H, W
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            #B, 128, H, W -> B, 3, H, W
            nn.Conv2d(128, 3, kernel_size=3, padding = 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: B, 4, H/8, W/8
        x /= 0.18215

        for layer in self:
            x = layer(x)
        
        #B, 3, H, W
        return x