"""
Dans ce fichier on code l'encodeur et le décodeur, qui vont respectivement compresser et décompresser nos images dans/depuis un espace latent.
Rendre les images plus petites aidera pour l'entrainement et l'inférence.

On utilise un VAE (variational auto-encoder) pour que même dans l'espace latent les images qui ont une forme similaire aient une représentation similaire.
Un VAE se distingue d'un auto-encodeur classique car il encode les données comme une distribution continue dans l'espace latent.
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
            self.residualLayer = nn.Identity()
        else:
            self.residualLayer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x : B, in_channels, H, W

        residue_x = self.residualLayer(x)

        x = self.groupNorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupNorm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + residue_x

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        print(channels)
        self.groupNorm = nn.GroupNorm(32, channels)
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



class VAE_Encoder(nn.Sequential):
    """
    L'encodeur du VAE (variational auto-encodeur)
    La ou un auto-encodeur classique apprend à transformer une image dans un espace latent, un variational apprend l'espace lattent lui-même qui est un
    espace de distributions. Ainsi l'encodage d'une image est un echantillonage depuis la distribution apprise par l'espace.
    """
    def __init__(self):
        super().__init__(
            #B, C=3, H, W -> B, 128, H, W
            nn.Conv2d(3, 128, kernel_size=3, padding = 1),

            #B, 128, H, W -> B, 128, H, W
            VAE_ResidualBlock(128,128),

            #B, 128, H, W -> B, 128, H, W
            VAE_ResidualBlock(128,128),
        
            #B, 128, H, W -> B, 128, H/2, W/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2),

            #B, 128, H/2, W/2 -> B, 256, H/2, W/2
            VAE_ResidualBlock(128,256),

            #B, 256, H/2, W/2 -> B, 256, H/2, W/2
            VAE_ResidualBlock(256,256),

            #B, 256, H/2, W/2 -> B, 256, H/4, W/4
            nn.Conv2d(256, 256, kernel_size=3, stride=2),

            #B, 256, H/4, W/4 -> B, 512, H/4, W/4
            VAE_ResidualBlock(256,512),

            #B, 512, H/4, W/4 -> B, 512, H/4, W/4
            VAE_ResidualBlock(512,512),

            #B, 512, H/4, W/4 -> B, 512, H/8, W/8
            nn.Conv2d(512, 512, kernel_size=3, stride=2),            
        
            #B, 512, H/4, W/4 -> B, 512, H/4, W/4
            VAE_ResidualBlock(512,512),

            #B, 512, H/4, W/4 -> B, 512, H/4, W/4
            VAE_ResidualBlock(512,512),

            #B, 512, H/4, W/4 -> B, 512, H/4, W/4
            VAE_ResidualBlock(512,512),
            
            #B, 512, H/4, W/4 -> B, 512, H/4, W/4
            VAE_AttentionBlock(512),

            #B, 512, H/4, W/4 -> B, 512, H/4, W/4
            VAE_ResidualBlock(512,512),

            #B, 512, H/4, W/4 -> B, 512, H/4, W/4
            nn.GroupNorm(32, 512),

            #Activation
            nn.SiLU(),

            #B, 512, H/4, W/4 -> B, 8, H/8, W/8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            #B, 8, H/4, W/4 -> B, 8, H/8, W/8
            nn.Conv2d(8, 8, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x: B, C, H, W
        noise: B, output_channels, H/8, W/8. le bruit suit une loi gaussienne N(0,1)
        """
        for couche in self:    
            if getattr(module, 'stride', None) == (2, 2):  # On veut un padding asymetrique
                x = F.pad(x, (0, 1, 0, 1))       
            x = couche(x)

        #B, 8, H/8, W/8 -> B, 4, H/8, W/8 + B, 4, H/8, W/8
        mean, log_variance = torch.chunk(x, 2, dim = 1)

        #Si les valeurs de la log_variance sont trop petites ou trop grandes, on les ramène dans une range acceptable
        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        ecartType = variance.sqrt()
        
        #N(0,1)     -->     N(mu, teta^2)
        #x          -->     mu + teta * x
        x = mean + ecartType * noise

        #Scale x: la constante est assez arbitraire et vient du papier
        x *= 0.18215

        return x



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
        
        #On annule le scaling de l'encodeur
        x /= 0.18215

        for layer in self:
            x = layer(x)
        
        #B, 3, H, W
        return x