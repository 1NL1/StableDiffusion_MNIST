"""
Dans ce fichier on code l'encodeur, qui va compresser nos images dans un espace latent.
Rendre les images plus petites aidera pour l'entrainement et l'inférence.

On utilise un VAE (variational auto-encoder) pour que même dans l'espace latent les images qui ont une forme similaire aient une représentation similaire.
"""

import torch
from torch import nn
from torch.nn import functional as F

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
            #On fait le padding à part car on en veut un qui soit asymétrique: que à droite et en bas
            lambda x : F.pad(x, (0,1,0,1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),

            #B, 256, H/4, W/4 -> B, 512, H/4, W/4
            VAE_ResidualBlock(256,512),

            #B, 512, H/4, W/4 -> B, 512, H/4, W/4
            VAE_ResidualBlock(512,512),

            #B, 512, H/4, W/4 -> B, 512, H/8, W/8
            lambda x : F.pad(x, (0,1,0,1)),
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