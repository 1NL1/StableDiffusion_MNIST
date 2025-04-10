"""
Dans ce fichier on implémente le modèle de diffusion, ie le Unet lui-même
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import selfAttention, crossAttention


class SwitchSequential(nn.Sequential):
    """
    Transformer une liste de couches en SwitchSequential permet de les appliquer une a une avec les bons arguments 
    """

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, Unet_attention):
                x = layer(x, context)

            elif isinstance(layer, Unet_residualBlock):
                x = layer(x, time)
            
            else:
                x = layer(x)
        
        return x


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #B, C, H, W -> B, C, H*2, W*2
        x = F.interpolate(x, scale_factor=2, mode = "nearest")
        x = self.conv(x)
        return x


class Unet_residualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_time: int = 1280):
        super().__init__()
        self.groupNorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.linear_time = nn.Linear(d_time, out_channels)

        self.groupNorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residualLayer = nn.Identity()
        else:
            self.residualLayer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, time):
        #x: B, in_channels, H, W
        #time: 1, 1280

        residue = x
        residue = self.residualLayer(residue)

        x = self.groupNorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        time = F.silu(time)
        time = self.linear_time(time)
        
        merge = x + time.unsqueeze(-1).unsqueeze(-1)

        merge = self.groupNorm2(merge)
        merge = F.silu(merge)
        merge = self.conv2(merge)

        return x + residue

class Unet_attentionBlock(nn.Module):
    def __init__(self, n_heads: int, d_emb: int, d_context = 768):
        super().__init__()
        channels = n_heads * d_emb

        self.groupNorm = nn.GroupNorm(32, channels, eps = 1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.layerNorm1 = nn.LayerNorm(channels)
        self.attention1 = selfAttention(n_heads, channels, in_proj_bias = False)
        self.layerNorm2 = nn.LayerNorm(channels)
        self.attention2 = crossAttention(n_heads, channels, d_context, in_proj_bias = False)
        self.layerNorm3 = nn.LayerNorm(channels)
        self.linear_geglu1 = nn.Linear(channels, 4 * channels * 2) #feed forward
        self.linear_geglu2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, context):
        #x: B, C, H, W
        #context: B, Seq_len, dim=768

        residue_long = x

        x = self.groupNorm(x)
        x = self.conv_input(x)

        b, c, h, w = x.shape

        #B, C, H, W -> B, C, H*W
        x = x.view((b,c,h*w))

        #B, C, H*W -> B, H*W, C
        x = x.transpose(-1, -2)

        #Normalization + SelfAttention + skip connection
        residue_short = x

        x = self.layerNorm1(x)
        x = self.attention1(x)
        x = x + residue_short

        #Normalisation + CrossAttention + skip connection
        residue_short = x

        x = self.layerNorm2(x)
        x = self.attention2(x, context)
        x = x + residue_short

        #Normalisation + feed forward (Geglu) + skip connection
        residue_short = x

        x = self.layerNorm3(x)
        x, gate = self.linear_geglu1(x).chunk(2, dim = -1)
        x = x * F.gelu(gate)

        x = self.linear_geglu2(x)

        x = x + residue_short

        #reshape pour retrouver la forme originale
        #B, H*W, C -> B, C, H*W
        x = x.transpose(-1, -2)
        
        # B, C, H*W -> B, C, H, W
        x = x.view(b, c, h, w)

        #Output
        x = self.conv_output(x)
        x = x + residue_long

        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoderLayers = nn.Module([
            #B, 4, H/8, W/8 -> B, 320, H/8, W/8
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)), #conv en place, pas de downsample
            SwitchSequential(Unet_residualBlock(320, 320), Unet_attentionBlock(8, 40)),            
            SwitchSequential(Unet_residualBlock(320, 320), Unet_attentionBlock(8, 40)),
            
            #B, 320, H/8, W/8 -> B, 640, H/16, W/16
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, padding=1, stride = 2)), #DownSample
            SwitchSequential(Unet_residualBlock(320, 640), Unet_attentionBlock(8, 80)),
            SwitchSequential(Unet_residualBlock(640, 640), Unet_attentionBlock(8, 80)),

            #B, 640, H/16, W/16 -> B, 1280, H/32, W/32
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, padding=1, stride = 2)), #DownSample
            SwitchSequential(Unet_residualBlock(640, 1280), Unet_attentionBlock(8, 160)),
            SwitchSequential(Unet_residualBlock(1280 , 1280), Unet_attentionBlock(8, 160)),

            #B, 1280, H/32, W/32 -> B, 1280, H/64, W/64
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, padding=1, stride = 2)), #DownSample

            #B, 1280, H/64, W/64
            SwitchSequential(Unet_residualBlock(1280, 1280)),
            SwitchSequential(Unet_residualBlock(1280, 1280))
        ])

        self.bottleNeck = SwitchSequential(
            Unet_residualBlock(1280, 1280),
            Unet_attentionBlock(8, 160),
            Unet_residualBlock(1280, 1280)
        )

        self.decoderLayers = nn.Module([
            #B, 2560, H/64, W/64 -> B, 1280, H/64, W/64
            SwitchSequential(Unet_residualBlock(2560, 1280)),

            SwitchSequential(Unet_residualBlock(2560, 1280)),
             
            SwitchSequential(Unet_residualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(Unet_residualBlock(2560, 1280), Unet_attentionBlock(8, 160)),

            SwitchSequential(Unet_residualBlock(2560, 1280), Unet_attentionBlock(8, 160)),

            SwitchSequential(Unet_residualBlock(1920, 1280), Unet_attentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(Unet_residualBlock(1920, 640), Unet_attentionBlock(8, 80)),

            SwitchSequential(Unet_residualBlock(1280, 640), Unet_attentionBlock(8, 80)),

            SwitchSequential(Unet_residualBlock(960, 640), Unet_attentionBlock(8, 80), UpSample(640)),
 
            SwitchSequential(Unet_residualBlock(960, 320), Unet_attentionBlock(8, 40)),

            SwitchSequential(Unet_residualBlock(640, 320), Unet_attentionBlock(8, 80)), #???pk pas 40?

            SwitchSequential(Unet_residualBlock(640, 320), Unet_attentionBlock(8, 40)),

        ])


class TimeEmbedding(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.linear1 = nn.Linear(d_emb, 4 * d_emb)
        self.linear2 = nn.Linear(4 * d_emb, 4 * d_emb)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        #1, 320 -> 1, 1280
        time = self.linear1(time)

        time = F.silu(time)

        #1, 1280 -> 1, 1280
        time = self.linear2(time)

        return time

class Unet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupNorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        #B, 320, H/8, W/8
        x = self.groupNorm(x)

        x = nn.SiLU(x)
        
        x = self.conv(x)

        #B, 4, H/8, W/8
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = Unet()
        self.final = Unet_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        #latent: B, 4, H/8, W/8
        #context: B, seq_len, d_model
        #time: (1, 320)
        
        # 1, 320 -> 1, 1280 
        time_embedding = self.time_embedding(time)
        
        #B, 4, H/8, W/8 -> B, 320, H/8, W/8
        unet_output = self.unet(latent, context, time_embedding)
        
        #B, 320, H/8, W/8 -> B, 4, H/8, W/8
        output = self.final(unet_output)

        #B, 4, H/8, W/8        
        return output