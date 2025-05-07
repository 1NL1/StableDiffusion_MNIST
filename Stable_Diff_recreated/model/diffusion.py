"""
Dans ce fichier on implémente le modèle de diffusion, ie le Unet lui-même
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class SwitchSequential(nn.Sequential):
    """
    Transformer une liste de couches en SwitchSequential permet de les appliquer une a une avec les bons arguments 
    """

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, Unet_attentionBlock):
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
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.linear_time = nn.Linear(d_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding = 0)
    
    def forward(self, x, time):
        #x: B, in_channels, H, W
        #time: 1, 1280

        residue = x
        residue = self.residual_layer(residue)

        x = self.groupnorm_feature(x)
        x = F.silu(x)
        x = self.conv_feature(x)
        
        time = F.silu(time)
        time = self.linear_time(time)
        
        merge = x + time.unsqueeze(-1).unsqueeze(-1)

        merge = self.groupnorm_merged(merge)
        merge = F.silu(merge)
        merge = self.conv_merged(merge)

        return merge + residue

class Unet_attentionBlock(nn.Module):
    def __init__(self, n_heads: int, d_emb: int, d_context = 768):
        super().__init__()
        channels = n_heads * d_emb

        self.groupnorm = nn.GroupNorm(32, channels, eps = 1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding = 0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias = False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias = False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2) #feed forward
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding = 0)

    def forward(self, x, context):
        #x: B, C, H, W
        #context: B, Seq_len, dim=768

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        b, c, h, w = x.shape

        #B, C, H, W -> B, C, H*W
        x = x.view((b,c,h*w))

        #B, C, H*W -> B, H*W, C
        x = x.transpose(-1, -2)

        #Normalization + SelfAttention + skip connection
        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_short

        #Normalisation + CrossAttention + skip connection
        residue_short = x

        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x = x + residue_short

        #Normalisation + feed forward (Geglu) + skip connection
        residue_short = x

        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim = -1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

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

        self.encoders = nn.ModuleList([
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
            SwitchSequential(Unet_residualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            Unet_residualBlock(1280, 1280),
            Unet_attentionBlock(8, 160),
            Unet_residualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
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

            SwitchSequential(Unet_residualBlock(640, 320), Unet_attentionBlock(8, 40)),

            SwitchSequential(Unet_residualBlock(640, 320), Unet_attentionBlock(8, 40)),

        ])

    def forward(self, x, context, time):
        # x: (B, 4, H / 8, W / 8)
        # context: (B, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoders:
            # On ajoute la skip connection avant de passer x dans la couche de decodeur
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layer(x, context, time)
        
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.linear_1 = nn.Linear(d_emb, 4 * d_emb)
        self.linear_2 = nn.Linear(4 * d_emb, 4 * d_emb)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        #1, 320 -> 1, 1280
        time = self.linear_1(time)

        time = F.silu(time)

        #1, 1280 -> 1, 1280
        time = self.linear_2(time)

        return time

class Unet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        #B, 320, H/8, W/8
        x = self.groupnorm(x)

        x = F.silu(x)
        
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