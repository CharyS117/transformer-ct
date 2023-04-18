import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.qkv_w = nn.Parameter(torch.ones((3, channels, 1)))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor):
        shape = x.shape
        # (n, c, h, w) -> (n, c, h*w)
        x = x.view(x.shape[0], x.shape[1], -1)
        q = x * self.qkv_w[0]
        k = x * self.qkv_w[1]
        v = x * self.qkv_w[2]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        mix_weight = F.softmax(q @ k.transpose(-1, -2), dim=-1)
        mixed = mix_weight @ v
        # (n, c, h*w) -> (n, c, h, w)
        mixed = mixed.view(shape)
        output = self.conv(mixed)
        return output


class FeedForward(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, channels*3, kernel_size=1, bias=False)
        self.conv_out = nn.Conv2d(channels*3, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        x = F.gelu(x)
        x = self.conv_out(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((channels, 1, 1)))
        
    def forward(self, x: Tensor):
        x = F.layer_norm(x, (x.shape[-2], x.shape[-1])) * self.weight
        return x
 

class Transformer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.self_attention = SelfAttention(channels)
        self.feed_forward = FeedForward(channels)
        self.layer_norms = nn.ModuleList([LayerNorm(channels) for _ in range(2)])

    def forward(self, x: Tensor):
        x = self.layer_norms[0](x)
        x = x + self.self_attention(x)
        x = self.layer_norms[1](x)
        x = x + self.feed_forward(x)
        return x


class Downsample(nn.Module):
    """
    input shape: (n, c, h, w)
    output shape: (n, c*2, h//2, w//2)
    """
    def __init__(self, channels: int):
        super().__init__()
        self._2c_c = nn.Conv2d(channels*4, channels*2, kernel_size=3, padding=1, bias=False)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x: Tensor):
        x = self.pixel_unshuffle(x)
        x = self._2c_c(x)
        return x
    

class Upsample(nn.Module):
    """
    input shape: (n, c, h, w)
    output shape: (n, c//2, h*2, w*2)
    """
    def __init__(self, channels: int):
        super().__init__()
        self._c_2c = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor):
        x = self._c_2c(x)
        x = self.pixel_shuffle(x)
        return x


class Net(nn.Module):
    def __init__(self, features: int, blocks: List[int]):
        super().__init__()
        # feature extraction
        self.feature_extract = nn.Conv2d(1, features, kernel_size=3, padding=1, bias=False)
        self.feature_merge = nn.Conv2d(features, 1, kernel_size=3, padding=1, bias=False)
        # encoder and decoder
        self.depth = len(blocks)
        self.e_transformers = nn.ModuleList([nn.Sequential(*[Transformer(features*2**i) for _ in range(blocks[i])]) for i in range(self.depth-1)])
        self.d_transformers = nn.ModuleList([nn.Sequential(*[Transformer(features*2**i) for _ in range(blocks[i])]) for i in range(self.depth-1)])
        self.middle_transformer = nn.Sequential(*[Transformer(features*2**(self.depth-1)) for _ in range(blocks[-1])])
        self.downsamples = nn.ModuleList([Downsample(features*2**i) for i in range(self.depth-1)])
        self.upsamples = nn.ModuleList([Upsample(features*2**(i+1)) for i in range(self.depth-1)])
        self.skip_connections = nn.ModuleList([nn.Conv2d(features*2**(i+1), features*2**i, kernel_size=3, padding=1, bias=False) for i in range(self.depth-1)])

    def forward(self, x: Tensor):
        x_res = x
        x = self.feature_extract(x)
        # encode
        encoder_out = []
        for i in range(self.depth-1):
            x = self.e_transformers[i](x)
            encoder_out.append(x)
            x = self.downsamples[i](x)
        # middle
        x = self.middle_transformer(x)
        # decode
        for i in range(self.depth-2, -1, -1):
            x = self.upsamples[i](x)
            x = self.skip_connections[i](torch.cat([x, encoder_out[i]], dim=1))
            x = self.d_transformers[i](x)
        x = x_res + self.feature_merge(x)
        return x
