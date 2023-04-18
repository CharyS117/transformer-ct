import torch.nn as nn
from typing import List
from arch import base


class FeedForward(base.FeedForward):
    def __init__(self, channels: int):
        super().__init__(channels)
        # change kernel_size from 1 to 3
        self.conv_in = nn.Conv2d(channels, channels*3, kernel_size=3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(channels*3, channels, kernel_size=3, padding=1, bias=False)


class Transformer(base.Transformer):
    def __init__(self, channels: int):
        super().__init__(channels)
        self.feed_forward = FeedForward(channels)


class Net(base.Net):
    def __init__(self, features: int, blocks: List[int]):
        super().__init__(features, blocks)
        self.e_transformers = nn.ModuleList([nn.Sequential(*[Transformer(features*2**i) for _ in range(blocks[i])]) for i in range(self.depth-1)])
        self.d_transformers = nn.ModuleList([nn.Sequential(*[Transformer(features*2**i) for _ in range(blocks[i])]) for i in range(self.depth-1)])
        self.middle_transformer = nn.Sequential(*[Transformer(features*2**(self.depth-1)) for _ in range(blocks[-1])])
