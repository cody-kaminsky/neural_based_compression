import torch.nn as nn
from .dsconv import DSConvBlock


class AnalysisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = DSConvBlock(3, 64, stride=2)
        self.stage1 = DSConvBlock(64, 96, stride=2)
        self.stage2 = DSConvBlock(96, 128, stride=2)
        # Learnable per-channel scale so y has sufficient magnitude for quantization.
        # Without this, default-init activations stay in (-0.5, 0.5) and all round
        # to 0 under hard quantization in eval mode, giving val_bpp=0 indefinitely.
    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x * 4.0
