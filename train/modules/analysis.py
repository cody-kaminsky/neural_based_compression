import torch.nn as nn
from .dsconv import ConvBlock


class AnalysisNet(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.stage0 = ConvBlock(3, N, stride=2)
        self.stage1 = ConvBlock(N, N, stride=2)
        # Final stage has no GDN — Ballé 2018 convention; lets latent magnitude
        # be set by the rate-distortion objective rather than a fixed normalizer.
        self.final = nn.Conv2d(N, M, 5, stride=2, padding=2)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        return self.final(x)
