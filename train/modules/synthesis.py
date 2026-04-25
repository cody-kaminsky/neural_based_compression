import torch.nn as nn
from .dsconv import ConvTransposeBlock


class SynthesisNet(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.stage0 = ConvTransposeBlock(M, N, stride=2)
        self.stage1 = ConvTransposeBlock(N, N, stride=2)
        # Final stage outputs RGB directly (no IGDN), then sigmoid → [0,1].
        self.final = nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1)
        self.act = nn.Sigmoid()

    def forward(self, y_hat):
        x = self.stage0(y_hat)
        x = self.stage1(x)
        x = self.final(x)
        return self.act(x)
