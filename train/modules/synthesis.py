import torch.nn as nn
from .dsconv import DSConvTransposeBlock


class SynthesisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = DSConvTransposeBlock(128, 96, stride=2)
        self.stage1 = DSConvTransposeBlock(96, 64, stride=2)
        self.stage2 = DSConvTransposeBlock(64, 3, stride=2)
        self.final = nn.Sigmoid()

    def forward(self, y_hat):
        x = self.stage0(y_hat)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.final(x)
