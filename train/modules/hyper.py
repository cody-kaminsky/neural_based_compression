import torch.nn as nn


class HyperAnalysis(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, y):
        return self.net(y.abs())


class HyperSynthesis(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Softplus(),
        )

    def forward(self, z_hat):
        return self.net(z_hat)
