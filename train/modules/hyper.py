import torch.nn as nn


class HyperAnalysis(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(M, N, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, y):
        return self.net(y.abs())


class HyperSynthesis(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(N, N, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, M, 3, padding=1),
            nn.Softplus(),
        )

    def forward(self, z_hat):
        return self.net(z_hat)
