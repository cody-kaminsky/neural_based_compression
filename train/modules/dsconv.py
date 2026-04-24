import torch.nn as nn


class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else None

    def forward(self, x):
        x = self.act1(self.depthwise(x))
        x = self.act2(self.pointwise(x))
        if self.pool is not None:
            x = self.pool(x)
        return x


class DSConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        if stride == 2:
            self.depthwise = nn.ConvTranspose2d(
                out_ch, out_ch, 3, stride=2, padding=1, output_padding=1, groups=out_ch
            )
        else:
            self.depthwise = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.act1(self.pointwise(x))
        x = self.act2(self.depthwise(x))
        return x
