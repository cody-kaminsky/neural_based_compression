import torch.nn as nn
from compressai.layers import GDN


class ConvBlock(nn.Module):
    """Strided Conv2d + GDN, used in analysis transform."""
    def __init__(self, in_ch, out_ch, stride=2, kernel=5):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad)
        self.gdn = GDN(out_ch)

    def forward(self, x):
        return self.gdn(self.conv(x))


class ConvTransposeBlock(nn.Module):
    """ConvTranspose2d + IGDN, used in synthesis transform."""
    def __init__(self, in_ch, out_ch, stride=2, kernel=5):
        super().__init__()
        pad = kernel // 2
        if stride == 2:
            self.conv = nn.ConvTranspose2d(
                in_ch, out_ch, kernel, stride=2, padding=pad, output_padding=1
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=pad)
        self.igdn = GDN(out_ch, inverse=True)

    def forward(self, x):
        return self.igdn(self.conv(x))
