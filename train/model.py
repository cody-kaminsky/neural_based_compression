import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from train.modules.analysis import AnalysisNet
from train.modules.synthesis import SynthesisNet
from train.modules.hyper import HyperAnalysis, HyperSynthesis


class NeuralEncoderModel(nn.Module):
    def __init__(self, lmbda=0.05):
        super().__init__()
        self.lmbda = lmbda
        self.analysis_net = AnalysisNet()
        self.synthesis_net = SynthesisNet()
        self.hyper_analysis = HyperAnalysis()
        self.hyper_synthesis = HyperSynthesis()
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(64)

    def forward(self, x):
        y = self.analysis_net(x)
        z = self.hyper_analysis(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales = self.hyper_synthesis(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales)
        x_hat = self.synthesis_net(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.analysis_net(x)
        z = self.hyper_analysis(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales = self.hyper_synthesis(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        scales = self.hyper_synthesis(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes)
        x_hat = self.synthesis_net(y_hat)
        return {"x_hat": x_hat}

    def strip_encode(self, x, strip_height=64, overlap=2):
        """Process image in horizontal strips; returns list of per-strip forward outputs."""
        _, _, H, W = x.shape
        strips = []
        y_start = 0
        while y_start < H:
            y_end = min(y_start + strip_height, H)
            strip = x[:, :, y_start:y_end, :]
            pad_bottom = strip_height - strip.shape[2]
            if pad_bottom > 0:
                strip = torch.nn.functional.pad(strip, (0, 0, 0, pad_bottom))
            out = self.forward(strip)
            strips.append(out)
            y_start += strip_height - overlap
        return strips
