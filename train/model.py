import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from train.modules.analysis import AnalysisNet
from train.modules.synthesis import SynthesisNet
from train.modules.hyper import HyperAnalysis, HyperSynthesis


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradient: round() in forward, identity in backward."""
    return x + (x.round() - x).detach()


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
        h, w = x.shape[2], x.shape[3]
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        y = self.analysis_net(x)
        z = self.hyper_analysis(y)

        # Get likelihoods via standard CompressAI path (noise in training, round in eval).
        # We call these just for the likelihood values — z_hat and y_hat below are
        # computed separately with STE so the synthesis always sees integer-valued latents.
        _, z_likelihoods = self.entropy_bottleneck(z)

        # STE z_hat: training and eval both use rounded integers for hyper_synthesis.
        # This makes the scales (and therefore y_likelihoods) consistent across modes.
        z_hat = _ste_round(z)
        scales = self.hyper_synthesis(z_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales)

        # STE y_hat for synthesis: forces the synthesis net to decode from rounded
        # integers in both training and eval. Without this, training uses noisy y
        # (which can be decoded even when y≈0), while eval uses round(y)=0 → collapse.
        y_hat = _ste_round(y)
        x_hat = self.synthesis_net(y_hat)
        x_hat = x_hat[:, :, :h, :w]

        return {
            "x_hat": x_hat,
            "y_likelihoods": y_likelihoods,
            "z_likelihoods": z_likelihoods,
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
