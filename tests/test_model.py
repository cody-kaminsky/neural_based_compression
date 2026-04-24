import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from train.modules.dsconv import DSConvBlock, DSConvTransposeBlock
from train.modules.analysis import AnalysisNet
from train.modules.synthesis import SynthesisNet
from train.modules.hyper import HyperAnalysis, HyperSynthesis
from train.model import NeuralEncoderModel


def test_dsconv_block_stride2_shape():
    block = DSConvBlock(3, 64, stride=2)
    x = torch.randn(2, 3, 64, 64)
    out = block(x)
    assert out.shape == (2, 64, 32, 32), f"Expected (2,64,32,32), got {out.shape}"


def test_dsconv_block_param_count():
    block = DSConvBlock(3, 64, stride=2)
    params = sum(p.numel() for p in block.parameters())
    # depthwise: 3*3*3=27, pointwise: 3*64=192, biases: 3+64=67 → 286
    assert params > 0
    assert params < 1000, f"Unexpectedly large param count: {params}"


def test_analysis_net_shape():
    net = AnalysisNet()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 128, 32, 32), f"Expected (2,128,32,32), got {y.shape}"
    assert not torch.isnan(y).any(), "NaN in AnalysisNet output"


def test_synthesis_net_shape():
    net = SynthesisNet()
    y = torch.randn(2, 128, 32, 32)
    with torch.no_grad():
        x_hat = net(y)
    assert x_hat.shape == (2, 3, 256, 256), f"Expected (2,3,256,256), got {x_hat.shape}"
    assert not torch.isnan(x_hat).any(), "NaN in SynthesisNet output"
    assert x_hat.min() >= 0.0, "SynthesisNet output < 0"
    assert x_hat.max() <= 1.0, "SynthesisNet output > 1"


def test_full_model_forward():
    model = NeuralEncoderModel()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    x_hat = out["x_hat"]
    assert x_hat.shape == (2, 3, 256, 256), f"x_hat shape mismatch: {x_hat.shape}"
    assert not torch.isnan(x_hat).any(), "NaN in model output"
    y_like = out["likelihoods"]["y"]
    z_like = out["likelihoods"]["z"]
    assert (y_like > 0).all(), "y likelihoods not > 0"
    assert (z_like > 0).all(), "z likelihoods not > 0"


def test_strip_encode_shape_matches_full():
    model = NeuralEncoderModel()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        full_out = model(x)
        strips = model.strip_encode(x, strip_height=64, overlap=2)

    assert full_out["x_hat"].shape == (1, 3, 256, 256)
    assert len(strips) > 0
    for s in strips:
        assert "x_hat" in s
        # each strip x_hat has same channels and width as the full output
        assert s["x_hat"].shape[1] == full_out["x_hat"].shape[1]
        assert s["x_hat"].shape[3] == full_out["x_hat"].shape[3]
