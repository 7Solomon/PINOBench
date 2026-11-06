"""Model factory: guard against neuraloperator's odd-grid bug and the n_modes check."""

import math

import pytest
import torch
from neuralop.layers.spectral_convolution import SpectralConv
from omegaconf import OmegaConf

from pino.models.factory import build_model


class _Problem:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.in_channels = 2
        self.out_channels = 1


MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 1, "domain_padding": 0.25})


def _identity_spectral_conv(n_dims: int) -> SpectralConv:
    conv = SpectralConv(1, 1, n_modes=(8,) * n_dims)
    conv.weight.tensor.data[...] = 1.0 + 0.0j
    conv.bias.data.zero_()
    return conv


@pytest.mark.parametrize("first,last,corrupt", [(64, 64, False), (65, 64, True), (64, 65, False)])
def test_neuraloperator_bug_affects_only_odd_non_last_dimensions(first, last, corrupt):
    """Documents the upstream bug. If this fails after an upgrade, revisit the guard."""
    x = torch.linspace(0, 1, first + 1)[:-1]
    y = torch.linspace(0, 1, last + 1)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    signal = (torch.sin(2 * math.pi * xx) * torch.cos(2 * math.pi * yy))[None, None]
    with torch.no_grad():
        error = ((_identity_spectral_conv(2)(signal) - signal).norm() / signal.norm()).item()
    assert (error > 0.5) if corrupt else (error < 1e-5)


def test_factory_rejects_odd_grid_in_non_last_dimension():
    model = build_model(MODEL_CFG, _Problem(2), 16)
    model(torch.zeros(1, 2, 16, 17))  # odd last dimension is fine
    with pytest.raises(ValueError, match="odd number of points"):
        model(torch.zeros(1, 2, 17, 16))


def test_factory_accepts_odd_grid_in_1d():
    model = build_model(MODEL_CFG, _Problem(1), 16)
    assert model(torch.zeros(1, 2, 33)).shape == (1, 1, 33)


def test_factory_expands_scalar_modes_to_problem_dimension():
    assert tuple(build_model(MODEL_CFG, _Problem(1), 16).n_modes) == (4,)
    assert tuple(build_model(MODEL_CFG, _Problem(2), 16).n_modes) == (4, 4)


def test_factory_rejects_too_many_modes():
    with pytest.raises(ValueError, match="exceeds half"):
        build_model(MODEL_CFG, _Problem(1), 7)
