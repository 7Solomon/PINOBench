"""beam_1d: closed-form solution, boundaries, residual and constraint."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.analytical.beam import simply_supported_beam
from pino.evaluation import predict
from pino.models.factory import build_model
from pino.problems import build_problem

CFG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "problem" / "beam_1d.yaml")
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


def _parameters(batch: int = 6):
    generator = torch.Generator().manual_seed(0)
    q0, q1 = torch.rand(2, batch, generator=generator).double()
    stiffness = 0.5 + 1.5 * torch.rand(batch, generator=generator).double()
    return q0, q1, stiffness


def _derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


@pytest.fixture
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def test_fourth_order_and_mixed_form_via_autograd():
    q0, q1, stiffness = _parameters()
    for b in range(len(q0)):
        x = torch.rand(50, dtype=torch.float64).requires_grad_()
        w, m = simply_supported_beam(q0[b : b + 1], q1[b : b + 1], stiffness[b : b + 1], x)
        w, m = w[0], m[0]
        load = q0[b] + q1[b] * x
        w2 = _derivative(_derivative(w, x), x)
        w4 = _derivative(_derivative(w2, x), x)
        m2 = _derivative(_derivative(m, x), x)
        assert (stiffness[b] * w4 - load).abs().max() < 1e-10
        assert (m2 + load).abs().max() < 1e-10
        assert (w2 + m / stiffness[b]).abs().max() < 1e-10


def test_supports_are_hinged():
    q0, q1, stiffness = _parameters()
    w, m = simply_supported_beam(q0, q1, stiffness, torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert w.abs().max() < 1e-15 and m.abs().max() < 1e-15


def test_textbook_midspan_values():
    one = torch.ones(1, dtype=torch.float64)
    w, m = simply_supported_beam(one, 0 * one, one, torch.tensor([0.5], dtype=torch.float64))
    assert w.item() == pytest.approx(5 / 384) and m.item() == pytest.approx(1 / 8)


def test_fd_residual_of_analytical_solution_is_small_and_converges(float64):
    problem = build_problem(CFG)
    inputs = problem.sample_inputs(8, torch.Generator().manual_seed(1))
    norms = []
    for resolution in (32, 64, 128):
        grid = problem.grid(resolution)
        residual = problem.residual(problem.analytical(inputs, grid), inputs, grid)
        norms.append(residual.pow(2).mean(dim=(0, 2)).sqrt())
    curvature, equilibrium = torch.stack(norms).T
    # M is a cubic, so the central difference of M'' is exact.
    assert equilibrium.max() < 1e-9
    # w is a quintic: second-order convergence of w''.
    assert curvature[-1] < 1e-3
    assert curvature[0] / curvature[1] == pytest.approx(4, rel=0.1)
    assert curvature[1] / curvature[2] == pytest.approx(4, rel=0.1)


def test_shapes_and_hard_constraint():
    problem = build_problem(CFG)
    inputs = problem.sample_inputs(3, torch.Generator().manual_seed(0))
    grid = problem.grid(16)
    assert grid.shape == (1, 16)
    assert problem.to_fno_input(inputs, grid).shape == (3, 3, 16)
    assert problem.analytical(inputs, grid).shape == (3, 2, 16)
    assert problem.residual(problem.analytical(inputs, grid), inputs, grid).shape == (3, 2, 14)
    u = predict(build_model(MODEL_CFG, problem, 16), problem, inputs, grid)
    assert (u[..., 0] == 0).all() and (u[..., -1] == 0).all()
    assert (u[..., 1:-1] != 0).any()


def test_deflection_is_in_millimetre_range():
    problem = build_problem(CFG)
    inputs = problem.sample_inputs(500, torch.Generator().manual_seed(0))
    w = problem.to_physical("w", problem.analytical(inputs, problem.grid(64))[:, 0])
    assert 0.1 < w.abs().amax(dim=1).median() < 50
