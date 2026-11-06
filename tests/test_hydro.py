"""darcy_layered_1d and richards_steady_1d: closed-form solutions, residuals, constraints."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.analytical.porous_media import gardner_steady_head, layered_darcy_flux, layered_darcy_head
from pino.evaluation import predict
from pino.models.factory import build_model
from pino.problems import build_problem

CONFIGS = Path(__file__).parents[1] / "configs" / "problem"
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


@pytest.fixture
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _problem(name):
    return build_problem(OmegaConf.load(CONFIGS / f"{name}.yaml"))


# --- darcy_layered_1d ---------------------------------------------------------------------


def _layers(batch=8):
    generator = torch.Generator().manual_seed(0)
    k1, k2 = 10.0 ** (2 * torch.rand(2, batch, generator=generator).double() - 1)
    interface = 0.3 + 0.4 * torch.rand(batch, generator=generator).double()
    return k1, k2, interface


def test_darcy_boundaries_and_continuity():
    k1, k2, interface = _layers()
    ends = layered_darcy_head(k1, k2, interface, torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(ends[:, 0], torch.ones_like(k1)) and ends[:, 1].abs().max() < 1e-12

    eps = 1e-9
    for b in range(len(k1)):
        around = torch.tensor([interface[b] - eps, interface[b], interface[b] + eps], dtype=torch.float64)
        head = layered_darcy_head(k1[b : b + 1], k2[b : b + 1], interface[b : b + 1], around)[0]
        assert (head[0] - head[2]).abs() < 1e-7, "head jumps at the interface"
        # Flux continuity k1 h'(x_s-) = k2 h'(x_s+), both equal to -q.
        q = layered_darcy_flux(k1[b], k2[b], interface[b])
        left = torch.tensor([interface[b] - 0.2, interface[b] - 0.1], dtype=torch.float64)
        right = torch.tensor([interface[b] + 0.1, interface[b] + 0.2], dtype=torch.float64)
        slope_left = torch.diff(layered_darcy_head(k1[b : b + 1], k2[b : b + 1], interface[b : b + 1], left)[0]) / 0.1
        slope_right = torch.diff(layered_darcy_head(k1[b : b + 1], k2[b : b + 1], interface[b : b + 1], right)[0]) / 0.1
        assert k1[b] * slope_left == pytest.approx(-q.item()) and k2[b] * slope_right == pytest.approx(-q.item())


def test_darcy_pde_holds_via_autograd():
    """-(k h')' = 0: the flux -k h' equals q at every point of both layers."""
    k1, k2, interface = _layers()
    for b in range(len(k1)):
        x = torch.rand(40, dtype=torch.float64, requires_grad=True)
        head = layered_darcy_head(k1[b : b + 1], k2[b : b + 1], interface[b : b + 1], x)[0]
        slope = torch.autograd.grad(head.sum(), x)[0]
        k = torch.where(x < interface[b], k1[b], k2[b])
        q = layered_darcy_flux(k1[b], k2[b], interface[b])
        assert torch.allclose(-k * slope, q.expand_as(slope))


def test_darcy_discrete_residual_of_analytical_solution_is_zero(float64):
    problem = _problem("darcy_layered_1d")
    inputs = problem.sample_inputs(32, torch.Generator().manual_seed(0))
    for resolution in (32, 64, 256):
        grid = problem.grid(resolution)
        residual = problem.residual(problem.analytical(inputs, grid), inputs, grid)
        assert residual.shape == (32, 1, resolution - 1)
        assert residual.abs().max() < 1e-9


def test_darcy_residual_is_relative_flux_error(float64):
    """Scaling the head drop by 1.1 changes every flux by 10 %, but the mean is subtracted:
    a uniform error is invisible, a flux mismatch between the layers is not."""
    problem = _problem("darcy_layered_1d")
    inputs = torch.tensor([[0.5, 5.0, 0.5]])
    grid = problem.grid(64)
    exact = problem.analytical(inputs, grid)
    linear = 1.0 - grid[0].expand_as(exact)  # right boundary values, wrong flux balance
    assert problem.residual(1.1 * exact, inputs, grid).abs().max() < 1e-9
    assert problem.residual(linear, inputs, grid).abs().max() > 1.0


def test_darcy_face_conductivity_is_harmonic_mean_on_the_interface_cell():
    problem = _problem("darcy_layered_1d")
    grid = problem.grid(11)  # dx = 0.1
    inputs = torch.tensor([[1.0, 4.0, 0.35]])
    faces = problem.face_conductivity(inputs, grid)[0, 0]
    assert faces[2] == pytest.approx(1.0) and faces[4] == pytest.approx(4.0)
    assert faces[3] == pytest.approx(0.1 / (0.05 / 1.0 + 0.05 / 4.0))


def test_darcy_shapes_and_hard_constraint():
    problem = _problem("darcy_layered_1d")
    inputs = problem.sample_inputs(3, torch.Generator().manual_seed(0))
    grid = problem.grid(16)
    x = problem.to_fno_input(inputs, grid)
    assert x.shape == (3, 2, 16) and torch.equal(x[0, 1], grid[0])
    assert x[:, 0].min() >= -1 and x[:, 0].max() <= 1
    u = predict(build_model(MODEL_CFG, problem, 16), problem, inputs, grid)
    assert u.shape == (3, 1, 16)
    assert (u[..., 0] == 1).all() and (u[..., -1] == 0).all()


# --- richards_steady_1d -------------------------------------------------------------------


def _soils(batch=8):
    generator = torch.Generator().manual_seed(0)
    alpha = 1 + 4 * torch.rand(batch, generator=generator).double()
    flux = -0.9 + 0.8 * torch.rand(batch, generator=generator).double()
    return alpha, flux


def test_richards_head_is_non_positive_and_zero_at_water_table():
    alpha, flux = _soils()
    head = gardner_steady_head(alpha, flux, torch.linspace(0, 1, 101, dtype=torch.float64))
    assert (head <= 0).all()
    assert head[:, 0].abs().max() < 1e-15


def test_richards_flux_is_constant_and_pde_holds_via_autograd():
    alpha, flux = _soils()
    for b in range(len(alpha)):
        z = torch.rand(50, dtype=torch.float64, requires_grad=True)
        head = gardner_steady_head(alpha[b : b + 1], flux[b : b + 1], z)[0]
        slope = torch.autograd.grad(head.sum(), z, create_graph=True)[0]
        darcy_flux = -torch.exp(alpha[b] * head) * (slope + 1)
        assert torch.allclose(darcy_flux, flux[b].expand_as(darcy_flux), atol=1e-12)
        divergence = torch.autograd.grad(darcy_flux.sum(), z)[0]
        assert divergence.abs().max() < 1e-10


def test_richards_fd_residual_is_small_and_converges(float64):
    problem = _problem("richards_steady_1d")
    inputs = problem.sample_inputs(16, torch.Generator().manual_seed(0))
    norms = []
    for resolution in (32, 64, 128, 256):
        grid = problem.grid(resolution)
        norms.append(problem.residual(problem.analytical(inputs, grid), inputs, grid).pow(2).mean().sqrt().item())
    assert norms[1] < 1e-2
    for coarse, fine in zip(norms, norms[1:]):
        assert coarse / fine == pytest.approx(4, rel=0.15)


def test_richards_shapes_and_hard_constraint():
    problem = _problem("richards_steady_1d")
    inputs = problem.sample_inputs(3, torch.Generator().manual_seed(0))
    grid = problem.grid(16)
    assert problem.to_fno_input(inputs, grid).shape == (3, 3, 16)
    u = predict(build_model(MODEL_CFG, problem, 16), problem, inputs, grid)
    exact = problem.analytical(inputs, grid)
    assert (u[..., 0] == 0).all()
    assert torch.allclose(u[..., -1], exact[..., -1])
