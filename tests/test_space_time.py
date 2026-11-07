"""Generic 2D+t pieces: n-dimensional diffusion residual, dirichlet_2d_t, resolution keys, 2d_t plot."""

import math

import numpy as np
import pytest
import torch

from pino.data.dataset import OperatorDataset
from pino.evaluation import EvaluationResult
from pino.numerics.constraints import apply_hard_constraint, dirichlet_2d_t
from pino.numerics.derivatives import central_d1, central_d2, interior
from pino.numerics.grids import grid_spacing, resolution_key, resolution_size, uniform_grid
from pino.pdes.diffusion import diffusion_residual
from pino.visualization import plot_resolution, plot_solution


def test_1d_residual_is_bit_identical_to_the_former_formula():
    grid = uniform_grid([(0.0, 1.0), (0.01, 0.05)], (33, 17))
    u = torch.randn(3, 1, 33, 17, generator=torch.Generator().manual_seed(0))
    diffusivity = torch.rand(3, 1, 1, 1, generator=torch.Generator().manual_seed(1))
    dx, dt = grid_spacing(grid, 0), grid_spacing(grid, 1)
    former = interior(central_d1(u, dt, -1), [-2]) - diffusivity * interior(central_d2(u, dx, -2), [-1])
    assert torch.equal(diffusion_residual(u, grid, diffusivity), former)


def _heat_2d(grid, a):
    x, y, t = grid
    return (torch.sin(math.pi * x) * torch.sin(2 * math.pi * y) * torch.exp(-a * 5 * math.pi**2 * t))[None, None]


@pytest.mark.parametrize("a", [1.0, 0.3])
def test_2d_residual_of_an_exact_solution_converges_with_second_order(a):
    norms = []
    for n in (16, 32, 64):
        grid = uniform_grid([(0.0, 1.0), (0.0, 1.0), (0.0, 0.05)], (n, n, n)).double()
        residual = diffusion_residual(_heat_2d(grid, a), grid, a)
        assert residual.shape == (1, 1, n - 2, n - 2, n - 2)
        norms.append(residual.pow(2).mean().sqrt().item())
    rates = [math.log2(norms[i] / norms[i + 1]) for i in range(2)]
    assert all(rate > 1.8 for rate in rates), rates


def test_2d_residual_detects_a_wrong_field():
    grid = uniform_grid([(0.0, 1.0), (0.0, 1.0), (0.0, 0.05)], (32, 32, 32)).double()
    exact, wrong = _heat_2d(grid, 1.0), _heat_2d(grid, 2.0)
    assert diffusion_residual(wrong, grid, 1.0).abs().mean() > 100 * diffusion_residual(exact, grid, 1.0).abs().mean()


def _field(x, y, t):
    """Smooth, non-trivial test function with non-zero values on every edge and corner."""
    # The last term vanishes on every edge but grows in time: g cannot reproduce it inside.
    interior_growth = 50 * x * (1 - x) * y * (1 - y) * t
    return torch.sin(1.3 * x + 2.1 * y) + x * y * t + torch.cos(3.0 * t * (x - y)) + t**2 + interior_growth


def _edges(grid):
    x, y, t = grid[0, :, 0, 0], grid[1, 0, :, 0], grid[2, 0, 0, :]
    X, T = torch.meshgrid(x, t, indexing="ij")
    Y, T2 = torch.meshgrid(y, t, indexing="ij")
    initial = _field(grid[0, ..., 0], grid[1, ..., 0], grid[2, ..., 0])
    bottom, top = _field(X, 0.0 * X, T), _field(X, 0.0 * X + 1.0, T)
    left, right = _field(0.0 * Y, Y, T2), _field(0.0 * Y + 1.0, Y, T2)
    as_bc = lambda v: v[None, None]  # noqa: E731
    return x, y, t, as_bc(initial), *(as_bc(edge) for edge in (bottom, top, left, right))


def test_dirichlet_2d_t_reproduces_initial_field_and_all_edges():
    grid = uniform_grid([(0.0, 1.0), (0.0, 1.0), (0.02, 0.05)], (9, 11, 7)).double()
    x, y, t, initial, bottom, top, left, right = _edges(grid)
    g, phi = dirichlet_2d_t(x, y, t, initial, bottom, top, left, right)
    assert g.shape == (1, 1, 9, 11, 7) and phi.shape == (1, 1, 9, 11, 7)
    exact = _field(*grid)[None, None]
    torch.testing.assert_close(g[..., 0], exact[..., 0], rtol=0, atol=1e-12)  # t = t0
    for index in ((slice(None), slice(None), 0), (slice(None), slice(None), -1)):  # x = 0, 1
        torch.testing.assert_close(g[..., index[2], :, :], exact[..., index[2], :, :], rtol=0, atol=1e-12)
    for end in (0, -1):  # y = 0, 1
        torch.testing.assert_close(g[..., :, end, :], exact[..., :, end, :], rtol=0, atol=1e-12)
    # Corners at every time, e.g. (1, 1, t).
    torch.testing.assert_close(g[..., -1, -1, :], exact[..., -1, -1, :], rtol=0, atol=1e-12)
    # The interior is not prescribed: g differs from the field there.
    assert (g - exact)[..., 1:-1, 1:-1, 1:].abs().max() > 1e-3


def test_dirichlet_2d_t_phi_vanishes_exactly_on_boundary_and_initial_slice():
    grid = uniform_grid([(0.0, 1.0), (0.0, 1.0), (0.02, 0.05)], (8, 10, 6))
    x, y, t, *data = _edges(grid)
    g, phi = dirichlet_2d_t(x, y, t, *data)
    expected = torch.zeros(8, 10, 6, dtype=torch.bool)
    expected[0], expected[-1], expected[:, 0], expected[:, -1], expected[..., 0] = True, True, True, True, True
    assert torch.equal(phi[0, 0] == 0, expected)
    assert torch.all(phi[0, 0][~expected] > 0)
    # With the hard constraint the network has no influence on the prescribed points.
    u = apply_hard_constraint(torch.randn(1, 1, 8, 10, 6), g, phi)
    assert torch.equal(u[..., expected], g[..., expected])


def test_resolution_keys():
    assert resolution_key(64) == 64 and resolution_size(64) == 64
    assert resolution_key([64, 64, 32]) == "64x64x32" and resolution_size("64x64x32") == 64


class _FakeProblem:
    variable_names = {"coordinates": ["x", "y", "t"], "solution": ["T"]}
    units = {"x": "m", "y": "m", "t": "s", "T": "K"}
    main_channel = 0
    plot_type = "2d_t"

    def to_physical(self, variable, values):
        return values

    def label(self, variable):
        return f"{variable} [{self.units[variable]}]"


def test_2d_t_plot_writes_best_median_and_worst(tmp_path):
    grid = uniform_grid([(0.0, 1.0), (0.0, 1.0), (0.0, 0.05)], (8, 8, 6))
    targets = torch.rand(5, 1, 8, 8, 6, generator=torch.Generator().manual_seed(0))
    predictions = targets + 0.01 * torch.arange(5.0)[:, None, None, None, None]
    errors = torch.arange(5.0)[:, None] / 100
    dataset = OperatorDataset(torch.zeros(5, 1), grid, targets)
    result = EvaluationResult(dataset, predictions, errors, 0, {}, "T")
    plot_solution(_FakeProblem(), result, "test", tmp_path / "solution.png")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["solution_best.png", "solution_median.png", "solution_worst.png"]


def test_resolution_plot_accepts_space_time_keys(tmp_path):
    errors = {"32x32x16": {"mean": 0.02}, "64x64x32": {"mean": 0.01}, "128x128x64": {"mean": 0.015}}
    plot_resolution(errors, "64x64x32", "test", tmp_path / "resolution.png")
    assert (tmp_path / "resolution.png").stat().st_size > 0
