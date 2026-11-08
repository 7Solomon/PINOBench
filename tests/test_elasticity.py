"""Generic mechanics pieces: mixed and one-sided derivatives, plane stress with thermal strain."""

import math

import pytest
import torch

from pino.numerics.derivatives import central_mixed, gradient, one_sided_d1
from pino.numerics.grids import grid_spacing, uniform_grid
from pino.pdes.elasticity import navier_lame_residual, plane_stress

E, NU = 2.5, 0.3
ALPHA_T0 = 0.01


def grid_of(n: int) -> torch.Tensor:
    return uniform_grid([(0.0, 1.0), (0.0, 1.0)], (n, n)).double()


def rate(errors: list[float]) -> list[float]:
    return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


def test_mixed_derivative_converges_with_second_order():
    errors = []
    for n in (17, 33, 65):
        grid = grid_of(n)
        x, y = grid
        u = torch.sin(2.0 * x) * torch.cos(3.0 * y)
        exact = (-6.0 * torch.cos(2.0 * x) * torch.sin(3.0 * y))[1:-1, 1:-1]
        result = central_mixed(u, grid_spacing(grid, 0), grid_spacing(grid, 1), -2, -1)
        assert result.shape == exact.shape
        errors.append((result - exact).abs().max().item())
    assert all(r == pytest.approx(2.0, abs=0.2) for r in rate(errors)), errors


@pytest.mark.parametrize("side,index", [("start", 0), ("end", -1)])
def test_one_sided_derivative_converges_with_second_order(side, index):
    errors = []
    for n in (17, 33, 65):
        x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        result = one_sided_d1(torch.sin(2.0 * x), 1.0 / (n - 1), -1, side)
        assert result.shape == (1,)
        errors.append(abs(result.item() - 2.0 * math.cos(2.0 * x[index].item())))
    assert all(r == pytest.approx(2.0, abs=0.2) for r in rate(errors)), errors


def test_one_sided_derivative_is_exact_for_a_parabola_and_rejects_unknown_sides():
    x = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    u = 3.0 * x**2 - x
    torch.testing.assert_close(one_sided_d1(u, 1 / 8, -1, "start")[0], torch.tensor(-1.0, dtype=torch.float64))
    torch.testing.assert_close(one_sided_d1(u, 1 / 8, -1, "end")[0], torch.tensor(5.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="Unknown side"):
        one_sided_d1(u, 1 / 8, -1, "middle")


def test_gradient_on_all_points():
    errors = []
    for n in (17, 33, 65):
        x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        result = gradient(torch.exp(x), 1.0 / (n - 1), -1)
        assert result.shape == x.shape
        errors.append((result - torch.exp(x)).abs().max().item())
    assert all(r == pytest.approx(2.0, abs=0.2) for r in rate(errors)), errors


def test_free_thermal_expansion_is_stress_free():
    grid = grid_of(9)
    x, y = grid
    dx, dy = grid_spacing(grid, 0), grid_spacing(grid, 1)
    u, v = ALPHA_T0 * x, ALPHA_T0 * y
    strain = torch.full_like(x, ALPHA_T0)
    stresses = plane_stress(u, v, dx, dy, E, NU, strain)
    assert all(field.abs().max() < 1e-12 for field in stresses.values())
    first, second = navier_lame_residual(u, v, dx, dy, E, NU, strain)
    assert first.abs().max() < 1e-10 and second.abs().max() < 1e-10


def test_constrained_thermal_expansion():
    grid = grid_of(9)
    x, y = grid
    dx, dy = grid_spacing(grid, 0), grid_spacing(grid, 1)
    u, v = torch.zeros_like(x), (1.0 + NU) * ALPHA_T0 * y
    stresses = plane_stress(u, v, dx, dy, E, NU, torch.full_like(x, ALPHA_T0))
    torch.testing.assert_close(stresses["sigma_xx"], torch.full_like(x, -E * ALPHA_T0), rtol=0, atol=1e-12)
    assert stresses["sigma_yy"].abs().max() < 1e-12 and stresses["sigma_xy"].abs().max() < 1e-12


def test_thermal_gradient_enters_the_residual():
    """A linear thermal strain in y and no displacement: only the second equation sees it."""
    grid = grid_of(33)
    x, y = grid
    dx, dy = grid_spacing(grid, 0), grid_spacing(grid, 1)
    strain = 0.1 * y
    first, second = navier_lame_residual(torch.zeros_like(x), torch.zeros_like(x), dx, dy, E, NU, strain)
    assert first.abs().max() < 1e-12
    torch.testing.assert_close(second, torch.full_like(second, -E / (1 - NU) * 0.1), rtol=1e-10, atol=0)


def test_without_thermal_strain_the_residual_is_that_of_the_mechanical_part():
    grid = grid_of(33)
    x, y = grid
    dx, dy = grid_spacing(grid, 0), grid_spacing(grid, 1)
    u, v = x**2, y**2  # u_xx = 2, v_yy = 2, no mixed terms
    first, second = navier_lame_residual(u, v, dx, dy, E, NU)
    torch.testing.assert_close(first, torch.full_like(first, E / (1 - NU**2) * 2.0), rtol=1e-9, atol=0)
    torch.testing.assert_close(second, torch.full_like(second, E / (1 - NU**2) * 2.0), rtol=1e-9, atol=0)
