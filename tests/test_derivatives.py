"""Finite differences against known functions, including their convergence order."""

import math

import pytest
import torch

from pino.numerics.derivatives import central_d1, central_d2, interior
from pino.numerics.grids import as_resolution, grid_spacing, uniform_grid


def _grid(n: int) -> tuple[torch.Tensor, float]:
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    return x, 1.0 / (n - 1)


def _max_error(n: int, operator, function, exact) -> float:
    x, h = _grid(n)
    return (operator(function(x), h, 0) - exact(x[1:-1])).abs().max().item()


CASES = [
    (central_d1, lambda x: torch.sin(2 * math.pi * x), lambda x: 2 * math.pi * torch.cos(2 * math.pi * x)),
    (central_d2, lambda x: torch.sin(2 * math.pi * x), lambda x: -((2 * math.pi) ** 2) * torch.sin(2 * math.pi * x)),
    (central_d1, lambda x: x**3, lambda x: 3 * x**2),
    (central_d2, lambda x: x**4, lambda x: 12 * x**2),
]


@pytest.mark.parametrize("operator,function,exact", CASES)
def test_convergence_order_is_two(operator, function, exact):
    errors = [_max_error(n, operator, function, exact) for n in (21, 41, 81, 161)]
    orders = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]
    assert all(order == pytest.approx(2.0, abs=0.1) for order in orders)


@pytest.mark.parametrize(
    "operator,function,exact",
    [
        (central_d1, lambda x: 3 * x**2 - x, lambda x: 6 * x - 1),
        (central_d2, lambda x: 3 * x**3 + x**2, lambda x: 18 * x + 2),
    ],
)
def test_exact_for_polynomials_within_stencil_order(operator, function, exact):
    assert _max_error(33, operator, function, exact) < 1e-10


def test_shapes_along_selected_dimension():
    u = torch.zeros(4, 1, 16, 32)
    assert central_d1(u, 0.1, -1).shape == (4, 1, 16, 30)
    assert central_d2(u, 0.1, -2).shape == (4, 1, 14, 32)
    assert interior(u, [-2, -1]).shape == (4, 1, 14, 30)


def test_derivative_acts_along_requested_dimension():
    x = torch.linspace(0, 1, 11, dtype=torch.float64)
    t = torch.linspace(0, 2, 21, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = 3 * xx + 5 * tt
    assert torch.allclose(central_d1(u, 0.1, 0), torch.full((9, 21), 3.0, dtype=torch.float64))
    assert torch.allclose(central_d1(u, 0.1, 1), torch.full((11, 19), 5.0, dtype=torch.float64))


def test_uniform_grid_contains_boundaries_and_spacing():
    grid = uniform_grid([(0.0, 1.0), (0.0, 0.1)], (11, 5))
    assert grid.shape == (2, 11, 5)
    assert grid[0, 0, 0] == 0.0 and grid[0, -1, 0] == 1.0
    assert grid[1, 0, 0] == 0.0 and grid[1, 0, -1] == pytest.approx(0.1)
    assert grid_spacing(grid, 0) == pytest.approx(0.1)
    assert grid_spacing(grid, 1) == pytest.approx(0.025)


def test_as_resolution():
    assert as_resolution(8, 2) == (8, 8)
    assert as_resolution([8, 16], 2) == (8, 16)
    with pytest.raises(ValueError):
        as_resolution([8, 16, 32], 2)
