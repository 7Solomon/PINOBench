"""Elastic block: the analytical solution against sympy, autograd and its boundary conditions."""

import math

import pytest
import sympy as sp
import torch

from pino.analytical.elasticity import block_mode_profile, elastic_block

P = torch.tensor([[0.4, -0.2, 0.1, 0.05, -0.08, 0.03]], dtype=torch.float64)
P0 = torch.tensor([0.8], dtype=torch.float64)
NU = torch.tensor([0.25], dtype=torch.float64)


def single_mode(n: int, amplitude: float = 1.0) -> torch.Tensor:
    coefficients = torch.zeros(1, n, dtype=torch.float64)
    coefficients[0, n - 1] = amplitude
    return coefficients


def test_sympy_derivation_of_one_mode():
    """Symbolically: the displacement formulas give the stresses and solve Navier-Lame."""
    x, y, lam, nu, E = sp.symbols("x y lambda nu E", positive=True)
    A, B, C, D = sp.symbols("A B C D")
    f = A * sp.cosh(lam * y) + B * sp.sinh(lam * y) + C * y * sp.cosh(lam * y) + D * y * sp.sinh(lam * y)
    f1, f2, f3 = (sp.diff(f, y, k) for k in (1, 2, 3))
    big_f = (2 * lam**2 * f1 - f3) / lam**4
    # F is an antiderivative of f (up to a constant): the compatibility condition of the shear strain.
    assert sp.simplify(sp.diff(big_f, y) - f) == 0
    u = sp.sin(lam * x) * (f2 + nu * lam**2 * f) / (lam * E)
    v = -sp.cos(lam * x) * (lam**2 * big_f + nu * f1) / E
    sxx_airy, syy_airy, sxy_airy = sp.cos(lam * x) * f2, -lam**2 * sp.cos(lam * x) * f, lam * sp.sin(lam * x) * f1
    factor = E / (1 - nu**2)
    sxx = factor * (sp.diff(u, x) + nu * sp.diff(v, y))
    syy = factor * (sp.diff(v, y) + nu * sp.diff(u, x))
    sxy = E / (2 * (1 + nu)) * (sp.diff(u, y) + sp.diff(v, x))
    for computed, airy in ((sxx, sxx_airy), (syy, syy_airy), (sxy, sxy_airy)):
        assert sp.simplify((computed - airy).rewrite(sp.exp)) == 0
    # Equilibrium of the Airy stresses.
    assert sp.simplify(sp.diff(sxx_airy, x) + sp.diff(sxy_airy, y)) == 0
    assert sp.simplify(sp.diff(sxy_airy, x) + sp.diff(syy_airy, y)) == 0


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("height", [1.0, 0.5])
def test_closed_form_matches_the_general_4x4_system(n, height):
    """The four boundary conditions as a linear system for A, B, C, D, solved numerically (fp64)."""
    lam = n * math.pi
    y = sp.symbols("y", positive=True)
    basis = [sp.cosh(lam * y), sp.sinh(lam * y), y * sp.cosh(lam * y), y * sp.sinh(lam * y)]
    d1 = [sp.lambdify(y, sp.diff(b, y, 1), "math") for b in basis]
    d3 = [sp.lambdify(y, sp.diff(b, y, 3), "math") for b in basis]
    d0 = [sp.lambdify(y, b, "math") for b in basis]
    matrix = torch.tensor(
        [[g(1e-300) for g in d1], [g(1e-300) for g in d3], [g(height) for g in d1], [g(height) for g in d0]],
        dtype=torch.float64,
    )
    # f'(0) = 0, f'''(0) = 0 (sigma_xy and v at the bottom), f'(H) = 0, f(H) = 1 / lambda^2.
    scale = matrix.abs().amax(dim=1, keepdim=True)
    rhs = torch.tensor([0.0, 0.0, 0.0, 1.0 / lam**2], dtype=torch.float64)[:, None] / scale
    constants = torch.linalg.solve(matrix / scale, rhs)[:, 0]
    points = torch.linspace(0.0, height, 21, dtype=torch.float64)
    general = sum(c * torch.tensor([g(float(p) if p > 0 else 1e-300) for p in points], dtype=torch.float64) for c, g in zip(constants, d0))
    f = block_mode_profile(torch.tensor([lam], dtype=torch.float64), points, height)[0][:, 0]
    torch.testing.assert_close(f, general, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("coefficients", [single_mode(n) for n in range(1, 7)] + [P], ids=[f"mode{n}" for n in range(1, 7)] + ["sum"])
def test_navier_lame_by_autograd(coefficients):
    x = torch.rand(40, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).requires_grad_()
    y = torch.rand(40, dtype=torch.float64, generator=torch.Generator().manual_seed(1)).requires_grad_()
    fields = elastic_block(P0, coefficients, NU, x, y)
    u, v = fields["u"][0], fields["v"][0]
    nu = NU.item()

    def d(field, var):
        return torch.autograd.grad(field.sum(), var, create_graph=True)[0]

    u_x, u_y, v_x, v_y = d(u, x), d(u, y), d(v, x), d(v, y)
    factor = 1.0 / (1 - nu**2)
    r1 = factor * (d(u_x, x) + (1 - nu) / 2 * d(u_y, y) + (1 + nu) / 2 * d(v_y, x))
    r2 = factor * (d(v_y, y) + (1 - nu) / 2 * d(v_x, x) + (1 + nu) / 2 * d(u_x, y))
    assert r1.abs().max() < 1e-8 and r2.abs().max() < 1e-8
    # The returned stresses are those of the displacements.
    torch.testing.assert_close(fields["sigma_xx"][0], factor * (u_x + nu * v_y), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(fields["sigma_yy"][0], factor * (v_y + nu * u_x), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(fields["sigma_xy"][0], (u_y + v_x) / (2 * (1 + nu)), rtol=1e-10, atol=1e-10)


def test_all_six_boundary_conditions():
    s = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    zero, one = torch.zeros_like(s), torch.ones_like(s)
    tolerance = 1e-12
    for side in (zero, one):  # left and right: u = 0, sigma_xy = 0
        fields = elastic_block(P0, P, NU, side, s)
        assert fields["u"].abs().max() < tolerance and fields["sigma_xy"].abs().max() < tolerance
    bottom = elastic_block(P0, P, NU, s, zero)
    assert bottom["v"].abs().max() < tolerance and bottom["sigma_xy"].abs().max() < tolerance
    top = elastic_block(P0, P, NU, s, one)
    assert top["sigma_xy"].abs().max() < tolerance
    k = torch.arange(1, 7, dtype=torch.float64) * math.pi
    load = P0 + (P * torch.cos(s[:, None] * k)).sum(-1)
    torch.testing.assert_close(top["sigma_yy"][0], -load, rtol=0, atol=tolerance)


def test_mode_zero_is_the_closed_formula():
    x, y = torch.meshgrid(torch.linspace(0, 1, 9, dtype=torch.float64), torch.linspace(0, 1, 9, dtype=torch.float64), indexing="ij")
    fields = elastic_block(P0, torch.zeros(1, 3, dtype=torch.float64), NU, x, y)
    assert fields["u"].abs().max() == 0 and fields["sigma_xy"].abs().max() == 0
    torch.testing.assert_close(fields["v"][0], -P0 * (1 - NU**2) * y, rtol=0, atol=1e-14)
    assert torch.all(fields["sigma_xx"] == -NU * P0) and torch.all(fields["sigma_yy"] == -P0)


def test_purely_positive_load_compresses_everywhere():
    """p > 0 everywhere (0.8 - sum |p_n| > 0 below): v <= 0 in the whole block."""
    coefficients = torch.tensor([[0.3, -0.1, 0.1, 0.05, -0.05, 0.05]], dtype=torch.float64)
    assert (P0 - coefficients.abs().sum()).item() > 0
    x, y = torch.meshgrid(torch.linspace(0, 1, 65, dtype=torch.float64), torch.linspace(0, 1, 65, dtype=torch.float64), indexing="ij")
    assert elastic_block(P0, coefficients, NU, x, y)["v"].max() <= 1e-14


def test_high_modes_decay_faster_with_depth():
    depth = 0.5
    ratio = {}
    for n in (1, 6):
        points = torch.tensor([1.0, 1.0 - depth], dtype=torch.float64)
        fields = elastic_block(0 * P0, single_mode(n), NU, torch.zeros(2, dtype=torch.float64), points)["sigma_yy"][0]
        ratio[n] = abs(fields[1] / fields[0]).item()
    assert ratio[6] < 0.05 * ratio[1], ratio
    # Rough rate of Saint-Venant decay for n = 6: exp(-6 pi depth) times the polynomial factor.
    assert ratio[6] < 20 * math.exp(-6 * math.pi * depth)


def test_large_modes_are_numerically_stable():
    fields = elastic_block(0 * P0, single_mode(60), NU, torch.tensor([0.123], dtype=torch.float64), torch.tensor([0.99], dtype=torch.float64))
    assert all(torch.isfinite(field).all() for field in fields.values())
    top = elastic_block(0 * P0, single_mode(60), NU, torch.tensor([0.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))
    assert top["sigma_yy"].item() == pytest.approx(-1.0, abs=1e-10)
