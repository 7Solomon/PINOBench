"""Heated block: the analytical solution against sympy, autograd and its boundary conditions."""

import math

import pytest
import sympy as sp
import torch

from pino.analytical.elasticity import thermal_mode_profile, thermoelastic_block

C = torch.tensor([[0.4, -0.2, 0.1, 0.05, -0.08, 0.03]], dtype=torch.float64)
T0 = torch.tensor([0.8], dtype=torch.float64)
NU = torch.tensor([0.25], dtype=torch.float64)


def single_mode(n: int, amplitude: float = 1.0) -> torch.Tensor:
    coefficients = torch.zeros(1, n, dtype=torch.float64)
    coefficients[0, n - 1] = amplitude
    return coefficients


def test_sympy_goodier_potential_and_displacement_formulas():
    """Goodier's potential, and the displacements of the Airy function, for arbitrary constants."""
    x, y, lam, nu, S = sp.symbols("x y lambda nu S", positive=True)
    T = sp.cos(lam * x) * sp.sinh(lam * y) / S
    psi = (1 + nu) * sp.cos(lam * x) * y * sp.cosh(lam * y) / (2 * lam * S)
    assert sp.simplify((sp.diff(psi, x, 2) + sp.diff(psi, y, 2) - (1 + nu) * T).rewrite(sp.exp)) == 0
    # sigma of u_i = d psi / d x_i with the thermal strain: the Airy form of the docstring.
    e_xx, e_yy = sp.diff(psi, x, 2) - T, sp.diff(psi, y, 2) - T
    factor = 1 / (1 - nu**2)
    assert sp.simplify((factor * (e_xx + nu * e_yy) + sp.diff(psi, y, 2) / (1 + nu)).rewrite(sp.exp)) == 0
    assert sp.simplify((factor * (e_yy + nu * e_xx) + sp.diff(psi, x, 2) / (1 + nu)).rewrite(sp.exp)) == 0

    A, B, C_, D = sp.symbols("A B C D")
    F = A * sp.cosh(lam * y) + B * sp.sinh(lam * y) + C_ * y * sp.cosh(lam * y) + D * y * sp.sinh(lam * y)
    F1, F2, F3 = (sp.diff(F, y, k) for k in (1, 2, 3))
    u = sp.sin(lam * x) * (F2 + nu * lam**2 * F + sp.sinh(lam * y) / S) / lam
    v = sp.cos(lam * x) * ((F3 - (2 + nu) * lam**2 * F1) / lam**2 + sp.cosh(lam * y) / (lam * S))
    e_xx, e_yy = sp.diff(u, x) - T, sp.diff(v, y) - T
    sxx, syy = sp.cos(lam * x) * F2, -lam**2 * sp.cos(lam * x) * F
    sxy = lam * sp.sin(lam * x) * F1
    assert sp.simplify((factor * (e_xx + nu * e_yy) - sxx).rewrite(sp.exp)) == 0
    assert sp.simplify((factor * (e_yy + nu * e_xx) - syy).rewrite(sp.exp)) == 0
    assert sp.simplify(((sp.diff(u, y) + sp.diff(v, x)) / (2 * (1 + nu)) - sxy).rewrite(sp.exp)) == 0


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("height", [1.0, 0.5])
def test_closed_form_matches_the_general_4x4_system(n, height):
    """F'(0) = 0, F'''(0) = -lambda / S, F'(H) = 0, F(H) = 0 as a linear system, solved in fp64."""
    lam = n * math.pi
    y = sp.symbols("y", positive=True)
    basis = [sp.cosh(lam * y), sp.sinh(lam * y), y * sp.cosh(lam * y), y * sp.sinh(lam * y)]
    d0, d1, d3 = ([sp.lambdify(y, sp.diff(b, y, k), "math") for b in basis] for k in (0, 1, 3))
    matrix = torch.tensor(
        [[g(1e-300) for g in d1], [g(1e-300) for g in d3], [g(height) for g in d1], [g(height) for g in d0]],
        dtype=torch.float64,
    )
    scale = matrix.abs().amax(dim=1, keepdim=True)
    rhs = torch.tensor([0.0, -lam / math.sinh(lam * height), 0.0, 0.0], dtype=torch.float64)[:, None] / scale
    constants = torch.linalg.solve(matrix / scale, rhs)[:, 0]
    points = torch.linspace(0.0, height, 21, dtype=torch.float64)
    general = sum(c * torch.tensor([g(float(p) if p > 0 else 1e-300) for p in points], dtype=torch.float64) for c, g in zip(constants, d0))
    f = thermal_mode_profile(torch.tensor([lam], dtype=torch.float64), points, height)[0][:, 0]
    torch.testing.assert_close(f, general, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("coefficients", [single_mode(n) for n in range(1, 7)] + [C], ids=[f"mode{n}" for n in range(1, 7)] + ["sum"])
@pytest.mark.parametrize("height", [1.0, 0.5])
def test_conduction_and_navier_lame_by_autograd(coefficients, height):
    x = torch.rand(40, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).requires_grad_()
    y = (height * torch.rand(40, dtype=torch.float64, generator=torch.Generator().manual_seed(1))).requires_grad_()
    fields = thermoelastic_block(T0, coefficients, NU, x, y, height)
    T, u, v = fields["T"][0], fields["u"][0], fields["v"][0]
    nu = NU.item()

    def d(field, var):
        return torch.autograd.grad(field.sum(), var, create_graph=True)[0]

    T_x, T_y, u_x, u_y, v_x, v_y = d(T, x), d(T, y), d(u, x), d(u, y), d(v, x), d(v, y)
    assert (d(T_x, x) + d(T_y, y)).abs().max() < 1e-8
    factor = 1.0 / (1 - nu**2)
    thermal = 1.0 / (1 - nu)
    r1 = factor * (d(u_x, x) + (1 - nu) / 2 * d(u_y, y) + (1 + nu) / 2 * d(v_y, x)) - thermal * T_x
    r2 = factor * (d(v_y, y) + (1 - nu) / 2 * d(v_x, x) + (1 + nu) / 2 * d(u_x, y)) - thermal * T_y
    assert r1.abs().max() < 1e-8 and r2.abs().max() < 1e-8
    # The returned stresses are those of the displacements, with the thermal strain.
    torch.testing.assert_close(fields["sigma_xx"][0], factor * ((u_x - T) + nu * (v_y - T)), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(fields["sigma_yy"][0], factor * ((v_y - T) + nu * (u_x - T)), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(fields["sigma_xy"][0], (u_y + v_x) / (2 * (1 + nu)), rtol=1e-9, atol=1e-9)


def test_all_boundary_conditions():
    s = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    zero, one = torch.zeros_like(s), torch.ones_like(s)
    tolerance = 1e-12
    for side in (zero, one):  # left and right: u = 0, sigma_xy = 0, insulated
        x = side.clone().requires_grad_()
        fields = thermoelastic_block(T0, C, NU, x, s)
        assert fields["u"].abs().max() < tolerance and fields["sigma_xy"].abs().max() < tolerance
        assert torch.autograd.grad(fields["T"].sum(), x)[0].abs().max() < tolerance
    bottom = thermoelastic_block(T0, C, NU, s, zero)
    assert bottom["v"].abs().max() < tolerance and bottom["sigma_xy"].abs().max() < tolerance
    assert bottom["T"].abs().max() < tolerance
    top = thermoelastic_block(T0, C, NU, s, one)
    assert top["sigma_xy"].abs().max() < tolerance and top["sigma_yy"].abs().max() < tolerance
    k = torch.arange(1, 7, dtype=torch.float64) * math.pi
    torch.testing.assert_close(top["T"][0], T0 + (C * torch.cos(s[:, None] * k)).sum(-1), rtol=0, atol=tolerance)


def test_mode_zero_is_the_closed_formula():
    x, y = torch.meshgrid(torch.linspace(0, 1, 9, dtype=torch.float64), torch.linspace(0, 1, 9, dtype=torch.float64), indexing="ij")
    fields = thermoelastic_block(T0, torch.zeros(1, 3, dtype=torch.float64), NU, x, y)
    torch.testing.assert_close(fields["T"][0], T0 * y, rtol=0, atol=1e-14)
    assert fields["u"].abs().max() == 0 and fields["sigma_yy"].abs().max() == 0 and fields["sigma_xy"].abs().max() == 0
    torch.testing.assert_close(fields["v"][0], (1 + NU) * T0 * y**2 / 2, rtol=0, atol=1e-14)
    # Fully restrained in x, as the test of the restrained thermal strain: sigma_xx = -E alpha T.
    torch.testing.assert_close(fields["sigma_xx"][0], -T0 * y, rtol=0, atol=1e-14)


def test_stresses_do_not_depend_on_poisson():
    s = torch.linspace(0.05, 0.95, 11, dtype=torch.float64)
    first = thermoelastic_block(T0, C, torch.tensor([0.15], dtype=torch.float64), s, s)
    second = thermoelastic_block(T0, C, torch.tensor([0.35], dtype=torch.float64), s, s)
    for name in ("T", "sigma_xx", "sigma_yy", "sigma_xy"):
        torch.testing.assert_close(first[name], second[name], rtol=1e-12, atol=1e-12)
    assert (first["v"] - second["v"]).abs().max() > 1e-3


def test_heating_expands_the_block_upwards():
    """Uniform positive temperature on top: the top surface rises."""
    x = torch.linspace(0, 1, 21, dtype=torch.float64)
    top = thermoelastic_block(T0, 0 * C, NU, x, torch.ones_like(x))
    assert top["v"].min() > 0


def test_high_modes_decay_faster_with_depth():
    ratio = {}
    for n in (1, 6):
        points = torch.tensor([1.0, 0.5], dtype=torch.float64)
        field = thermoelastic_block(0 * T0, single_mode(n), NU, torch.zeros(2, dtype=torch.float64), points)["T"][0]
        ratio[n] = abs(field[1] / field[0]).item()
    assert ratio[6] < 0.01 * ratio[1], ratio


def test_large_modes_are_numerically_stable():
    point = torch.tensor([0.123], dtype=torch.float64), torch.tensor([0.99], dtype=torch.float64)
    fields = thermoelastic_block(0 * T0, single_mode(60), NU, *point)
    assert all(torch.isfinite(field).all() for field in fields.values())
    top = thermoelastic_block(0 * T0, single_mode(60), NU, torch.tensor([0.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))
    assert top["T"].item() == pytest.approx(1.0, abs=1e-10)
    assert top["sigma_yy"].abs().item() < 1e-10
