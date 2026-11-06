"""Problem interface, hard constraint, residual and one training step."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from pino.evaluation import predict, relative_l2
from pino.numerics.constraints import apply_hard_constraint, dirichlet_1d
from pino.models.factory import build_model
from pino.problems import available_problems, build_problem
from pino.problems.thermal import Heat1D
from pino.trainer import train_step

CFG = OmegaConf.load(Path(__file__).parents[1] / "configs" / "problem" / "heat_1d.yaml")
MODEL_CFG = OmegaConf.create({"n_modes": 4, "hidden_channels": 4, "n_layers": 2, "domain_padding": 0.25})


@pytest.fixture
def problem():
    return build_problem(CFG)


@pytest.fixture
def inputs(problem):
    return problem.sample_inputs(3, torch.Generator().manual_seed(0))


def test_registry_finds_heat_1d(problem):
    assert "heat_1d" in available_problems()
    assert isinstance(problem, Heat1D)
    with pytest.raises(KeyError):
        build_problem(OmegaConf.create({"name": "does_not_exist"}))


def test_shapes(problem, inputs):
    grid = problem.grid((16, 24))
    assert grid.shape == (2, 16, 24)
    assert inputs.shape == (3, CFG.n_series)
    assert problem.to_fno_input(inputs, grid).shape == (3, problem.in_channels, 16, 24)
    u = problem.analytical(inputs, grid)
    assert u.shape == (3, problem.out_channels, 16, 24)
    assert problem.residual(u, inputs, grid).shape == (3, 1, 14, 22)
    g, phi = problem.boundary(inputs, grid)
    assert g.shape == u.shape
    assert (g + phi * u).shape == u.shape


def test_coordinates_reach_the_model_exactly_once(problem, inputs):
    model = build_model(MODEL_CFG, problem, 16)
    assert model.positional_embedding is None
    assert model.lifting.fcs[0].in_channels == problem.in_channels
    grid = problem.grid(16)
    x = problem.to_fno_input(inputs, grid)
    assert torch.equal(x[0, 1], grid[0])
    assert torch.allclose(x[0, 2], grid[1] / CFG.t_end)
    assert model(x).shape == (3, 1, 16, 16)


def test_hard_constraint_is_exact_for_any_network_output(problem, inputs):
    grid = problem.grid(16)
    model = build_model(MODEL_CFG, problem, 16)
    u = predict(model, problem, inputs, grid)
    exact = problem.analytical(inputs, grid)

    assert (u[..., 0, :] == 0).all()
    assert (u[..., -1, :] == 0).all()
    g, _ = problem.boundary(inputs, grid)
    assert torch.equal(u[..., 0], g[..., 0])
    assert torch.allclose(u[..., 1:-1, 0], exact[..., 1:-1, 0], atol=1e-6)
    assert (u[..., 1:-1, 1:] != g[..., 1:-1, 1:]).any(), "the network must act away from the constraint"


def test_fno_input_carries_the_initial_profile(problem, inputs):
    grid = problem.grid(16)
    profile = problem.to_fno_input(inputs, grid)[:, 0]
    initial_state = problem.analytical(inputs, grid)[:, 0, :, 0]
    for time_index in (0, 8, 15):
        assert torch.allclose(profile[..., time_index], initial_state)


def test_fd_residual_of_analytical_solution_is_small_and_converges(problem, inputs):
    previous = None
    for resolution in (32, 64, 128):
        grid = problem.grid(resolution)
        norm = problem.residual(problem.analytical(inputs, grid), inputs, grid).pow(2).mean().sqrt().item()
        if previous is not None:
            assert norm < previous / 2
        previous = norm
    assert previous < 2e-3


@pytest.mark.parametrize("mode", ["physics", "data", "hybrid"])
def test_training_step_runs_on_cpu(problem, inputs, mode):
    torch.manual_seed(0)
    grid = problem.grid(16)
    model = build_model(MODEL_CFG, problem, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = OmegaConf.create({"physics": 1.0, "data": 1.0})
    target = problem.analytical(inputs, grid) if mode != "physics" else None
    before = [p.detach().clone() for p in model.parameters()]

    components = train_step(model, optimizer, problem, inputs, grid, mode, weights, target)

    assert components["total"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_relative_l2_scale():
    target = torch.ones(2, 2, 4, 4)
    prediction = target.clone()
    prediction[:, 0] *= 1.1
    assert torch.allclose(relative_l2(prediction, target), torch.tensor([[0.1, 0.0], [0.1, 0.0]]))


def test_dirichlet_1d_constraint_reproduces_boundary_values_exactly():
    x = torch.linspace(0, 1, 17)
    left = torch.tensor([[1.0, -2.0], [0.5, 0.0]])
    right = torch.tensor([[0.0, 3.0], [-0.25, 7.0]])
    g, phi = dirichlet_1d(x, left, right)
    u = apply_hard_constraint(torch.randn(2, 2, 17), g, phi)
    assert g.shape == (2, 2, 17) and phi.shape == (1, 1, 17)
    assert torch.equal(u[..., 0], left) and torch.equal(u[..., -1], right)
