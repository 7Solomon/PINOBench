# PINO-Benchmarks

This Project trains Fourier Neural Operators (FNOs) on PDE benchmarks with analytical solutions

## Installation

Requires Python ≥ 3.10.

```bash
conda create -n pinobench python=3.11 -y
conda activate pinobench
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
pytest
```

The device is chosen automatically (`device=auto`). Without a GPU everything runs on the CPU.

## Usage

```bash
# one training run: choose problem, model size and mode
python scripts/train.py problem=heat_1d model=fno_small training=physics
python scripts/train.py problem=beam_1d training=data
python scripts/train.py problem=darcy_layered_1d training=hybrid   # uses loss_weights from its YAML

# override single values
python scripts/train.py problem=richards_steady_1d training=physics training.epochs=1000 seed=1

# variants (defaults: the benchmark's original input family, output_param constrained)
python scripts/train.py problem=beam_1d_sine training=physics problem.output_param=unscaled
python scripts/train.py problem=beam_1d training=data problem.output_param=direct seed=1
python scripts/train.py problem=heat_1d training=physics problem.output_param=soft training.weights.bc=100

# re-evaluate and re-plot a finished run (refuses runs with a different wrapper)
python scripts/evaluate.py outputs/heat_1d/sine-constrained/physics/<timestamp>
python scripts/evaluate.py <run_dir> --allow-legacy  # runs without wrapper_version

# collect all runs under outputs/
python scripts/summarize.py
```

Each run writes to `outputs/<benchmark>/<inputs>-<output_param>/<mode>/<timestamp>/`:

| File | Content |
|---|---|
| `best_model.pt` | weights with the best validation score, rewritten on every improvement |
| `history.json` | loss components per epoch (`physics`, `data`, `bc`, `total`), validation physics loss (and `val_bc` for `soft`) and validation rel. L2 |
| `metrics.json` | rel. L2 per variable (mean, median, max), boundary error `bc_error_mean`/`bc_error_max` per variable, resolution study, time, parameters |
| `training.png` | loss components and validation error with a 5 % line and the selected epoch |
| `solution.png` | best, median and worst test example |
| `error_hist.png` | distribution of the rel. L2 over the test set |
| `resolution.png` | rel. L2 against the test resolution |
| `.hydra/config.yaml` | full configuration of the run |

`summarize.py` writes `outputs/summary.csv`, `outputs/summary.md` and `outputs/summary.png`.
It groups by benchmark × `inputs` × `output_param` × mode, takes the most recent run per seed
and reports the mean, minimum and maximum over the seeds. Runs under `outputs/_superseded/`
are ignored.

### Training modes

| Mode | Loss | Analytical solution |
|---|---|---|
| `physics` | mean square of the normalized PDE residual | evaluation only |
| `data` | MSE against the analytical solution, normalized per variable | training target |
| `hybrid` | `weights.physics · physics + weights.data · data` | training target |

With `output_param: soft` every mode adds `weights.bc · L_bc` (see Variants).

Defaults: Adam with lr = 1e-3 and a cosine schedule, 500 epochs, batch size 32, 1000 training,
200 validation and 200 test examples, each drawn with its own seed.

### Validation, checkpoint and test set

| Mode | Checkpoint selected by (validation set) |
|---|---|
| `physics` | physics loss, without the analytical solution; with `soft` the objective physics + `weights.bc` · L_bc |
| `data`, `hybrid` | mean rel. L2 over **all** output variables |

The validation rel. L2 is logged in every mode, in `physics` mode for observation only. The
test set is evaluated once, after training, with the selected checkpoint. `metrics.json` and
the summary report the main variable of each benchmark.

### Loss weights in `hybrid` mode

With the default weights 1:1, `hybrid` is dominated by the physics term in every benchmark,
because the data loss is orders of magnitude smaller (for example 1e-7 against 1e-4). This is
deliberately not calibrated for every problem. A problem YAML may set

```yaml
loss_weights: {physics: 1.0, data: 300.0}
```

which replaces `training.weights` in `hybrid` mode; `physics` and `data` keep their single
fixed term. `bc` (the weight of `L_bc`, default 10 in the training YAMLs) may also be set there and
acts in every mode. Only `darcy_layered_1d` sets it, because its hybrid run did not reach the data
mode's accuracy at 1:1. A command-line override such as `training.weights.data=50` still wins.

### Variants

Two switches in every problem YAML select how hard the learning task is. The defaults are the
original benchmarks.

| Switch | Values | Default |
|---|---|---|
| `inputs` | input families of the benchmark, see the table below | the first one, the original family |
| `output_param` | `constrained`, `unscaled`, `direct`, `soft` | `constrained` |

| `output_param` | Prediction | Modes |
|---|---|---|
| `constrained` | u = g + φ·s·NN, s = `output_scale(inputs)` per sample or 1 | all |
| `unscaled` | u = g + φ·c·NN, c = `output_scale_global` per channel from the YAML (default 1) | all |
| `direct` | u = c·NN, no boundary values | `data` only |
| `soft` | u = c·NN; boundary and initial values through the penalty `L_bc` | all |

g and φ come from `boundary()`, where φ is only the fixed shape part such as x(1−x). Only
`beam_1d` has a per-sample scale s = (w_c, M_c). `output_scale_global` is the RMS of each
output over a large sample of the input family, rounded. Loss normalizations (`data_scale()`, residual
scaling) may still depend on the inputs per sample; they act on the loss, not on the output.
`direct` is refused in `physics` and `hybrid`: without the hard constraint and without a
boundary loss the PDE does not determine the solution.

**Soft constraints.** `soft` predicts like `direct` but adds, in every mode,

    L_bc = mean over the boundary points of ((u − g) / s)²,   weight `weights.bc` (default 10)

The boundary points are exactly those where the hard constraint acts, φ = 0 in `boundary()`:
the Dirichlet edges and, for problems in time, the initial time slice (`Problem.boundary_values`;
a test checks the mask against the grid coordinates for every problem). g are the values of
`boundary()` there. s is the data-loss normalization `data_scale()`, otherwise the global
constant c (1 if unset). In `physics` mode the checkpoint is chosen by physics + `weights.bc` · L_bc
on the validation set, because the residual alone does not see the boundary values.
`metrics.json` reports for every variant the boundary error max|u − g| over the boundary
points relative to max|u_exact| over the domain, per test example, as mean (`bc_error_mean`)
and maximum (`bc_error_max`) over the test set; with the hard constraint it is 0.

Variant configs inherit from the base YAML through Hydra `defaults`, e.g.
`configs/problem/beam_1d_sine.yaml`. `output_param` is set as an override. The pilot of all
variants on `beam_1d` is summarized in [docs/VariantPilot.md](docs/VariantPilot.md).

`inputs` names what is sampled, so the same name can mean a different family in another
benchmark. Each problem class lists its families in `SUPPORTED_INPUTS`, the first one is the
default; every other value is refused. Only `beam_1d` has a second family so far.

| Benchmark | `inputs` | Sampled operator input |
|---|---|---|
| `beam_1d` | **`linear`** | q = q₀ + q₁x, q₀, q₁ ~ U(0, 1); EI ~ U(0.5, 2) |
| | `sine` | q = Σ_{n=1}^{8} qₙ sin(nπx), qₙ ~ U(−1, 1)/n; EI ~ U(0.5, 2) |
| `heat_1d` | **`sine`** | T₀ = Σ_{n=1}^{5} bₙ sin(nπx), bₙ ~ U(−1, 1) |
| `richards_steady_1d` | **`uniform`** | α ~ U(1, 5), q/Ks ~ U(−0.9, −0.1) |
| `darcy_layered_1d` | **`two_layer`** | k₁, k₂ log-uniform in [0.1, 10], x_s ~ U(0.3, 0.7) |
| `chloride_1d` | **`uniform`** | Cs ~ U(0.5, 1), D ~ U(0.2, 1) |
| `darcy_poisson_2d` | **`sine`** | f = Σ_{m,n=1}^{4} a_mn sin(mπx) sin(nπy), a_mn ~ U(−1, 1) |

Default in bold. `beam_1d` with `inputs: sine`:

| Exact solution | Normalization (loss only) |
|---|---|
| M = Σ qₙ sin(nπx)/(nπ)², w = Σ qₙ sin(nπx)/(EI (nπ)⁴) | q_rms = √(Σqₙ²/2); equilibrium / q_rms, curvature / (M_c/EI) with M_c = q_rms/π²; `data_scale` = (q_rms/(π⁴EI), q_rms/π²), the amplitudes of a single first mode |

Every problem class has a `WRAPPER_VERSION`, increased by hand whenever g, φ or the output
scaling change. `metrics.json` and `history.json` store `wrapper_version`, `inputs` and
`output_param`, and `evaluate.py` refuses a run whose values differ from the current code: a
checkpoint holds only NN, so the old heat_1d checkpoints (φ = t·x(1−x), version 1) gave about
2280 % error with the current wrapper (version 2).

### 3D FNO: memory and runtime (2D + t)

The problems in time over two space dimensions use a 3D FNO over (x, y, t); time is the last
grid axis (neuraloperator 2.0.0 is only correct for odd sizes in the last dimension, see the
grid check in `models/factory.py`). Measured with `scripts/measure_step.py` on an NVIDIA
A100-PCIE-40GB: one training step (prediction with the hard constraint, `hybrid` loss,
backward, Adam) of `fno_small` (4 layers, 32 channels, padding 0.25) for chloride_corner_2d
(5 input channels), peak allocated memory and mean time after a warm-up:

| Grid (x × y × t) | Batch | n_modes 16 (18.9 M parameters) | n_modes 8 (2.6 M) |
|---|---|---|---|
| 32 × 32 × 32 | 4 | 1.5 GiB, 34 ms | |
| 32 × 32 × 32 | 8 | 2.8 GiB, 57 ms | 2.6 GiB, 55 ms |
| 32 × 32 × 32 | 16 | 5.4 GiB, 106 ms | 5.2 GiB, 102 ms |
| 32 × 32 × 64 | 8 | | 5.2 GiB, 105 ms |
| 64 × 64 × 32 | 4 | 5.4 GiB, 106 ms | |
| 64 × 64 × 32 | 8 | 10.6 GiB, 202 ms | 10.4 GiB, 199 ms |
| 64 × 64 × 32 | 16 | 20.9 GiB, 394 ms | 20.7 GiB, 391 ms |
| 64 × 64 × 64 | 4 | 10.5 GiB, 208 ms | |
| 64 × 64 × 64 | 8 | 20.8 GiB, 405 ms | |
| 64 × 64 × 64 | 16 | out of memory | |

Memory and time grow linearly with the number of grid points × batch (about 7 ms per example
at 32³); the number of modes changes the parameters but hardly the time (n_modes 12: same
times, 8.3 M parameters). 1000 examples × 500 epochs as in 1D would take 60 min at 32³ and
3.5 h at 64 × 64 × 32.

**Choice** for heat_2d and chloride_corner_2d: grid **32 × 32 × 32**, batch **8**,
`fno_small` with n_modes **16** per axis (= N/2), **200 epochs** of 1000 examples, i.e.
125 steps × 57 ms ≈ 7 s per epoch, ≈ 25 min per run including validation. The problem YAMLs set
`epochs` and `batch_size` in their optional `training` block, which replaces the training
config unless `training.<key>` is given on the command line (allowed keys: `epochs`,
`batch_size`, `lr`, `n_train`, `n_val`, `n_test`, `eval_every`). On this grid the fastest
heat_2d mode decays by exp(−178 Δt) = 0.751 per time step, and the thinnest chloride front
(2√(D_min t₀) = 0.179) spans 5.5 grid points. The resolution study scales space and time
together: 24³, 32³, 48³, 64³.

## Restore points (Sicherungspunkte)

The state after Phase 3 is tagged `pre-variants` (on `main`); the variant work lives on
`feature/variants`. The outputs of that state are archived outside the repository.

```bash
git checkout main                  # or: git checkout pre-variants
git branch -D feature/variants     # discard the variant branch (from main)
tar -xzf ../pino-outputs-pre-variants.tar.gz   # restore outputs/ (run in the repository root)
```

## Benchmarks

All quantities are non-dimensional. The reference values are set in the YAML files and are
only used to show the plots in physical units.

| Benchmark | Equation | Operator input | Analytical solution | Main variable |
|---|---|---|---|---|
| `heat_1d` | ∂T/∂t = a ∂²T/∂x², T(0,t) = T(1,t) = 0 | T₀(x) = Σₙ bₙ sin(nπx), N = 5, bₙ ~ U(−1, 1) | T = Σₙ bₙ sin(nπx) exp(−a(nπ)²t) | T |
| `beam_1d` | M″ = −q, w″ = −M/EI, w = M = 0 at both supports | q(x) = q₀ + q₁x with q₀, q₁ ~ U(0, 1); EI ~ U(0.5, 2) | w = q₀x(1−2x²+x³)/(24EI) + q₁x(7−10x²+3x⁴)/(360EI), M = q₀x(1−x)/2 + q₁x(1−x²)/6 | w |
| `darcy_layered_1d` | −(k h′)′ = 0, h(0) = 1, h(1) = 0 | k₁, k₂ log-uniform in [0.1, 10], layer interface xₛ ~ U(0.3, 0.7) | q = 1/(xₛ/k₁ + (1−xₛ)/k₂); h = 1 − qx/k₁ for x < xₛ, else h(xₛ) − q(x−xₛ)/k₂ | h |
| `richards_steady_1d` | d/dz[K(h)(h′ + 1)] = 0, K = Ks e^{αh}, h(0) = 0, h(1) analytical | α ~ U(1, 5), q/Ks ~ U(−0.9, −0.1) | Φ = e^{αh} = −q/Ks + (1 + q/Ks) e^{−αz}, h = ln(Φ)/α | h |
| `chloride_1d` | ∂C/∂t = D ∂²C/∂x², x ∈ [0, 1], t ∈ [0.01, 0.05]; C(0, t) = Cs, C(1, t) and C(x, t₀) analytical | Cs ~ U(0.5, 1), D ~ U(0.2, 1) | C = Cs erfc(x / (2√(Dt))) (Crank, C₀ = 0) | C |
| `darcy_poisson_2d` | −Δh = f on [0, 1]², h = 0 on the boundary, k = 1 | f = Σ_{m,n=1}^{4} a_mn sin(mπx) sin(nπy), a_mn ~ U(−1, 1) | h = Σ a_mn / (π²(m² + n²)) sin(mπx) sin(nπy) | h |
| `heat_2d` | ∂T/∂t = a ΔT on [0, 1]², t ∈ [0, 0.05], a = 1, T = 0 on the boundary | T₀ = Σ_{m,n=1}^{3} b_mn sin(mπx) sin(nπy), b_mn ~ U(−1, 1) | T = Σ b_mn sin(mπx) sin(nπy) exp(−aπ²(m² + n²)t) | T |
| `chloride_corner_2d` | ∂C/∂t = D ΔC on [0, 1]², t ∈ [0.02, 0.05]; C = Cs on x = 0 and y = 0, x = 1, y = 1 and C(x, y, t₀) analytical | Cs ~ U(0.5, 1), D ~ U(0.4, 1) | C = Cs [1 − erf(x/(2√(Dt))) · erf(y/(2√(Dt)))] (product of two 1D solutions, C₀ = 0) | C |

| Benchmark | FNO | Hard constraint | Residual normalization | Reference values |
|---|---|---|---|---|
| `heat_1d` | 2D over (x, t), input T₀, x, t/t_end | `dirichlet_1d_t`: T = T₀ + (t/t_end)·x(1−x)·NN | divided by a(Nπ)², the fastest decay rate | non-dimensional |
| `beam_1d` | 1D, input q, EI, x; output w, M | w, M = x(1−x)·c·NN with c = w_c or M_c | curvature by M_c/EI, equilibrium by q̄ = q₀ + q₁/2 | L = 5 m, q_ref = 10 kN/m, EI_ref = 10⁴ kNm²; w in mm, M in kNm |
| `darcy_layered_1d` | 1D, input log₁₀ k, x | h = (1−x) + x(1−x)·NN | flux form: deviation of each face flux from the mean flux, divided by the effective conductivity | L = 10 m, h in m, k_ref = 10⁻⁵ m/s |
| `richards_steady_1d` | 1D, input α, q/Ks, z | h = h(1)·z + z(1−z)·NN | in units of Ks, no further scaling | L = 1 m, Ks = 10⁻⁵ m/s |
| `chloride_1d` | 2D over (x, t), input Cs, D, x, τ | `dirichlet_1d_t` with C(x, t₀), Cs and C(1, t); φ = τ·x(1−x), τ = (t − t₀)/(t_end − t₀) | per sample divided by Cs/t₀: ∂C/∂t peaks at ≈ 0.24 Cs/t for every D | L = 200 mm, D_ref = 10⁻¹² m²/s, t_ref = L²/D_ref = 4·10¹⁰ s ≈ 1268 a, so t ∈ [12.7, 63.4] a; C in M.-%/binder |
| `darcy_poisson_2d` | 2D over (x, y), input f, x, y | `dirichlet_2d` (Coons patch, here g = 0), φ = x(1−x)y(1−y) | per sample divided by the RMS of f, √(Σa²)/2 | L = 10 m, H = 1 m; f = source / k in 1/m, normalized f̂ = f L²/H, so f = 0.01 f̂ 1/m |
| `heat_2d` | 3D over (x, y, t), input T₀, x, y, t/t_end | `dirichlet_2d_t` with u₀ = T₀ and zero edges: T = T₀ + τ·x(1−x)y(1−y)·NN | divided by aπ²·2M² = 178, the fastest decay rate (as heat_1d) | L and ΔT in the YAML (1 m, 1 K); time non-dimensional, t̂ = a t/L² |
| `chloride_corner_2d` | 3D over (x, y, t), input Cs, D, x, y, τ | `dirichlet_2d_t` with C(x, y, t₀), Cs on x = 0 and y = 0, analytical values on x = 1 and y = 1 | per sample divided by Cs/t₀ (as chloride_1d) | as chloride_1d: L = 200 mm, D_ref = 10⁻¹² m²/s, t_ref ≈ 1268 a, so t ∈ [25.3, 63.4] a; C in M.-%/binder |

The normalizations are explained in the docstrings of the problem classes.

### Generic helpers

| Helper | Purpose |
|---|---|
| `numerics.constraints.dirichlet_1d` | stationary 1D, Dirichlet values on both ends |
| `numerics.constraints.dirichlet_1d_t` | 1D in time: initial profile u₀(x) and time-dependent end values u_L(t), u_R(t) |
| `numerics.constraints.dirichlet_2d` | stationary 2D: transfinite (Coons) interpolation of four edge functions |
| `numerics.constraints.dirichlet_2d_t` | 2D in time: g = u₀(x, y) + Coons(edges(t)) − Coons(edges(t₀)), φ = τ·x(1−x)·y(1−y) |
| `numerics.derivatives.laplacian` | second-order Laplacian in any number of dimensions |
| `numerics.derivatives.face_flux`, `conservative_divergence` | fluxes k(u′ + g) on faces and their divergence |
| `pdes.diffusion.diffusion_residual` | ∂u/∂t − D·Δu in any number of space dimensions (time last), Δ via `laplacian`; D scalar or per sample |

### Planned

`plane_stress_bending_2d`: a
**plane-stress disk** (Navier–Lamé equations for the displacements u and v) under pure bending.
It is not a plate; there is no plate-bending benchmark.


## Data flow

```mermaid
flowchart TD
    cfg["configs/*.yaml<br/>problem · model · training"] --> problem["Problem<br/>problems/&lt;area&gt;.py"]
    cfg --> model["FNO<br/>models/factory.py"]
    cfg -- "training.weights, problem loss_weights" --> loss

    problem -- "sample_inputs, analytical" --> data["OperatorDataset<br/>data/dataset.py"]
    data -- "inputs" --> fnoin["to_fno_input<br/>fields + coordinates"]
    fnoin --> model
    model -- "NN" --> constraint["u = g + φ·NN<br/>numerics/constraints.py"]
    problem -- "boundary: g, φ" --> constraint

    constraint -- "prediction u" --> loss["losses.py<br/>physics | data | hybrid"]
    problem -- "residual (numerics/derivatives.py)" --> loss
    data -- "analytical target (data, hybrid)" --> loss
    loss --> trainer["trainer.py<br/>Adam, cosine schedule"]
    trainer -- "weights" --> model

    trainer -- "every eval_every epochs" --> val["evaluation.validate<br/>validation set: physics loss, rel. L2"]
    val -- "on improvement" --> ckpt["best_model.pt, history.json"]
    ckpt -- "once, after training" --> eval["evaluation.evaluate_model<br/>test set + resolution study"]
    eval --> out["metrics.json + plots<br/>visualization.py"]
    out --> summary["scripts/summarize.py<br/>summary.csv, .md, .png"]
```

Trainer, losses, evaluation and plots talk to a benchmark only through the `Problem`
interface. Benchmark-specific code lives only in `problems/<area>.py`,
`analytical/<file>.py` and `configs/problem/<name>.yaml`.

To add a benchmark, see [docs/AddBenchmark.md](docs/AddBenchmark.md).
