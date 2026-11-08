# thermoelastic_block_2d: the block of elastic_block_2d, heated on top

Seed 42, `fno_small` (`n_modes` 16), 500 epochs, lr 1e-3, batch 32, 1000 training examples, grid
64 × 64, input family `cosine`. One network with the outputs T, u and v. The equations, the
supports and the hard constraint are in the README.

**Result: all three `data` variants meet the 5 % criterion in T, u and v (`constrained`: v 0.62 %,
T 0.33 %, u 0.69 %). `constrained` `physics` misses it clearly (v 15.4 %), and `hybrid` misses it
narrowly (v 5.54 %).** As agreed for the mechanics benchmarks, the `physics` result is reported
without further tuning. σ_xx is predicted well; σ_yy and σ_xy are not, see below.

## Results

Rel. L2 of the test set, mean over 200 examples. All stresses are computed by finite differences
from the predicted displacements and the predicted temperature. `stress_rmse` is the root mean
square error of the stress in MPa, over all test points. Reference values of the plots and of
`stress_rmse`: L = H = 2 m, E = 30 GPa, α = 10⁻⁵/K, ΔT_ref = 20 K, so 0.4 mm and 6 MPa.

| Variant | Mode | neumann | v | v worst | T | u | σ_xx | RMSE σ_xx / σ_yy / σ_xy [MPa] | Time |
|---|---|---|---|---|---|---|---|---|---|
| constrained | physics | 10 | **15.37 %** | 34.3 % | 8.48 % | 37.5 % | 8.64 % | 0.223 / 0.104 / 0.061 | 2770 s |
| constrained | data | 0 | **0.62 %** | 2.86 % | 0.33 % | 0.69 % | 1.62 % | 0.040 / 0.107 / 0.055 | 1889 s |
| constrained | hybrid | 10 | 5.54 % | 19.0 % | 1.60 % | 5.51 % | 2.56 % | 0.064 / 0.130 / 0.041 | 2690 s |
| soft (bc 1) | physics | 10 | 20.30 % | 33.5 % | 13.0 % | 65.5 % | 12.5 % | 0.331 / 0.097 / 0.051 | 2855 s |
| soft (bc 1) | data | 0 | 0.43 % | 1.45 % | 0.32 % | 0.65 % | 1.51 % | 0.038 / 0.118 / 0.049 | 2056 s |
| direct | data | 0 | 0.36 % | 1.26 % | 0.31 % | 0.56 % | 1.30 % | 0.032 / 0.094 / 0.041 | 2047 s |

Boundary error: max |u − g| relative to max |u_exact|, mean / max over the test set. Neumann error:
max deviation on the edge relative to the RMS of T_top, mean over the test set.

| Variant | Mode | Boundary T | Boundary u | Boundary v | T_x left / right | σ_xy left / right / bottom / top | σ_yy top |
|---|---|---|---|---|---|---|---|
| constrained | physics | 0 | 0 | 0 | 4.2 / 4.6 % | 1.9 / 1.8 / 0.2 / 4.3 % | 3.4 % |
| constrained | data | 0 | 0 | 0 | 48.6 / 52.4 % | 13.6 / 14.7 / 0.6 / 16.3 % | 32.9 % |
| constrained | hybrid | 0 | 0 | 0 | 7.7 / 6.6 % | 2.2 / 2.0 / 0.5 / 5.2 % | 9.0 % |
| soft (bc 1) | physics | 9.3 / 45.2 % | 3.3 / 15.2 % | 0.9 / 2.3 % | 5.5 / 4.6 % | 1.2 / 1.2 / 0.3 / 4.1 % | 3.8 % |
| soft (bc 1) | data | 0.34 / 7.4 % | 0.16 / 1.0 % | 0.07 / 0.18 % | 47.2 / 47.8 % | 10.6 / 10.1 / 1.2 / 12.2 % | 33.5 % |
| direct | data | 1.04 / 8.1 % | 0.53 / 3.3 % | 0.19 / 0.58 % | 66.4 / 67.8 % | 8.8 / 7.4 / 2.2 / 9.7 % | 23.6 % |

Resolution study, training at 64 × 64:

| Variant | Mode | v at 32 / 64 / 128 / 256 | T at 32 / 64 / 128 / 256 | u at 32 / 64 / 128 / 256 |
|---|---|---|---|---|
| constrained | physics | 15.6 / 15.4 / 15.3 / 15.3 % | 8.41 / 8.48 / 8.53 / 8.55 % | 37.9 / 37.5 / 37.4 / 37.3 % |
| constrained | data | 0.85 / 0.62 / 0.69 / 0.78 % | 0.47 / 0.33 / 0.38 / 0.42 % | 3.09 / 0.69 / 1.66 / 2.37 % |
| constrained | hybrid | 5.43 / 5.54 / 5.64 / 5.69 % | 1.72 / 1.60 / 1.56 / 1.54 % | 7.05 / 5.51 / 5.23 / 5.23 % |
| soft (bc 1) | physics | 21.0 / 20.3 / 19.9 / 19.8 % | 13.3 / 13.0 / 12.9 / 12.8 % | 65.6 / 65.5 / 65.5 / 65.5 % |
| soft (bc 1) | data | 1.20 / 0.43 / 0.71 / 0.97 % | 1.05 / 0.32 / 0.60 / 0.81 % | 3.44 / 0.65 / 1.78 / 2.56 % |
| direct | data | 1.18 / 0.36 / 0.69 / 0.96 % | 1.06 / 0.31 / 0.58 / 0.80 % | 3.48 / 0.56 / 1.79 / 2.58 % |

Caveats: one seed per configuration. Differences below a factor of about 1.5 are not significant
(see VariantPilot.md), so the `data` variants cannot be ranked with confidence.

## Weight search for L_neumann

`constrained`, `physics`, seed 42.

| neumann | v | T | u | σ_xx | RMSE σ_xx / σ_yy / σ_xy [MPa] | σ_yy top | Time |
|---|---|---|---|---|---|---|---|
| 1 | 15.29 % | 6.22 % | 71.8 % | 13.2 % | 0.328 / 0.204 / 0.110 | 8.1 % | 2561 s |
| **10** | **15.37 %** | 8.48 % | 37.5 % | 8.64 % | 0.223 / 0.104 / 0.061 | 3.4 % | 2770 s |
| 100 | 16.51 % | 10.1 % | 40.3 % | 9.67 % | 0.252 / 0.084 / 0.058 | 1.9 % | 2703 s |

`loss_weights.neumann: 10` is in the YAML. By v alone, 1 is ahead by 0.08 percentage points, which
is no difference at one seed. 10 was chosen because it halves the error of u and improves σ_xx and
the edge conditions at the same v. Only T favours 1. The `bc` weight for `soft` stayed at 1, as for
the other 2D benchmarks, and was not searched.

## Physics of the benchmark

- **A harmonic temperature leaves a free body almost free of stress.** In plane stress a steady
  temperature field causes no stress in a simply connected free body. Here only the supports
  restrain it, so the stresses are very unequal: RMS over the test family σ_xx 2.6 MPa, σ_yy
  0.028 MPa, σ_xy 0.016 MPa.
- **The stress function does not depend on ν**, and T enters only through v(x, 0) = 0, which gives
  one inhomogeneous condition on F‴(0) per mode (see `analytical/elasticity.py`). Mode 0 is simple:
  T = t₀y/H, u = 0, v = (1+ν)t₀y²/(2H), σ_xx = −t₀y/H, σ_yy = σ_xy = 0.
- **This is why `stress_rmse` exists.** The relative error of σ_yy and σ_xy divides by almost zero:
  the exact solution itself has a finite-difference floor of 16.9 % and 14.1 % on the 64 grid
  (σ_xx: 0.05 %), and the networks show means of about 500–1500 %. These numbers stay in
  `metrics.json` and `stress.png`, but say little. Compare in MPa instead. The floor corresponds to
  about 0.005 and 0.002 MPa.

## Stresses against displacements

- σ_xx follows the displacements: in `data` it is 1.3–1.6 % at v 0.36–0.62 %, the best result here.
- **σ_yy and σ_xy are not predicted usefully by any variant.** Their error of about 0.09–0.13 MPa
  and 0.04–0.06 MPa is about 3–5 times the signal, and about 20 times the floor of the finite differences,
  so it comes from the noise of the prediction. Relative to σ_xx it is small, about 4 % and 2 %.
- The error of σ_yy hardly depends on the mode. In `elastic_block_2d` `hybrid` improved σ_yy and
  σ_xy over `data`. Here it improves σ_xy (0.041 against 0.055 MPa) and worsens σ_yy (0.130 against
  0.107 MPa). The lowest σ_yy error comes from neumann = 100 in the search (0.084 MPa).
- What the physics term does improve is the edges: σ_yy on top 3.4 % against 32.9 % in `data`, and
  T_x 4–5 % against about 50 %.

## Why physics stagnates

No tuning was done, as agreed, so this describes what the runs show and is not a tested cause.

- **The analytical solution and the residual are consistent:** sympy derivation, autograd for T and
  Navier–Lamé, all boundary conditions; the FD residual of the exact solution converges. Its mean
  square is about 8e-6, against 0.012 for the best `physics` checkpoint (best epoch 160 of 500).
- **The residual is a poor guide to the error:** `data` has v 0.62 % at a residual of 0.72, while
  `physics` has v 15.4 % at 0.012, and `hybrid` v 5.54 % at 0.028. Small high-frequency noise
  dominates the residual, as in `elastic_block_2d`, while long-wave errors barely show in it.
- **Even T alone is not solved,** though it is a Laplace equation that does not depend on u and v.
  g = (y/H)·T_top already has an error of about 17 %; `physics` only reaches 8.5 %.
- **The temperature error drives the mechanics,** because thermal strain from the predicted T enters
  the Navier–Lamé residual. Unlike in `elastic_block_2d`, mode 0 of v is visible to the residual
  here; this did not help, and v is worse than the 9.06 % there.

`hybrid` has T 1.6 % and v 5.54 %. Its weights are not calibrated (physics 1, neumann 10, data 1,
while the data loss is about ten times smaller than the physics loss) and were not tuned.

## Runtime

The target was about 30 minutes. `data` took 31–34 minutes, `physics` and `hybrid` 43–48 minutes.
These times are not clean: a foreign job on the same A100 (34 processes, about 14 GB) ran
throughout, and two lanes of this benchmark shared the GPU nearly all the time.

## Where the runs are

The kept runs are under `outputs/thermoelastic_block_2d/cosine-<output_param>/`. The search runs
with neumann 1 and 100 are under `outputs/_superseded/neumann_ladder/`, ignored by `summarize.py`.
