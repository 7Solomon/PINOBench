# elastic_block_2d: plane-stress block in a frictionless box

The first 2D mechanics benchmark. Seed 42, `fno_small` (`n_modes` 16), 500 epochs, lr 1e-3,
batch 32, 1000 training examples, grid 64 × 64, input family `cosine`. The equations and the
hard constraint are in the README.

**Result: `constrained` `data` meets the criterion (v 0.34 %, limit 5 %), `constrained` `physics`
does not (v 9.06 %).** The user accepted the `physics` result and stopped the tuning, because
`data` and `hybrid` give working networks for this problem. The criterion itself was not changed:
it is missed on `physics`.

## Results

rel. L2 of the test set, mean over 200 examples. All stresses are computed by finite differences
from the predicted displacements and compared with the analytical stresses. Boundary error:
max |u − g| relative to max |u_exact|. Neumann error: max deviation on the edge relative to the
load RMS, mean over the test set.

| Variant | Mode | neumann | v | v worst | u | σ_xx | σ_yy | σ_xy | Time |
|---|---|---|---|---|---|---|---|---|---|
| constrained | physics | 100 | **9.06 %** | 24.4 % | 77.0 % | 21.6 % | 7.4 % | 38.5 % | 2538 s |
| constrained | data | 0 | **0.34 %** | 0.92 % | 0.81 % | 6.6 % | 6.7 % | 39.7 % | 1583 s |
| constrained | hybrid | 100 | 4.58 % | 17.1 % | 15.6 % | 8.2 % | 4.7 % | 22.1 % | 2884 s |
| soft (bc 1) | physics | 100 | 15.5 % | 36.6 % | 80.4 % | 21.9 % | 7.7 % | 37.1 % | 2612 s |
| soft (bc 1) | data | 0 | 0.27 % | 0.86 % | 0.68 % | 4.3 % | 4.4 % | 31.6 % | 2014 s |
| direct | data | 0 | 0.27 % | 0.73 % | 0.70 % | 4.3 % | 4.4 % | 32.7 % | 1864 s |

| Variant | Mode | Boundary error u mean / max | Boundary error v mean / max | Neumann mean: top σ_yy | top σ_xy | left σ_xy | right σ_xy | bottom σ_xy |
|---|---|---|---|---|---|---|---|---|
| constrained | physics | 0 | 0 | 3.7 % | 3.3 % | 2.0 % | 1.7 % | 0.6 % |
| constrained | data | 0 | 0 | 87.9 % | 24.1 % | 23.8 % | 21.1 % | 0.9 % |
| constrained | hybrid | 0 | 0 | 7.4 % | 3.7 % | 1.9 % | 1.8 % | 0.9 % |
| soft (bc 1) | physics | 5.5 / 24.0 % | 4.3 / 12.5 % | 4.2 % | 1.8 % | 1.4 % | 1.4 % | 0.5 % |
| soft (bc 1) | data | 0.14 / 0.83 % | 0.12 / 0.38 % | 48.1 % | 16.4 % | 16.8 % | 16.8 % | 3.8 % |
| direct | data | 0.79 / 5.45 % | 0.37 / 1.02 % | 44.0 % | 15.4 % | 15.6 % | 17.3 % | 12.5 % |

Resolution study, training at 64 × 64:

| Variant | Mode | v at 32 / 64 / 128 / 256 | u at 32 / 64 / 128 / 256 |
|---|---|---|---|
| constrained | physics | 9.15 / 9.06 / 9.02 / 9.01 % | 77.9 / 77.0 / 76.8 / 76.7 % |
| constrained | data | 0.59 / 0.34 / 0.40 / 0.46 % | 6.55 / 0.81 / 3.30 / 4.85 % |
| constrained | hybrid | 4.61 / 4.58 / 4.58 / 4.57 % | 18.2 / 15.6 / 15.0 / 14.9 % |
| soft (bc 1) | physics | 15.6 / 15.5 / 15.4 / 15.4 % | 80.7 / 80.4 / 80.5 / 80.5 % |
| soft (bc 1) | data | 0.79 / 0.27 / 0.46 / 0.60 % | 6.77 / 0.68 / 3.40 / 5.01 % |
| direct | data | 0.77 / 0.27 / 0.45 / 0.60 % | 6.60 / 0.70 / 3.30 / 4.87 % |

Caveats: one seed per configuration. Differences below a factor of about 1.5 are not significant
(see VariantPilot.md), which includes `data` constrained against soft and direct.

## Weight search for L_neumann

`constrained`, `physics`, seed 42. The experiment with 1000 was outside the required search.

| neumann | v | u | σ_xx | σ_yy | σ_xy | Time |
|---|---|---|---|---|---|---|
| 1 | 20.5 % | 99.1 % | 35.4 % | 20.3 % | 82.8 % | 1806 s |
| 10 | 11.1 % | 84.1 % | 25.2 % | 9.8 % | 52.4 % | 2714 s |
| **100** | **9.06 %** | 77.0 % | 21.6 % | 7.4 % | 38.5 % | 2538 s |
| 1000 | 12.8 % | 85.2 % | | | | 2106 s |

`loss_weights.neumann: 100` is in the YAML. Global default in `physics` and `hybrid`: 10.
The `bc` weight for `soft` stayed at 1 as for the other 2D benchmarks; {1, 10, 100} were not tested,
because the soft runs are not binding and the tuning was stopped. The boundary error of
`soft` `physics` (4–5 % mean, up to 24 %) shows that bc = 1 is too weak next to neumann = 100.

## Why physics stagnates

Working through the checklist of the task:

1. **Analytical solution: correct.** sympy derivation, autograd per mode, all six boundary
   conditions, stability at large λ; the FD residual of the exact solution is small and convergent.
2. **Normalization:** residual and L_neumann are divided by the load RMS per sample.
3. **Weight of L_neumann:** 100 is the best, 1000 is worse (see above).
4. **One-sided derivatives:** second-order convergence is tested. The FD floor of the exact
   solution on the 64 grid is 0.07 / 0.04 / 0.90 % for σ_xx / σ_yy / σ_xy.
5. **Learning rate, batch size:** lr 3e-4 reached about 17 % at epoch 240, batch 8 about 12 % at
   epoch 85 and was too slow for the budget. Both were stopped before the end; neither was ahead.
6. **Model size:** `fno_medium` was not run; the tuning was stopped before that.

The network can represent the solution, as `data` shows with the same model and wrapper. The
limit is the conditioning of the loss:

- The error left over is smooth and long-wave. A second-derivative residual sees an error of
  wavenumber k with weight k⁴ in the squared loss, so it hardly sees long waves. The residual stays
  at 6e-2 against 3e-5 for the exact solution, but the remaining gradient is too small.
- Mode 0 (v ≈ 0.4) is about 20 times larger than the load modes (≈ 0.02) and invisible to the
  residual, because it is linear in y. Only the Neumann term on the top edge determines it.
- u is about 22 times smaller than v and is under-predicted (RMS 0.0063 against 0.0188).

An untested proposal: put mode 0 into `g` of the hard constraint, so that the network only has to
learn the load modes. This changes the wrapper and was deliberately not done.

`hybrid` lies between the two (v 4.58 %) because its weights are not calibrated: physics 1 and
neumann 100 against data 1, while the data loss is much smaller. Its v is 13 times worse than
`data`.

## Stresses against displacements

- In `data` the displacements are very accurate (v 0.34 %, u 0.81 %), but the normal stresses
  are 20 times worse (6.6 %), and σ_xy reaches 40 %, because σ_xy is small (about 1 MPa against
  10 MPa for σ_yy) and FD amplifies the high-frequency noise of the prediction (visible in
  `stress.png`). The Neumann errors of `data` are large for the same reason: 88 % at the top, where
  the one-sided derivative falls on the noisiest part of the field, which is not part of its loss.
- With a physics term the order reverses. `hybrid` has a worse v than `data`, but better
  σ_yy (4.7 %), σ_xy (22 %) and top-edge load (7.4 %). `physics` has the worst displacements, but
  σ_xy (38.5 %) as good as `data` and the top load met to 3.7 %.
- A displacement error therefore says little about the stress error, in either direction.
  Both are reported per run.

## n_modes: a lesson from heat_2d

`n_modes` must stay below N/2: 16 at 64 points, so that the model does not use every frequency
of the training grid. heat_2d ran with 16 at 32 points (= N/2), and its error rises away from the
training grid. Here `physics` and `hybrid` have nearly flat errors in v from 32 to 256 (4.61 to
4.57 % in `hybrid`). `data` still varies, between 0.34 and 0.59 % in v and between 0.81 and 6.55 %
in u, so `n_modes` < N/2 alone does not remove the resolution dependence. Whether the flat errors
come from `n_modes` or from the physics term was not tested separately.

## Runtime

The target was about 30 minutes. `data` took 26–34 minutes, `physics` and `hybrid` 42–48 minutes.
These times are not clean: a foreign job on the same A100 (about 35–60 processes, 14–16 GB) ran
throughout, and some of the runs shared the GPU with parallel runs of this benchmark.

## Where the runs are

The kept runs are under `outputs/elastic_block_2d/cosine-<output_param>/`. The search runs with
neumann 1 and 10 are under `outputs/_superseded/neumann_ladder/`, ignored by `summarize.py`.
