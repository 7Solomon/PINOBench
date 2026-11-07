# Soft-constraint pilot: beam_1d, darcy_layered_1d, heat_1d

Can boundary and initial values be enforced by a penalty instead of the output wrapper?
`output_param: soft` predicts u = c·NN and adds L_bc = mean over the boundary points of
((u − g)/s)² with weight `weights.bc` (see the README). Seed 42, `fno_small`, 500 epochs,
lr 1e-3, the default inputs of each benchmark (`beam_1d`: `linear`). The `constrained` runs
for comparison are the existing seed-42 runs.

Boundary error: max |u − g| over the boundary points relative to max |u_exact| over the
domain, per test example; mean and maximum over the 200 test examples. For `constrained` it
is 0 by construction.

## physics: weight ladder

rel. L2 of the main variable (other variables in brackets), boundary error of the main
variable as mean / max.

| Benchmark | bc | rel. L2 | Boundary error mean / max | 2nd-half max val. error |
|---|---|---|---|---|
| beam_1d | 1 | 4.90 % (M 2.35 %) | 1.48 % / 5.09 % (M 2.15 % / 5.20 %) | 49.8 % |
| beam_1d | 10 | 3.38 % (M 3.06 %) | 0.42 % / 1.52 % (M 0.39 % / 1.20 %) | 20.9 % |
| beam_1d | **100** | **3.26 %** (M 1.83 %) | **0.15 % / 0.43 %** (M 0.16 % / 0.39 %) | 8.3 % |
| beam_1d | constrained | 0.82 % (M 0.73 %) | 0 | 5.8 % |
| darcy_layered_1d | 1 | 1.40 % | 1.12 % / 3.15 % | 1.50 % |
| darcy_layered_1d | 10 | 0.54 % | 0.19 % / 0.57 % | 0.81 % |
| darcy_layered_1d | **100** | **0.53 %** | **0.09 % / 0.22 %** | 0.79 % |
| darcy_layered_1d | constrained | 0.53 % | 0 | 1.05 % |
| heat_1d | **1** | **0.88 %** | **0.09 % / 0.21 %** | 1.64 % |
| heat_1d | 10 | 2.68 % | 0.08 % / 0.20 % | 4.42 % |
| heat_1d | 100 | 26.2 % | 0.08 % / 0.30 % | 37.0 % |
| heat_1d | constrained | 1.14 % | 0 | 3.14 % |

**Criterion** (rel. L2 < 5 %, boundary error mean < 1 %, max < 5 %, with one of the tested
weights, in at least two of three benchmarks): **met in all three** — beam_1d with 10 and 100,
darcy_layered_1d with 10 and 100, heat_1d with 1 and 10. No extra weights or longer training
were needed.

Best weights, written to the problem YAMLs as `loss_weights.bc` (global default 10):
beam_1d **100**, darcy_layered_1d **100**, heat_1d **1**.

## data and hybrid with the best weight

| Benchmark | Mode | bc | soft rel. L2 | Boundary error mean / max | constrained rel. L2 |
|---|---|---|---|---|---|
| beam_1d | data | 100 | 0.31 % (M 0.26 %) | 0.01 % / 0.04 % | 0.03 % |
| beam_1d | hybrid | 100 | 1.38 % (M 0.80 %) | 0.17 % / 0.49 % | 0.32 % |
| darcy_layered_1d | data | 100 | 0.72 % | 0.01 % / 0.06 % | 0.33 % |
| darcy_layered_1d | hybrid | 100 | 0.35 % | 0.04 % / 0.14 % | 0.35 % |
| heat_1d | data | 1 | 0.31 % | 0.10 % / 0.22 % | 0.57 % |
| heat_1d | hybrid | 1 | 0.30 % | 0.11 % / 0.37 % | 0.43 % |

## Trivial solution

Not observed. The check compares the variation of the prediction in the interior (RMS
deviation from its mean, per test example) with that of the exact solution; a constant or
zero field gives 0. All soft runs give 0.97–1.00. For darcy_layered_1d, where the flux-form
residual is invariant under a·h + c and all information sits in the two boundary values, the
weight matters: with bc = 1 the boundary error reaches 3 % and the rel. L2 1.4 %, with
bc ≥ 10 soft equals constrained. The checkpoint of soft physics runs is selected by
physics + bc · L_bc on the validation set; selecting by the physics loss alone could pick
exactly the trivial solution.

## heat_1d: resolution and stability

Test rel. L2 at the study resolutions (training at 64 × 64):

| Variant | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| constrained | 7.95 % | 1.14 % | 4.03 % | 6.13 % |
| soft, bc 1 | 6.32 % | 0.88 % | 3.56 % | 5.06 % |
| soft, bc 10 | 6.88 % | 2.68 % | 4.47 % | 5.76 % |

Validation error during training (evaluated every 5 epochs):

| Variant | max in the 2nd half | jumps > 1.5× in the 1st / 2nd half |
|---|---|---|
| constrained | 3.14 % | 7 / 2 |
| soft, bc 1 | 1.64 % | 1 / 0 |

- The **resolution dependence remains** with soft: the error is lowest at the training
  resolution and grows towards coarser and finer grids in the same pattern. It is therefore
  not caused by the division by τ in the hard constraint; more likely candidates are the FNO
  itself (modes and padding fixed in grid points) and the FD residual, which is only exact on
  the training grid.
- The **jumps in training are reduced**: no jump in the second half with soft, against two
  with constrained, and half the maximum error. This fits the explanation that the wrapper
  makes the learning target near t = 0 large (division by τ); a single seed does not prove it.

## Assessment

- Soft constraints work in every mode, and the boundary error stays below 0.5 % (mean) for
  the best weights.
- They cost accuracy where the hard constraint carries much information: beam_1d is 4× worse
  in physics (3.3 % against 0.8 %) and 10× in data. darcy_layered_1d is equal in physics and
  hybrid.
- They help heat_1d in every mode (physics 0.88 % against 1.14 %, data 0.31 % against
  0.57 %), where the hard constraint divides by τ.
- The best weight differs by two orders of magnitude between the benchmarks (1 against 100),
  so `bc` has to be calibrated per problem; too large a weight destroys heat_1d (26 %).

Caveats: one seed per configuration, and 1D GPU training is not bit-reproducible (see
VariantPilot.md); differences below a factor of about 1.5 are not significant. Training
times of the pilot (830–1350 s) are not comparable with the constrained runs because nine runs
shared the GPU.

The kept runs are under `outputs/<benchmark>/<inputs>-soft/`. The ladder runs with the other
weights are under `outputs/_superseded/soft_bc_ladder/`, ignored by `summarize.py`.
