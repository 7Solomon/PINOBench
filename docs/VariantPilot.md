# Variant pilot: beam_1d

Does beam_1d still train when less physics sits in the output wrapper? Five stages, seeds 42, 0
and 1, identical model (`fno_small`), epochs (500) and learning rate (1e-3) everywhere. Test set
rel. L2, mean over the seeds with [min–max].

| Stage | inputs | output_param | Mode | w | M |
|---|---|---|---|---|---|
| 1 | linear | constrained | data | 0.04 % [0.03–0.06] | 0.06 % [0.05–0.09] |
| 1 | linear | constrained | physics | 0.67 % [0.30–0.88] | 0.51 % [0.36–0.73] |
| 2 | linear | unscaled | data | 0.41 % [0.18–0.76] | 0.27 % [0.14–0.44] |
| 2 | linear | unscaled | physics | 2.43 % [2.11–2.61] | 1.10 % [0.59–1.53] |
| 3 | sine | constrained | data | 1.21 % [0.94–1.47] | 1.00 % [0.84–1.16] |
| 3 | sine | constrained | physics | **10.6 % [8.8–13.0]** | **5.3 % [4.2–6.9]** |
| 4 | sine | unscaled | data | 0.62 % [0.49–0.75] | 0.44 % [0.38–0.49] |
| 4 | sine | unscaled | physics | **9.7 % [6.2–12.6]** | 4.0 % [3.3–5.4] |
| 5 | sine | direct | data | 0.51 % [0.48–0.57] | 0.40 % [0.37–0.43] |

Findings:

- `data` reaches < 1.5 % in every stage, also without any constraint (`direct`, boundary values
  up to 15 % of the maximum in the worst sample, 0.6 % on average).
- `physics` breaks with the sine inputs (stages 3 and 4), with and without the per-sample scale.
  The physics loss plateaus near 1e-2 while the exact solution scores 4e-6, train and validation
  agree: an optimization problem, not overfitting or discretization. Learning rates 3e-4 and
  1e-4 (seed 42) give 8.5–10.9 % as well.
- With sine inputs the per-sample scale hurts: s = q_rms/(π⁴ EI) follows the size of w only
  within a factor of 7 (linear: 0.1 %), and dividing by q_rms turns the linear operator into a
  nonlinear target. Doubling the load changes the prediction by 2× only within 4.7 % (stage 3,
  data) against 0.7 % (stage 4).
- 1D GPU training is not bit-reproducible (non-deterministic CUDA kernels); single runs of the
  same seed differ, e.g. stage 1 physics seed 42: 0.43 %, 1.39 %, 0.82 %.

The raw runs are under `outputs/beam_1d/<inputs>-<output_param>/`.
