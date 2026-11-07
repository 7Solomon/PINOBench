# Adding a benchmark

1. **Analytical solution** as a pure function in `src/pino/analytical/<file>.py`, without any
   dependency on the network.
2. **Problem class** in `src/pino/problems/<area>.py`, derived from `Problem` and registered
   with `@register_problem("<name>")`. New modules in `problems/` are imported automatically,
   so there is no central list. To implement:

   | Method / attribute | Purpose |
   |---|---|
   | `grid(resolution)` | coordinates `(n_dims, *resolution)` including the boundary, via `numerics.grids.uniform_grid` |
   | `sample_inputs(n, generator)` | operator inputs `(B, ...)` |
   | `analytical(inputs, grid)` | exact solution `(B, C, *resolution)` |
   | `residual(u, inputs, grid)` | normalized PDE residual on the interior points |
   | `WRAPPER_VERSION` | string, increase it whenever `boundary()` or `output_scale()` change |
   | `boundary(inputs, grid)` | `(g, φ)` for `u = g + φ·NN`, φ only the fixed shape part; use `numerics.constraints.dirichlet_1d`, `dirichlet_1d_t`, `dirichlet_2d` or `dirichlet_2d_t` |
   | `to_fno_input(inputs, grid)` | network input `(B, in_channels, *resolution)` including coordinates |
   | `output_scale(inputs, grid)` | optional: per-sample scale s of the output, `constrained` only |
   | `data_scale(inputs, grid)` | optional: typical size of each output channel for the data loss |
   | `SUPPORTED_INPUTS` | required: tuple of input family names, named after what is sampled (e.g. `("linear", "sine")`); the first one is the original family and the default |
   | `n_dims`, `in_channels`, `out_channels` | set the FNO dimension and the channels |
   | `plot_type` | `"1d"`, `"1d_t"`, `"2d"` or `"2d_t"` (writes `solution_best/median/worst.png`); for problems in time, time is the last grid axis |
   | `variable_names`, `units`, `scales` | axis labels and conversion to physical units |
   | `main_variable` | optional, defaults to the first output variable |

3. **Config** `configs/problem/<name>.yaml` with `name`, the parameters, the reference values,
   `train_resolution`, `test_resolution` and `resolution_study`, plus `inputs` (the first entry
   of `SUPPORTED_INPUTS`, it appears in the run directory),
   `output_param` and `output_scale_global` (see the README). Optional: `loss_weights` for
   the `hybrid` mode.
4. **Tests** in `tests/`: PDE satisfied via autograd, boundaries exact, FD residual of the
   analytical solution small and convergent, shapes and hard constraint.
5. Train with `python scripts/train.py problem=<name> training=physics`.

The network gets no coordinates from neuraloperator (`positional_embedding=None`), so
`to_fno_input` has to append the coordinates as channels itself.

Grid sizes must be even in every dimension except the last one: neuraloperator 2.0.0
computes wrong spectral convolutions on odd grids there, and the model factory rejects them.
