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
   | `boundary(inputs, grid)` | `(g, φ)` for `u = g + φ·NN`; for stationary 1D problems use `numerics.constraints.dirichlet_1d` |
   | `to_fno_input(inputs, grid)` | network input `(B, in_channels, *resolution)` including coordinates |
   | `data_scale(inputs, grid)` | optional: typical size of each output channel for the data loss |
   | `n_dims`, `in_channels`, `out_channels` | set the FNO dimension and the channels |
   | `plot_type` | `"1d"` or `"1d_t"`; `"2d"` and `"2d_t"` are not yet in `visualization.SOLUTION_PLOTTERS` |
   | `variable_names`, `units`, `scales` | axis labels and conversion to physical units |
   | `main_variable` | optional, defaults to the first output variable |

3. **Config** `configs/problem/<name>.yaml` with `name`, the parameters, the reference values,
   `train_resolution`, `test_resolution` and `resolution_study`.
4. **Tests** in `tests/`: PDE satisfied via autograd, boundaries exact, FD residual of the
   analytical solution small and convergent, shapes and hard constraint.
5. Train with `python scripts/train.py problem=<name> training=physics`.

The network gets no coordinates from neuraloperator (`positional_embedding=None`), so
`to_fno_input` has to append the coordinates as channels itself.

Grid sizes must be even in every dimension except the last one: neuraloperator 2.0.0
computes wrong spectral convolutions on odd grids there, and the model factory rejects them.
