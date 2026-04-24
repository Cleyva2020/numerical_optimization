# MATE 6026 Final Project — Code & Report

Comparison of **Levenberg–Marquardt (LM)**, **Nonlinear Conjugate Gradient (NCG)**,
and **Adam** for training a feedforward neural network on the UCI Concrete
Compressive Strength regression problem, framed as a nonlinear least squares
problem in the network parameters.

- Written report: [report/main.tex](report/main.tex)
- Slides: [presentation/main.tex](presentation/main.tex)
- MATLAB code: [code/](code/)
- Dataset: [data/concrete_data.csv](data/concrete_data.csv) (UCI, 1030 samples × 8 features)

## Repository layout

```
.
├── README.md                          (this file)
├── data/
│   └── concrete_data.csv              UCI Concrete Compressive Strength
├── course_code/                       optimizers from prior assignments (reference only)
├── code/                              project source
│   ├── default_config.m               single-source-of-truth for all hyperparameters
│   ├── data_loader.m                  CSV -> stratified 70/15/15 split, standardize on train
│   ├── nn.m                           forward pass + Glorot init + pack/unpack
│   ├── residual_jacobian.m            analytical r, J via per-sample backprop
│   ├── jacobian_fd.m                  centered FD Jacobian (correctness check only)
│   ├── line_search_wolfe.m            numeric strong Wolfe (zoom + cubic interp)
│   ├── opt_lm.m                       Levenberg-Marquardt
│   ├── opt_ncg.m                      Polak-Ribiere+ NCG with strong Wolfe
│   ├── opt_adam.m                     Adam with mini-batches and val-loss early stopping
│   ├── run_experiments.m              driver: 5 seeds x 3 methods, saves to results/
│   ├── make_plots.m                   produces the 5 figures into report/figures/
│   ├── make_tables.m                  prints LaTeX-ready rows for the 4 tables
│   ├── test_jacobian.m                analytical-vs-FD correctness gate
│   ├── test_optimizers.m              NCG/LM sanity check on a quadratic bowl
│   └── results/                       .mat files, one per (seed, method)
├── report/
│   ├── main.tex                       full report (Introduction, Methods, Results, Conclusions)
│   ├── references.bib
│   └── figures/                       PDFs written by make_plots.m
└── presentation/
    └── main.tex                       Beamer slides (metropolis theme)
```

## Prerequisites

- MATLAB R2020a or newer (uses `exportgraphics`, `readmatrix`, `discretize`).
  No toolboxes required — only base MATLAB.
- For the report/slides: a LaTeX distribution with `pdflatex`, `bibtex`, and
  (for slides) the `metropolis` Beamer theme.

## Quick start

### Running the experiments

From the repository root:

```matlab
cd code
test_jacobian           % gate: analytical J vs centered FD, should print PASSED
test_optimizers         % gate: NCG/LM on a quadratic bowl, should print PASSED
run_experiments         % ~2-5 minutes; tunes Adam LR, then 5 seeds x 3 methods
make_plots              % writes 5 PDFs into ../report/figures/
make_tables             % prints LaTeX rows to paste into report/main.tex
```

`make_tables` accepts an optional loss threshold:

```matlab
make_tables(20)         % use tau_L = 20 for iters/time-to-tol tables
```

Pick a `tau_L` that all three methods reach but that still differentiates
them — 20 is a good starting point for this dataset (the default heuristic
can be too loose).

### Building the PDFs

```bash
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd ../presentation && pdflatex main.tex
```

## What each method does, in one sentence

- **LM** ([opt_lm.m](code/opt_lm.m)): solve the damped Gauss-Newton system
  `(J'J + lambda*I) p = -J'r`, accept/reject via the gain ratio, decrease
  lambda toward pure Gauss-Newton when the model predicts accurately and
  increase it toward scaled steepest descent when it doesn't.
- **NCG** ([opt_ncg.m](code/opt_ncg.m)): conjugate-gradient direction with
  Polak-Ribière<sup>+</sup> coefficient and strong-Wolfe step length, with
  a periodic restart every `n` iterations.
- **Adam** ([opt_adam.m](code/opt_adam.m)): mini-batch stochastic gradient
  with adaptive per-parameter step sizes from first- and second-moment
  estimates; early stop on validation loss.

## Shared infrastructure

All three optimizers call the same `residual_jacobian` for their gradient
and (for LM) Jacobian needs. This is important: it keeps the comparison
fair by construction — no subtle differences in the forward pass or loss
definition can contaminate the methods. The shared pieces:

- **Loss**: `L(theta) = 0.5 * ||r(theta)||^2` (no 1/N normalization, to
  match the Ch. 10 NLS formulation in Nocedal & Wright).
- **Residual**: `r = y - f(X; theta)`, shape `N x 1`.
- **Jacobian**: `J(i, :) = -df(x_i; theta)/dtheta`, shape `N x n`.
- **Gradient**: `grad L = J' * r`.

The analytical Jacobian is validated against centered finite differences
to `1e-6` relative error (see `test_jacobian.m`); in practice the agreement
is better than `1e-9`.

## Interpreting the output

`run_experiments` prints, for each (seed, method) pair:

```
train/val/test RMSE (MPa): X.XXX / X.XXX / X.XXX
```

and saves a file `results/seedN_METHOD.mat` containing:

- `theta`: final parameter vector (for Adam: the iterate with best val loss)
- `hist`: struct with per-iteration `loss`, `val_loss`, `grad_inf`, `time`,
  plus method-specific fields (`lambda` for LM, `alpha`/`beta` for NCG,
  `lr` for Adam)
- `meta`: train/val/test RMSE and seed/method tags
- `cfg`: the hyperparameter configuration used

`make_plots` reads these files and produces five PDFs into `report/figures/`:

| File                | Content                                                    |
| ------------------- | ---------------------------------------------------------- |
| `loss_vs_iter.pdf`  | training loss vs. iteration, log y, 3 curves + std band    |
| `grad_norm.pdf`     | `||grad L||_inf` vs. iteration, log y, 3 curves + std band |
| `loss_vs_time.pdf`  | training loss vs. wall-clock time, log-log                 |
| `lm_lambda.pdf`     | LM damping trajectory for each seed                        |
| `pred_vs_actual.pdf`| predicted vs. actual test-set strength, one panel per method |

`make_tables` prints the LaTeX rows for the four tables in the Results
section. The workflow is: copy each row into the corresponding `\begin{tabular}`
block in [report/main.tex](report/main.tex), replacing the `[X ± X]`
placeholders.

## Results from the current run (five seeds)

From the `run_experiments` output:

| Method | Train RMSE | Val RMSE  | Test RMSE | Train time (s) |
| ------ | ---------- | --------- | --------- | -------------- |
| LM     | 1.22 ± 0.24 | 9.34 ± 0.61 | **8.99 ± 1.25** | 2.96 ± 0.03 |
| NCG    | 2.28 ± 0.17 | 5.06 ± 0.52 | 5.65 ± 0.49 | 2.99 ± 0.10 |
| Adam   | 3.45 ± 0.30 | 4.88 ± 0.20 | **5.49 ± 0.52** | 1.53 ± 0.44 |

**LM achieves the lowest training loss by a wide margin but generalizes
worst** — classic overfitting. Its Gauss-Newton curvature lets it drive
the training loss down aggressively and the model memorizes noise in the
small (≈720-sample) training set. NCG and Adam reach comparable test RMSE
at ≈5.5 MPa, with Adam about 2× faster in wall clock due to its smaller
per-batch cost.

This is a *substantive* result for the central question of the report —
"does curvature information pay off?". The short answer: **yes for
fitting, no for generalization on small data**. This is where the
Discussion section should spend its ink.

## Known caveats

- The current Adam LR tuning uses a 200-epoch budget (1/10 of the main run)
  to stay cheap. If Adam's best LR is close to a grid boundary, consider
  widening the grid in `default_config.m` before the final run.
- LM on rank-deficient `J'J` is safeguarded by bumping `lambda` when
  `rcond(J'J + lambda*I) < 1e-14`. If you see the message
  `"opt_lm: lambda blew up"`, the linear system became persistently
  ill-conditioned — typically a sign of a bad initialization.
- Full-batch (LM, NCG) vs. mini-batch (Adam) is a genuine asymmetry
  acknowledged in the report (see `rem:batch-asymmetry` in
  [report/main.tex](report/main.tex)).

## Reproducibility

Every run is keyed by a seed. The same seed reproduces:

- the train/val/test split (`data_loader.m`)
- the initial parameter vector θ₀ (`nn('init', arch, seed)`)
- the Adam mini-batch order per epoch (`rng(1000 + epoch, 'twister')`)

Within a seed, all three optimizers start from the *same* θ₀, so
performance differences reflect optimizer behavior rather than initialization.
