# skepsis — Design Spec

**Date:** 2026-07-12
**Status:** Approved (design review with project owner)
**PyPI name:** `skepsis` (availability verified 2026-07-12)
**License:** Apache-2.0

## 1. Purpose

skepsis is a Python library that answers one question: **"is this backtest result real, or overfit?"**

The user feeds it the returns of a backtested strategy — and, ideally, the returns of all the variants they tried along the way — and skepsis produces statistical evidence for or against the result being luck, plus a self-contained HTML report suitable for handing to a portfolio manager.

**Positioning:** the missing successor to `mlfinlab`'s overfitting tools (now paywalled), built modern: clean API, real documentation, implementations verified against the source papers. skepsis is explicitly **not a backtester**. It sits downstream of vectorbt, backtesting.py, zipline, or a homegrown engine. That neutrality is a feature: any quant shop can adopt it regardless of stack.

**Audience:** quant researchers and quant developers at funds; serious independent quants. Secondary audience: hiring managers reading the repo.

## 2. Goals and non-goals

### Goals (v1)

1. Implement four overfitting diagnostics correctly, verified against published paper values.
2. One-call API (`skepsis.evaluate`) with progressive disclosure — more inputs unlock more diagnostics.
3. A polished, self-contained single-file HTML report.
4. Documentation site with a plain-English explainer per diagnostic.
5. Launchable v0.1 on PyPI within ~12 weeks at 5–10 hrs/week.

### Non-goals (v1)

- Running backtests, fetching market data, or portfolio optimization.
- Backtester-specific adapters (vectorbt/zipline importers) — post-v1.
- Walk-forward decay analysis — post-v1.
- CLI — post-v1.
- GUI/web app beyond the static HTML report.

## 3. v1 diagnostics

### 3.1 Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)

- **PSR** (Bailey & López de Prado, *The Sharpe Ratio Efficient Frontier*, 2012): probability that the true Sharpe exceeds a benchmark Sharpe, correcting the observed Sharpe for sample length, skewness, and kurtosis of returns.
- **DSR** (Bailey & López de Prado, *The Deflated Sharpe Ratio*, 2014): PSR evaluated against a benchmark Sharpe that reflects multiple testing — computed from the number of effectively independent trials and the variance of trial Sharpes.
- **Inputs:** chosen strategy's returns; optionally the trial returns matrix (to estimate trial count and Sharpe variance). If no trials are given, DSR runs with trial count = 1 and the report states plainly that this is almost certainly optimistic.
- **Outputs:** `PsrResult(value, benchmark_sr, ...)`, `DsrResult(value, p_value, n_trials_effective, ...)`.

### 3.2 Probability of Backtest Overfitting (PBO) via CSCV

- **Reference:** Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*, Journal of Computational Finance, 2015.
- **Method (CSCV):** split the T×N trial-returns matrix into S equal time blocks (S even, default 16, configurable). For every combination of S/2 blocks as in-sample: rank trials in-sample, take the in-sample winner, find its out-of-sample relative rank ω, compute logit λ = ln(ω/(1−ω)). **PBO = fraction of combinations with λ ≤ 0** (the in-sample winner performs in the bottom half out-of-sample).
- **Inputs:** trial returns matrix (T×N, N ≥ 2). Ranking metric defaults to Sharpe; pluggable.
- **Outputs:** `PboResult(value, logits, n_combinations, ...)` plus the logit distribution for plotting.
- **Performance guard:** C(16, 8) = 12,870 combinations; computation is vectorized numpy. S is capped at 24 (C(24, 12) ≈ 2.7M combinations) with a documented error above that; a warning suggests reducing S when N·T makes the run slow.

### 3.3 Block-bootstrap Monte Carlo

- **Method:** stationary bootstrap (Politis & Romano, 1994) over the strategy's returns, preserving autocorrelation structure; mean block length chosen automatically (Politis & White, 2004) with manual override. Produces bootstrap distributions of annualized Sharpe and max drawdown, plus a p-value against the no-skill null (mean-zero resampling).
- **Inputs:** strategy returns; `n_resamples` default 5,000 (configurable); seeded RNG for reproducibility.
- **Outputs:** `BootstrapResult(sharpe_ci, drawdown_ci, p_value_no_skill, distributions, ...)`.

### 3.4 Parameter sensitivity map

- **Method:** given trial metrics indexed by parameter values (1D or 2D grids in v1; higher dimensions rendered as 2D slices through the chosen configuration), compute a **neighborhood stability score**: the chosen configuration's metric relative to the median of its grid neighbors. A configuration on a plateau scores near 1.0; a configuration on an isolated spike scores well above its neighbors and is flagged as fitted-to-noise.
- **Inputs:** `params` DataFrame (one row per trial, one column per parameter) + per-trial metric (computed from the trials matrix, or supplied).
- **Outputs:** `SensitivityResult(stability_score, grid, ...)`; rendered as the heatmap in the report — the project's signature visual.
- **Irregular grids:** if trials do not form a regular grid, nearest-neighbor distances in normalized parameter space are used instead of grid adjacency; the report notes the method used.

## 4. API

```python
import skepsis

# Minimal: single strategy's returns
result = skepsis.evaluate(returns, freq="daily")

# Full: include every variant tried
result = skepsis.evaluate(
    returns,              # chosen strategy: 1-D array/Series of periodic returns
    trials=trials_df,     # T×N returns of all variants (columns = trials)
    params=params_df,     # optional: one row per trial, parameter values
    freq="daily",         # "daily" | "weekly" | "monthly" | "hourly" | int periods/year
)

result.psr                # PsrResult
result.deflated_sharpe    # DsrResult(value=0.31, p_value=0.38, ...)
result.pbo                # PboResult(value=0.47, ...)
result.bootstrap          # BootstrapResult
result.sensitivity        # SensitivityResult | None
result.verdict            # Verdict(level="WEAK", reasons=[...])
result.save_html("skepsis_report.html")
result.to_dict()          # JSON-serializable summary
```

- **Input coercion:** numpy arrays, pandas Series/DataFrame, and polars Series/DataFrame accepted; converted internally to numpy. All math is numpy/scipy. pandas and polars are optional dependencies.
- **Progressive disclosure:** with only `returns`, skepsis runs PSR, single-trial DSR, and bootstrap. With `trials`, it adds full DSR and PBO. With `params`, it adds sensitivity. The report lists any diagnostic that could not run and exactly why.
- **Verdict:** a rule-based aggregate — levels `STRONG / MODERATE / WEAK / LIKELY_OVERFIT` derived from documented thresholds (e.g., DSR p-value, PBO < 0.2 vs > 0.5, stability score). The verdict is explicitly labeled a heuristic; the report shows every rule that fired. Thresholds live in one module (`skepsis/verdict.py`) and are overridable.
- **Escape hatch:** every diagnostic is callable directly with explicit arguments, e.g. `skepsis.core.pbo.cscv(trials, n_blocks=16, metric="sharpe")`. `evaluate()` is convenience; `core/` is the product. A skeptical quant can bypass all orchestration.

## 5. Architecture

```
skepsis/
  __init__.py        # exports: evaluate, Result, core
  core/
    psr.py           # PSR + DSR (pure numpy/scipy; paper-cited)
    pbo.py           # CSCV / PBO
    bootstrap.py     # stationary bootstrap + block-length selection
    sensitivity.py   # parameter neighborhood stability
    moments.py       # shared: Sharpe, skew/kurtosis, annualization
  inputs.py          # coercion + validation → internal numpy arrays
  evaluate.py        # orchestration; builds Result
  verdict.py         # rule thresholds + Verdict
  report/
    template.html.j2 # jinja2 single-file template
    figures.py       # plotly figure builders
    render.py        # Result → self-contained HTML
docs/                # mkdocs-material site; one explainer page per diagnostic
tests/
  golden/            # values reproduced from the source papers
  unit/              # per-module tests
  properties/        # hypothesis invariants
```

- Each `core/` module carries the paper citation and formula references in its docstring.
- `core/` functions take plain numpy arrays and scalars — no DataFrames, no global state, deterministic given a seed.
- Report HTML embeds plotly figures with inlined JS so the file is fully self-contained and shareable over email/Slack.

## 6. Error handling

Fail loud and specific — a tool about statistical honesty must not silently paper over bad inputs.

- **Exceptions** (all subclass `SkepsisError`): `InsufficientDataError` (sample shorter than each diagnostic's documented minimum), `InvalidInputError` (NaNs/infs, non-numeric, empty), `MisalignedTrialsError` (returns length ≠ trials length; params rows ≠ trials columns), `InvalidFrequencyError`.
- **Warnings** (Python `warnings`, also surfaced in the report): strained assumptions are flagged, never silently adjusted — e.g., heavy autocorrelation where PSR assumes IID-ish returns, trial count too small for a stable DSR variance estimate, S reduced for tractability.
- NaN policy: reject, with a message telling the user to align/clean first. No silent dropping.

## 7. Testing

- **Golden tests:** reproduce worked examples and figures from the source papers to published precision (PSR/DSR examples from Bailey & López de Prado; CSCV example cases from the PBO paper). The README links these tests — "verified against the papers" is the credibility backbone.
- **Property-based tests** (hypothesis): e.g., increasing trial count never increases DSR; PBO of iid-noise trials converges to ≈ 0.5; bootstrap p-value under a true zero-mean process is uniform-ish; sensitivity score of a flat metric surface ≈ 1.0.
- **Cross-checks:** compare CSCV output against `pypbo` (surviving reference implementation) on shared inputs.
- **Report smoke test:** render HTML from a fixture Result; assert self-containment (no external network references).
- CI runs tests + ruff + type checks on 3.11/3.12/3.13.

## 8. Tooling

- Python ≥ 3.11. Managed with `uv`; lint/format with `ruff`; tests with `pytest` + `hypothesis`; types checked with `mypy` (strict on `core/`).
- Runtime deps kept minimal: `numpy`, `scipy`, `plotly`, `jinja2`. Optional extras: `pandas`, `polars`.
- GitHub Actions: CI (test matrix), docs deploy (mkdocs-material → GitHub Pages), release workflow (tag → PyPI via trusted publishing).

## 9. Launch & visibility plan

Getting seen by funds is half the project.

1. **README** leads with the report screenshot and a 5-line quickstart.
2. **Signature demo:** a notebook that takes a well-known "amazing backtest" from a popular trading blog and publicly deflates it — the shareable story for launch.
3. **Docs explainers:** one plain-English page per diagnostic (the math, when it lies, worked example). These earn search traffic and credibility.
4. **Launch v0.1** to r/quant, Hacker News, and quant fintwit/LinkedIn — only once the report looks great, not before.
5. Author's name on everything; the repo is the CV.

## 10. Milestones (5–10 hrs/week)

- **M1 (weeks 1–3):** repo scaffolding; `core/psr.py` (PSR + DSR) with golden tests; `inputs.py`; minimal `evaluate()` returning a text summary.
- **M2 (weeks 4–6):** `core/pbo.py` (CSCV) + `core/bootstrap.py`, cross-checked and property-tested.
- **M3 (weeks 7–9):** `core/sensitivity.py`; HTML report; docs site with four explainers.
- **M4 (weeks 10–12):** signature demo notebook; README polish; v0.1 to PyPI; launch posts.

## 11. Risks

- **Correctness risk:** a public library about statistical rigor with a math bug is a reputation liability. Mitigation: golden tests against papers, cross-checks, property tests, and conservative claims in docs.
- **Scope creep:** the adjacent-feature pull (adapters, walk-forward, data checks) is strong. Mitigation: non-goals list above; post-v1 roadmap absorbs the pressure.
- **Audience risk:** quants are a hostile-review crowd. Mitigation: cite everything, expose `core/` escape hatches, never oversell the verdict heuristic.
