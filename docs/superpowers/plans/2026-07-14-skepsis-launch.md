# skepsis Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take skepsis v1 public: real GitHub repo, vectorized bootstrap, paper-reproduced golden test, docs site, the "Deflating the Golden Cross" demo notebook, PyPI v0.1.0 release machinery, and launch post drafts.

**Architecture:** The library is done (v1 merged); this phase is release engineering plus content. Code changes are small and surgical (one vectorization, one golden test, one polish batch); the rest is repo/CI/docs/notebook/release artifacts around the existing code.

**Tech Stack:** existing (uv, numpy/scipy/plotly/jinja2, pytest/hypothesis, ruff, mypy) plus mkdocs-material + mkdocstrings (docs group), pandas + nbconvert + ipykernel + jupytext + playwright (demo group), GitHub Actions Pages/PyPI trusted publishing.

**Spec:** `docs/superpowers/specs/2026-07-14-skepsis-launch-design.md`. All §2 locked decisions apply. **One documented deviation from §3.5:** the notebook evaluates the sweep TWICE — raw returns (Act 1), then excess-over-buy-and-hold (Act 2) — because plan-time rehearsal on the real data showed raw returns survive DSR (BTC's drift means every variant carries real beta; verdict only MODERATE). The two-act structure delivers the spec's intended kill honestly: the fair null for a timing strategy is holding, and against it the family collapses to LIKELY_OVERFIT.

**Plan-time verification (2026-07-14, all numbers below are MEASURED, not estimated):**
- **DSR paper example reproduces exactly** with the current library: `expected_max_sharpe(1/500, 100)` = 0.113172 (paper prints ≈0.1132); DSR = 0.900397 (paper prints 0.9004); N=46 → 0.950502 (paper: 0.9505); N=88 with skew 0/kurt 3 → 0.950491 (paper: 0.9505). Source PDF: davidhbailey.com/dhbpapers/deflated-sharpe.pdf, "A NUMERICAL EXAMPLE", pp. 9–10.
- **Vectorized bootstrap verified**: statistically equivalent to the loop (continuation rate 0.9005 vs loop 0.9002 vs theory 0.9000 at L=10; identical marginal uniformity and resampled lag-1 autocorrelation 0.5101 vs 0.5093); benchmark `stationary_bootstrap_indices(4000, 1.0, 5000)` runs in **0.035s vs 29.05s** for the loop (830×), and 0.213s vs 3.05s at L=10.
- **Demo dress-rehearsed on real data** (4012 daily BTC-USD closes from Coinbase Exchange, 2015-07-20 → 2026-07-13): Act 1 (raw returns): best combo f5_s100, ann. Sharpe **1.464**, total return **650.8×**, all 34 trials Sharpe ≥ 1.050, buy-and-hold Sharpe **1.064**, PBO **0.448**, verdict **MODERATE**. Act 2 (excess over buy-and-hold): best excess Sharpe **0.084**, only **1 of 34** trials positive, DSR **0.345** (fail), PBO **0.405**, bootstrap p **0.3718**, stability **∞** (fail) → verdict **LIKELY_OVERFIT**. The two-act structure is the notebook's narrative: the Sharpe is real but it's beta; the timing skill is statistically indistinguishable from luck.
- **Dataset fingerprint** (for reproducibility assertions): 4012 rows, first date 2015-07-20 (close 280.0), last date 2026-07-13 (close 62264.94), sum of closes 124818690.85.

## Global Constraints

- Repo: `github.com/abhinay/skepsis`, public. Docs: `https://abhinay.github.io/skepsis`. `gh` CLI is authenticated as `abhinay`.
- All existing gates stay green after every task: `uv run ruff check .`, `uv run mypy skepsis` (strict on core), `uv run pytest -q` — pristine output, no warnings summary.
- Runtime deps stay exactly: numpy, scipy, plotly, jinja2 (pandas/polars optional extras). New tooling goes in dependency-groups (`docs`, `demo`), never runtime deps.
- Demo notebook must NOT need network (committed CSV); the fetch script needs network and is run only to regenerate data.
- Signal timing in the demo (spec §3.5): position on day t uses info through t−1 (`shift(1)`); drop the first 200 rows of every trial; 34 trials (fast ∈ {5,10,15,20,25,30,40,50} × slow ∈ {20,50,100,150,200}, fast < slow); annualization freq=365.
- Release publishes ONLY on published GitHub release, with tag == `v<pyproject version>` validated (spec §3.6).
- The bootstrap seed-stream change lands before v0.1.0 (spec §7) and is noted in the CHANGELOG.
- No individual is named or targeted in demo or posts (spec §5).
- Commit messages: conventional-commit style ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Public GitHub repo

**Files:**
- Modify: none (repo metadata + push only)

**Interfaces:**
- Consumes: local `main` at the current HEAD.
- Produces: `github.com/abhinay/skepsis` exists, public, with `main` pushed and the existing `ci.yml` observed green. All later tasks assume `origin` exists.

- [ ] **Step 1: Create the repo and push**

```bash
cd /Users/abhinay/Dev/skepsis
gh repo create abhinay/skepsis --public --source . --push \
  --description "Is your backtest result real, or overfit? Statistical diagnostics for backtest overfitting."
```
Expected: repo created, `main` pushed, origin remote configured.

- [ ] **Step 2: Set topics**

```bash
gh repo edit abhinay/skepsis --add-topic quant --add-topic backtesting \
  --add-topic overfitting --add-topic sharpe-ratio --add-topic finance --add-topic python
```

- [ ] **Step 3: Watch CI go green**

```bash
gh run watch --exit-status $(gh run list --workflow ci.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```
Expected: exit 0, all three matrix legs pass. If a leg fails, read the log (`gh run view --log-failed`), fix, push, re-watch — do not proceed red.

- [ ] **Step 4: No commit needed** (nothing changed locally). Report the repo URL and the green run URL.

---

### Task 2: Vectorized stationary bootstrap

**Files:**
- Modify: `skepsis/core/bootstrap.py` (replace `stationary_bootstrap_indices` body only)
- Test: `tests/unit/test_bootstrap.py` (add one performance regression test; existing tests unchanged)

**Interfaces:**
- Consumes: existing `stationary_bootstrap_indices(n_obs, mean_block_length, n_resamples, rng) -> np.ndarray` signature — UNCHANGED.
- Produces: same signature and contract ((n_resamples, n_obs) int64, values in [0, n_obs)); the seeded output stream CHANGES (documented in Task 8's CHANGELOG).

- [ ] **Step 1: Write the failing performance test**

Append to `tests/unit/test_bootstrap.py`:

```python
def test_index_generation_is_vectorized_fast() -> None:
    # Spec benchmark: (n_obs=4000, L=1.0, n_resamples=5000). Verified 0.035s
    # vectorized vs 29s looped locally; the 10s CI bound is deliberately loose
    # so loaded runners never flake. A regression to the Python loop fails this.
    import time

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    idx = stationary_bootstrap_indices(4000, 1.0, 5000, rng)
    elapsed = time.perf_counter() - t0
    assert idx.shape == (5000, 4000)
    assert elapsed < 10.0
```

- [ ] **Step 2: Run it to verify it fails (too slow with the loop)**

Run: `uv run pytest tests/unit/test_bootstrap.py::test_index_generation_is_vectorized_fast -q`
Expected: FAIL — `assert 29.0... < 10.0` (elapsed will be ~20–40s on the current loop).

- [ ] **Step 3: Replace the implementation**

In `skepsis/core/bootstrap.py`, replace the entire body of `stationary_bootstrap_indices` (keep the signature and docstring position) with:

```python
def stationary_bootstrap_indices(
    n_obs: int, mean_block_length: float, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_resamples, n_obs) index matrix: geometric block lengths, circular wrap.

    Vectorized formulation: each position independently starts a new block with
    probability p = 1/mean_block_length (this IS the stationary bootstrap of
    Politis & Romano — geometric block lengths emerge from the per-position
    Bernoulli trials); block starts are uniform on [0, n_obs); within a block,
    indices continue circularly from the block's start.
    """
    if mean_block_length < 1.0:
        raise InvalidInputError(f"mean_block_length must be >= 1, got {mean_block_length}")
    if mean_block_length == 1.0:
        # every position is its own block: plain iid resampling
        return rng.integers(0, n_obs, size=(n_resamples, n_obs), dtype=np.int64)
    p = 1.0 / mean_block_length
    new_block = rng.random((n_resamples, n_obs)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n_obs, size=(n_resamples, n_obs), dtype=np.int64)
    pos = np.arange(n_obs, dtype=np.int64)
    start_pos = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    starts_at_block = np.take_along_axis(starts, start_pos, axis=1)
    out: np.ndarray = (starts_at_block + (pos - start_pos)) % n_obs
    return out
```

- [ ] **Step 4: Run the full bootstrap test file**

Run: `uv run pytest tests/unit/test_bootstrap.py -q`
Expected: `11 passed` (10 existing + the new perf test), no warnings. The existing statistical property tests (block-length ordering, drift p-values, determinism, CI containment) are the correctness net — they were verified in plan-time scratch to pass against this exact implementation.

- [ ] **Step 5: Full gates and commit**

```bash
uv run ruff check . && uv run mypy skepsis && uv run pytest -q
git add -A
git commit -m "perf: vectorize stationary bootstrap index generation (~830x on iid path)

Seeded resample streams change with this commit; no released version
depended on them.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 3: Paper-reproduced golden test + README claim

**Files:**
- Create: `tests/golden/test_dsr_paper_example.py`
- Modify: `README.md` (the "verified against" sentence)

**Interfaces:**
- Consumes: `skepsis.core.psr.expected_max_sharpe`, `probabilistic_sharpe_ratio` (existing, unchanged).
- Produces: nothing new — a credibility artifact later tasks (docs, posts) link to.

- [ ] **Step 1: Write the golden test (values transcribed from the paper's printed example — do NOT adjust them to make anything pass)**

`tests/golden/test_dsr_paper_example.py`:

```python
"""Reproduces the numerical example of Bailey & Lopez de Prado (2014),
"The Deflated Sharpe Ratio", Journal of Portfolio Management 40(5),
section "A Numerical Example" (pp. 9-10 of the working-paper PDF at
davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

The paper's strategist discloses: N=100 independent trials, variance of the
trials' (non-annualized, daily) Sharpe ratios V = 1/(2*250) = 0.002, sample
length T=1250 (5 years at 250 obs/year), skewness -3, (non-excess) kurtosis
10, and a best-trial Sharpe of 2.5 annualized = 2.5/sqrt(250) daily.

The paper prints: SR0 ~= 0.1132 (non-annualized) and DSR = 0.9004 < 0.95,
plus two secondary results: DSR = 0.9505 had only N=46 trials been run, and
DSR = 0.9505 under Normal returns (skew 0, kurtosis 3) after N=88 trials.

skepsis reproduces all four printed values (verified 2026-07-14).
"""

import math

import pytest

from skepsis.core.psr import expected_max_sharpe, probabilistic_sharpe_ratio

V_TRIALS = 1.0 / (2.0 * 250.0)          # 0.002, daily (non-annualized)
SR_DAILY = 2.5 / math.sqrt(250.0)       # observed Sharpe, daily
T = 1250


def test_expected_max_sharpe_matches_paper_sr0() -> None:
    assert expected_max_sharpe(V_TRIALS, 100) == pytest.approx(0.1132, abs=5e-5)


def test_dsr_matches_paper() -> None:
    sr0 = expected_max_sharpe(V_TRIALS, 100)
    dsr = probabilistic_sharpe_ratio(SR_DAILY, sr0, T, -3.0, 10.0)
    assert dsr == pytest.approx(0.9004, abs=5e-5)
    assert dsr < 0.95  # the paper's conclusion: not significant at 95%


def test_paper_secondary_claims() -> None:
    dsr_46 = probabilistic_sharpe_ratio(
        SR_DAILY, expected_max_sharpe(V_TRIALS, 46), T, -3.0, 10.0
    )
    assert dsr_46 == pytest.approx(0.9505, abs=5e-5)
    dsr_88_normal = probabilistic_sharpe_ratio(
        SR_DAILY, expected_max_sharpe(V_TRIALS, 88), T, 0.0, 3.0
    )
    assert dsr_88_normal == pytest.approx(0.9505, abs=5e-5)
```

- [ ] **Step 2: Run to verify it passes** (this is a verification of existing code, not TDD of new code — the test must pass immediately; a failure means a transcription bug in the test)

Run: `uv run pytest tests/golden/test_dsr_paper_example.py -v`
Expected: `3 passed`.

- [ ] **Step 3: Update the README claim**

In `README.md`, replace the sentence:

```markdown
Every implementation is verified against reference values in
[`tests/golden/`](tests/golden/) — the numbers, not just the shapes.
```

with:

```markdown
The implementations are verified against reference values in
[`tests/golden/`](tests/golden/) — including reproducing the published
numerical example of the Deflated Sharpe Ratio paper to all four printed
decimals ([`test_dsr_paper_example.py`](tests/golden/test_dsr_paper_example.py)).
```

- [ ] **Step 4: Full gates and commit**

```bash
uv run ruff check . && uv run mypy skepsis && uv run pytest -q
git add -A
git commit -m "test: reproduce the DSR paper's published numerical example as a golden test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 4: Library & report polish batch

**Files:**
- Create: `skepsis/formatting.py`
- Create: `tests/unit/test_formatting.py`
- Modify: `skepsis/verdict.py` (stability reason strings)
- Modify: `skepsis/evaluate.py` (summary formatting, chosen-without-params warning, autocorrelation warning, MIN_OBS import)
- Modify: `skepsis/core/bootstrap.py` (rename `_MIN_OBS` → public `MIN_OBS`)
- Modify: `skepsis/core/moments.py` (docstring citation)
- Modify: `skepsis/report/render.py` + `skepsis/report/template.html.j2` (formatted stability/trials text, footer docs link)
- Test: additions to `tests/unit/test_evaluate.py`, `tests/unit/test_verdict.py`, `tests/unit/test_report.py`

**Interfaces:**
- Consumes: everything existing.
- Produces: `skepsis.formatting.format_stability(score: float) -> str` and `skepsis.formatting.count_trials(n: int) -> str` — used by verdict, evaluate, and report. `skepsis.core.bootstrap.MIN_OBS: int = 50` (public; `evaluate.py` imports it instead of duplicating). `evaluate.py` module constant `_AUTOCORR_WARN_BLOCK_LENGTH = 10.0`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_formatting.py`:
```python
import math

from skepsis.formatting import count_trials, format_stability


def test_format_stability_finite() -> None:
    assert format_stability(1.7) == "1.70"


def test_format_stability_inf_and_nan() -> None:
    assert format_stability(math.inf) == "∞ (isolated spike)"
    assert format_stability(math.nan) == "undefined (chosen metric non-positive)"


def test_count_trials_grammar() -> None:
    assert count_trials(1) == "1 trial"
    assert count_trials(34) == "34 trials"
```

Append to `tests/unit/test_verdict.py`:
```python
def test_infinite_stability_reason_is_formatted() -> None:
    v = decide(dsr=0.99, dsr_single_trial=False, pbo=0.10, bootstrap_p=0.01,
               stability_score=math.inf)
    assert any("∞ (isolated spike)" in r for r in v.reasons)
    assert not any("inf >" in r for r in v.reasons)
```

Append to `tests/unit/test_evaluate.py`:
```python
def test_chosen_without_params_warns() -> None:
    rng = np.random.default_rng(2)
    trials = rng.normal(0.001, 0.01, size=(120, 3))
    with pytest.warns(SkepsisWarning, match="ignored"):
        res = skepsis.evaluate(trials[:, 0].copy(), trials=trials, chosen=0,
                               n_resamples=100)
    assert any("ignored" in w for w in res.warnings)


def test_heavy_autocorrelation_warns() -> None:
    rng = np.random.default_rng(5)
    eps = rng.normal(0, 0.01, 600)
    r = np.empty(600)
    r[0] = 0.001
    for t in range(1, 600):
        r[t] = 0.001 + 0.85 * (r[t - 1] - 0.001) + eps[t]
    with pytest.warns(SkepsisWarning, match="autocorrelated"):
        res = skepsis.evaluate(r, n_resamples=100)
    assert any("autocorrelated" in w for w in res.warnings)


def test_summary_formats_inf_stability() -> None:
    r, trials, params = _sweep()
    with pytest.warns(SkepsisWarning):
        res = skepsis.evaluate(r, trials=trials, params=params, n_resamples=200)
    assert "∞ (isolated spike)" in res.summary()
    assert "34 trials" not in res.summary()  # this sweep has 9 trials
    assert "9 trials" in res.summary()
```

Append to `tests/unit/test_report.py` (inside the existing `test_full_report_renders_and_is_self_contained`, add two assertions at the end):
```python
    assert "∞ (isolated spike)" in html  # fixture's stability is inf
    assert "9 trials" in html
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/unit/test_formatting.py tests/unit/test_verdict.py tests/unit/test_evaluate.py tests/unit/test_report.py -q`
Expected: new tests FAIL (`ModuleNotFoundError: skepsis.formatting`, missing warnings, "inf" instead of "∞"). Existing tests still pass.

- [ ] **Step 3: Create skepsis/formatting.py**

```python
"""Shared human-facing formatting for diagnostic values.

One helper per convention so the report, summary(), and verdict reasons can
never drift apart (spec 2026-07-14 §3.2.5: all user-visible stability text)."""

import math


def format_stability(score: float) -> str:
    """Finite → 2dp; inf → labeled spike; nan → labeled undefined."""
    if math.isnan(score):
        return "undefined (chosen metric non-positive)"
    if math.isinf(score):
        return "∞ (isolated spike)"
    return f"{score:.2f}"


def count_trials(n: int) -> str:
    """Grammatical trial count: '1 trial', '34 trials'."""
    return "1 trial" if n == 1 else f"{n} trials"
```

- [ ] **Step 4: Wire the helpers and warnings**

a) `skepsis/verdict.py` — add `from skepsis.formatting import format_stability` and replace the two stability reason f-strings:
```python
        elif stability_score > t.sensitivity_fail:  # inf lands here
            fails.append(
                f"stability score {format_stability(stability_score)} > "
                f"{t.sensitivity_fail} (fail)"
            )
        elif stability_score > t.sensitivity_warn:
            warns.append(
                f"stability score {format_stability(stability_score)} > "
                f"{t.sensitivity_warn} (warn)"
            )
```

b) `skepsis/core/bootstrap.py` — rename `_MIN_OBS` to `MIN_OBS` (module-level, keep value 50; update its two internal uses).

c) `skepsis/evaluate.py`:
- Replace `_MIN_BOOTSTRAP_OBS = 50` with `from skepsis.core.bootstrap import MIN_OBS as _MIN_BOOTSTRAP_OBS` (added to the existing bootstrap import line) and add below the imports:
```python
_AUTOCORR_WARN_BLOCK_LENGTH = 10.0
"""Politis-White mean block length above which returns are considered heavily
autocorrelated; PSR/DSR assume IID-ish returns, so skepsis warns."""
```
- Add `from skepsis.formatting import count_trials, format_stability` to imports.
- In `summary()`, replace the trials fragment `over {self.deflated_sharpe.n_trials} trial(s))` with `over {count_trials(self.deflated_sharpe.n_trials)})` and the stability line with:
```python
        if self.sensitivity is not None:
            lines.append(
                f"  stability score: {format_stability(self.sensitivity.stability_score)}"
            )
```
- At the top of the `with _warnings.catch_warnings(...)` block (first statement inside it), add:
```python
        if chosen is not None and params_arr is None:
            _warnings.warn(
                "`chosen=` was provided without `params=`; it only affects the "
                "sensitivity diagnostic and is ignored",
                SkepsisWarning,
                stacklevel=2,
            )
```
- In the bootstrap branch, after `boot_res = bootstrap(...)`:
```python
            if boot_res.mean_block_length > _AUTOCORR_WARN_BLOCK_LENGTH:
                _warnings.warn(
                    f"estimated mean block length {boot_res.mean_block_length:.1f} "
                    f"exceeds {_AUTOCORR_WARN_BLOCK_LENGTH:.0f}: returns are heavily "
                    "autocorrelated, which strains the IID-ish assumptions behind "
                    "PSR/DSR — read those diagnostics with extra skepticism",
                    SkepsisWarning,
                    stacklevel=2,
                )
```

d) `skepsis/core/moments.py` — append to the module docstring:
```
Reference: Sharpe, W. F. (1994), "The Sharpe Ratio", Journal of Portfolio
Management 21(1), for the Sharpe ratio convention.
```

e) `skepsis/report/render.py` — add `from skepsis.formatting import count_trials, format_stability` and pass two extra template variables in the `render()` call:
```python
        stability_text=(
            format_stability(result.sensitivity.stability_score)
            if result.sensitivity is not None
            else None
        ),
        n_trials_text=count_trials(int(result.meta["n_trials"])),
```

f) `skepsis/report/template.html.j2`:
- Banner line: replace `{{ meta.n_trials }} trial(s)` with `{{ n_trials_text }}`.
- Summary row: replace `Deflated Sharpe Ratio ({{ dsr.n_trials }} trials)` with `Deflated Sharpe Ratio ({{ n_trials_text }})`.
- Stability row: replace `{{ "%.2f" | format(sensitivity.stability_score) }}` with `{{ stability_text }}`.
- Footer: after the skepsis repo link, add ` &middot; <a href="https://abhinay.github.io/skepsis">docs</a>`.

- [ ] **Step 5: Run all tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass (75+ tests incl. Tasks 2–3 additions), pristine output.

- [ ] **Step 6: Gates and commit**

```bash
uv run ruff check . && uv run mypy skepsis
git add -A
git commit -m "feat: shared stability/trial formatting, chosen-without-params and autocorrelation warnings, MIN_OBS de-dup, Sharpe citation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 5: BTC dataset + fetch script

**Files:**
- Create: `notebooks/data/fetch_btc.py`
- Create: `notebooks/data/README.md`
- Create: `notebooks/data/btc_usd_daily.csv` (generated by the script, then committed)
- Test: `tests/unit/test_demo_data.py`

**Interfaces:**
- Consumes: network (once, at generation time only).
- Produces: `notebooks/data/btc_usd_daily.csv` with columns `date,close`, exactly 4012 rows, 2015-07-20 → 2026-07-13 — the dataset Task 6's notebook and Task 9's post numbers depend on. The fingerprint test locks it against accidental modification.

- [ ] **Step 1: Write the fetch script**

`notebooks/data/fetch_btc.py`:
```python
"""Fetch daily BTC-USD closes from the Coinbase Exchange public candles API.

Provenance: https://api.exchange.coinbase.com/products/BTC-USD/candles
(granularity 86400, max 300 candles per request, paginated). The date range
is PINNED so the committed dataset — and every number in the demo notebook
and launch posts — is reproducible. Extend END only deliberately, together
with re-running the notebook and updating its narrative numbers.

Run: python notebooks/data/fetch_btc.py  (writes btc_usd_daily.csv next to itself)
"""

import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
START = dt.date(2015, 7, 20)  # first day Coinbase BTC-USD daily candles exist
END = dt.date(2026, 7, 13)    # PINNED: dataset fingerprint 4012 rows (see README)


def fetch_daily(start: dt.date, end: dt.date) -> pd.DataFrame:
    rows = []
    cur = start
    while cur < end:
        chunk_end = min(cur + dt.timedelta(days=299), end)
        url = (
            f"{BASE}?granularity=86400&start={cur.isoformat()}T00:00:00Z"
            f"&end={chunk_end.isoformat()}T00:00:00Z"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "skepsis-demo/0.1"})
        with urllib.request.urlopen(req) as resp:
            rows.extend(json.loads(resp.read()))  # [time, low, high, open, close, volume]
        cur = chunk_end + dt.timedelta(days=1)
        time.sleep(0.15)  # public rate-limit courtesy
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
    return (
        df.drop_duplicates("date").sort_values("date").reset_index(drop=True)[["date", "close"]]
    )


if __name__ == "__main__":
    out = Path(__file__).parent / "btc_usd_daily.csv"
    df = fetch_daily(START, END)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df['date'].iloc[0]} .. {df['date'].iloc[-1]}")
```

- [ ] **Step 2: Generate the dataset**

```bash
uv run python notebooks/data/fetch_btc.py
```
Expected output: `wrote .../btc_usd_daily.csv: 4012 rows, 2015-07-20 .. 2026-07-13`.
(A plan-time run of this exact logic produced this exact file; if the API returns a different row count, STOP and report BLOCKED — do not adjust the fingerprint test.)

- [ ] **Step 3: Write the fingerprint test**

`tests/unit/test_demo_data.py`:
```python
"""Locks the committed demo dataset: the notebook's narrative numbers and the
launch posts quote results computed from exactly this data."""

from pathlib import Path

import pandas as pd
import pytest

CSV = Path(__file__).parents[2] / "notebooks" / "data" / "btc_usd_daily.csv"


def test_dataset_fingerprint() -> None:
    df = pd.read_csv(CSV)
    assert list(df.columns) == ["date", "close"]
    assert len(df) == 4012
    assert df["date"].iloc[0] == "2015-07-20" and df["close"].iloc[0] == 280.0
    assert df["date"].iloc[-1] == "2026-07-13" and df["close"].iloc[-1] == 62264.94
    assert df["close"].sum() == pytest.approx(124818690.85, abs=0.01)
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/unit/test_demo_data.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Write the data README**

`notebooks/data/README.md`:
```markdown
# BTC-USD daily closes

`btc_usd_daily.csv` — 4012 daily closes, 2015-07-20 → 2026-07-13 (UTC),
fetched 2026-07-14 from the Coinbase Exchange public market-data API
(`api.exchange.coinbase.com/products/BTC-USD/candles`, granularity 86400) by
`fetch_btc.py`. Prices are factual market data; the fetch script, source,
and pinned date range are committed so the file is reproducible bit-for-bit.

The demo notebook reads this file and needs no network. The date range is
pinned because the notebook's narrative and the launch posts quote numbers
computed from exactly this dataset (fingerprint enforced by
`tests/unit/test_demo_data.py`). To extend the range: edit `END` in
`fetch_btc.py`, re-run it, re-execute the notebook, and update the quoted
numbers and the fingerprint test together.
```

- [ ] **Step 6: Gates and commit**

```bash
uv run ruff check . && uv run mypy skepsis && uv run pytest -q
git add notebooks/ tests/unit/test_demo_data.py
git commit -m "feat: committed BTC-USD demo dataset with pinned, fingerprinted provenance

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 6: Demo notebook — "Deflating the Golden Cross"

**Files:**
- Create: `notebooks/deflating-the-golden-cross.py` (jupytext percent source of truth)
- Create: `notebooks/deflating-the-golden-cross.ipynb` (generated + executed, committed WITH outputs so GitHub renders the story)
- Modify: `pyproject.toml` (add `demo` dependency group)
- Modify: `.github/workflows/ci.yml` (add notebook-execution job)
- Modify: `.gitignore` (allow the committed .ipynb despite `*.html` patterns being unrelated — verify nothing blocks it; add `notebooks/*_report.html`)

**Interfaces:**
- Consumes: `notebooks/data/btc_usd_daily.csv` (Task 5, fingerprinted), `skepsis.evaluate` with the Task 4 polish.
- Produces: the executed notebook whose printed numbers Task 9's posts quote, and `notebooks/golden_cross_report.html` (generated at run time, gitignored) that Task 10 screenshots.

- [ ] **Step 1: Add the demo dependency group**

In `pyproject.toml` under `[dependency-groups]`:
```toml
demo = [
    "pandas>=2.0",
    "nbconvert>=7.16",
    "ipykernel>=6.29",
    "jupytext>=1.16",
    "playwright>=1.45",
]
```
Run `uv sync --group demo` and confirm it resolves.

- [ ] **Step 2: Write the jupytext source**

`notebooks/deflating-the-golden-cross.py` — complete content (percent format; the `assert`s make CI fail if the numbers ever drift from the committed narrative):

```python
# %% [markdown]
# # Deflating the Golden Cross
#
# The moving-average crossover — buy when the fast MA crosses above the slow
# MA — is the most-published trading strategy on the internet. Nobody's blog
# post is targeted here: we build the whole family ourselves, honestly, with
# no look-ahead and no costs, on 11 years of daily BTC-USD (4,012 closes,
# Coinbase Exchange, committed in `data/` with provenance).
#
# Then we do the thing almost no blog post does: we account for **how many
# variants we tried**, using [skepsis](https://github.com/abhinay/skepsis).

# %%
import math

import numpy as np
import pandas as pd

import skepsis

df = pd.read_csv("data/btc_usd_daily.csv")
close = df["close"].to_numpy(dtype=float)
ret = np.diff(close) / close[:-1]
px = pd.Series(close)
print(f"{len(df)} daily closes, {df.date.iloc[0]} .. {df.date.iloc[-1]}")

# %% [markdown]
# ## The sweep
#
# 34 fast/slow combinations. **No look-ahead:** the position held on day *t*
# comes from MAs computed through day *t−1* (`shift(1)`), and every trial
# drops the same 200-day warm-up so all series align.

# %%
FAST = [5, 10, 15, 20, 25, 30, 40, 50]
SLOW = [20, 50, 100, 150, 200]
WARMUP = 200

trials, params = {}, []
for f in FAST:
    for s in SLOW:
        if f >= s:
            continue
        signal = (px.rolling(f).mean() > px.rolling(s).mean()).astype(float).shift(1)
        trials[f"f{f}_s{s}"] = (signal.to_numpy()[1:] * ret)[WARMUP:]
        params.append((f, s))
trials_df = pd.DataFrame(trials)
params_df = pd.DataFrame(params, columns=["fast", "slow"])
assert trials_df.shape[1] == 34 and not trials_df.isna().any().any()

ANN = math.sqrt(365.0)  # crypto trades every day
sharpes = trials_df.mean() / trials_df.std(ddof=1) * ANN
best = sharpes.idxmax()
bh_sharpe = float(np.mean(ret[WARMUP:]) / np.std(ret[WARMUP:], ddof=1) * ANN)
print(f"best combo: {best}  in-sample annualized Sharpe {sharpes[best]:.3f}")
print(f"total return of best combo: {float(np.prod(1 + trials_df[best])):.0f}x")
print(f"buy-and-hold Sharpe over the same period: {bh_sharpe:.3f}")
assert best == "f5_s100"
assert abs(sharpes[best] - 1.464) < 0.01 and abs(bh_sharpe - 1.064) < 0.01

# %% [markdown]
# ## Act 1 — the pitch
#
# Sharpe **1.46**. A **650×** total return. This is the chart that sells the
# course. Let's hand the *entire sweep* — winner plus the 33 variants nobody
# publishes — to skepsis.

# %%
result_raw = skepsis.evaluate(
    trials_df[best].to_numpy(), trials=trials_df, params=params_df,
    freq=365, n_resamples=2000, seed=0,
)
print(result_raw.summary())
assert result_raw.verdict.level == "MODERATE"
assert abs(result_raw.pbo.value - 0.448) < 0.02

# %% [markdown]
# ## Reading Act 1 honestly
#
# The Deflated Sharpe Ratio does **not** kill this backtest — because the
# returns are real. But look closer: *every one of the 34 variants* has
# Sharpe ≥ 1.05, and buy-and-hold alone scores 1.06. The strategy family
# isn't finding timing skill; it's holding BTC through a historic bull run.
# The Sharpe is **beta wearing a costume**. And PBO ≈ 0.45 is the tell: the
# in-sample winner lands in the *bottom half* out-of-sample on ~45% of
# CSCV splits — the parameter choice is close to a coin flip.
#
# ## Act 2 — the fair null
#
# A *timing* strategy earns its fee only if it beats what you'd get by doing
# nothing: holding. So evaluate the **excess return over buy-and-hold**.

# %%
excess_df = trials_df.sub(pd.Series(ret[WARMUP:], index=trials_df.index), axis=0)
ex_sharpes = excess_df.mean() / excess_df.std(ddof=1) * ANN
print(f"trials with positive excess Sharpe: {(ex_sharpes > 0).sum()} of 34")
print(f"best excess Sharpe: {ex_sharpes.max():.3f}")
assert int((ex_sharpes > 0).sum()) == 1
result_excess = skepsis.evaluate(
    excess_df[ex_sharpes.idxmax()].to_numpy(), trials=excess_df, params=params_df,
    freq=365, n_resamples=2000, seed=0,
)
print(result_excess.summary())
assert result_excess.verdict.level == "LIKELY_OVERFIT"
assert abs(result_excess.deflated_sharpe.value - 0.345) < 0.02

# %% [markdown]
# ## The verdict
#
# Against the null that matters, the Golden Cross family collapses:
# **33 of 34 variants would have made you poorer than doing nothing**, the
# best survivor's excess Sharpe (0.08) deflates to a DSR of **0.35** — worse
# than a coin flip — and the sensitivity map shows the "best" parameters are
# an isolated spike, not a plateau. Verdict: **LIKELY_OVERFIT**.
#
# The one-line moral: *a backtest without its trials is an advertisement,*
# *not evidence.* skepsis exists to ask for the trials.

# %%
result_excess.save_html("golden_cross_report.html")
print("wrote golden_cross_report.html")
```

- [ ] **Step 3: Generate and execute the notebook**

```bash
cd notebooks
uv run --group demo jupytext --to ipynb deflating-the-golden-cross.py
uv run --group demo jupyter nbconvert --to notebook --execute --inplace deflating-the-golden-cross.ipynb
cd ..
```
Expected: executes cleanly (~1–2 min; the asserts pass). Add `notebooks/golden_cross_report.html` to `.gitignore`.

- [ ] **Step 4: Add the CI job**

Append to `.github/workflows/ci.yml` under `jobs:`:
```yaml
  notebook:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync --group demo
      - run: >
          uv run --group demo jupyter nbconvert --to notebook --execute
          --stdout notebooks/deflating-the-golden-cross.ipynb > /dev/null
```

- [ ] **Step 5: Gates, commit, verify CI**

```bash
uv run ruff check . && uv run mypy skepsis && uv run pytest -q
git add -A
git commit -m "feat: Deflating the Golden Cross demo notebook with CI execution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
gh run watch --exit-status $(gh run list --workflow ci.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```
Expected: all jobs green including `notebook`.

---

### Task 7: Docs site

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`, `docs/explainers/deflated-sharpe.md`, `docs/explainers/pbo.md`, `docs/explainers/bootstrap.md`, `docs/explainers/sensitivity.md`, `docs/verdict.md`, `docs/api.md`
- Create: `.github/workflows/docs.yml`
- Modify: `pyproject.toml` (docs group), `.gitignore` (`site/`), `.github/workflows/ci.yml` (docs build gate)

**Interfaces:**
- Consumes: the library docstrings (mkdocstrings), the demo numbers (plan header).
- Produces: `https://abhinay.github.io/skepsis` live; `uv run --group docs mkdocs build --strict` as a CI gate.

- [ ] **Step 1: Add docs group and config**

`pyproject.toml` `[dependency-groups]`: `docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.25"]`. Add `site/` to `.gitignore`.

`mkdocs.yml`:
```yaml
site_name: skepsis
site_description: Is your backtest result real, or overfit?
site_url: https://abhinay.github.io/skepsis
repo_url: https://github.com/abhinay/skepsis
repo_name: abhinay/skepsis
theme:
  name: material
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      toggle: { icon: material/brightness-7, name: dark mode }
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      toggle: { icon: material/brightness-4, name: light mode }
  features: [navigation.sections, content.code.copy]
nav:
  - Home: index.md
  - Explainers:
      - Deflated Sharpe Ratio: explainers/deflated-sharpe.md
      - Probability of Backtest Overfitting: explainers/pbo.md
      - Block Bootstrap: explainers/bootstrap.md
      - Parameter Sensitivity: explainers/sensitivity.md
  - The Verdict: verdict.md
  - API: api.md
exclude_docs: |
  superpowers/
  launch/
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options: { show_source: true, docstring_section_style: list }
markdown_extensions:
  - admonition
  - pymdownx.superfences
  - pymdownx.arithmatex: { generic: true }
extra_javascript:
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js
```

- [ ] **Step 2: Write the pages**

`docs/index.md` — complete content (note the outer fence is 4 backticks because the page embeds fenced blocks):
````markdown
# skepsis

> Is your backtest result real, or overfit?

skepsis takes the returns of a backtested strategy — and, ideally, the
returns of **every variant you tried along the way** — and produces
statistical evidence for or against the result being luck, plus a
self-contained HTML report you can hand to a PM.

![skepsis report](assets/report.png)

```bash
pip install skepsis
```

```python
import skepsis

result = skepsis.evaluate(returns, trials=trials_df, params=params_df, freq="daily")
print(result.summary())
result.save_html("skepsis_report.html")
```

Start with the demo: [Deflating the Golden Cross](https://github.com/abhinay/skepsis/blob/main/notebooks/deflating-the-golden-cross.ipynb)
— a Sharpe-1.46, 650× backtest on 11 years of BTC, and what's left of it
once you account for the 34 variants tried (spoiler: a DSR of 0.35).

skepsis is **not a backtester**. It sits downstream of whatever you use to
generate returns, and it never silently repairs bad input: NaNs are
rejected, strained assumptions are warned about, skipped diagnostics say
why. Every implementation is verified against published paper values —
including the DSR paper's own numerical example, reproduced to all four
printed decimals.
````

The four explainers + `verdict.md`: each page MUST contain, in this order — (1) *The question it answers*, one paragraph; (2) *The math, in plain English* with the exact formulas rendered in MathJax (PSR/DSR: the PSR formula with skew/kurtosis denominator and the E[max] expression with the Euler–Mascheroni mixture; PBO: the CSCV split/rank/logit procedure and PBO = fraction of λ ≤ 0; bootstrap: stationary bootstrap resampling + Politis-White selection + the no-skill p-value formula; sensitivity: k-NN in z-scored space, score = chosen/median(neighbors)); (3) *When it lies* — the honest caveats (PSR/DSR: raw N vs the paper's effective-trials clustering, per spec §3.2.8, plus autocorrelation strain; PBO: needs the real trials, garbage-in; bootstrap: circular wrap and stationarity assumptions; sensitivity: grid resolution, k-NN in scaled space); (4) *Worked example* using the REAL demo numbers from the plan header (Act 1/Act 2 of the Golden Cross); (5) *References* with full citations (Bailey & López de Prado 2012/2014; Bailey, Borwein, López de Prado & Zhu 2015; Politis & Romano 1994; Politis & White 2004; Sharpe 1994). `verdict.md` additionally reproduces the exact `Thresholds` defaults table (dsr 0.5/0.95, pbo 0.5/0.2, bootstrap 0.5/0.05, sensitivity 2.0/1.5), the aggregation rule (any fail → LIKELY_OVERFIT; ≥2 warns → WEAK; 1 → MODERATE; 0 → STRONG), the single-trial rule, and the override example `skepsis.evaluate(..., thresholds=Thresholds(pbo_warn=0.3))`.

`docs/api.md` — complete content:
```markdown
# API reference

::: skepsis.evaluate
::: skepsis.Result
::: skepsis.Thresholds
::: skepsis.core.psr
::: skepsis.core.pbo
::: skepsis.core.bootstrap
::: skepsis.core.sensitivity
```

- [ ] **Step 3: Build strictly, iterate until clean**

Run: `uv run --group docs mkdocs build --strict`
Expected: builds with zero warnings (strict turns warnings into errors — broken links, missing pages, bad mkdocstrings identifiers all fail here). `docs/assets/report.png` won't exist until Task 10 — create `docs/assets/` with a 1×1 placeholder PNG now (`uv run python -c "import base64,pathlib; pathlib.Path('docs/assets').mkdir(parents=True, exist_ok=True); pathlib.Path('docs/assets/report.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='))"`), replaced by the real screenshot in Task 10.

- [ ] **Step 4: Docs workflow + CI gate**

`.github/workflows/docs.yml`:
```yaml
name: docs
on:
  push:
    branches: [main]
permissions:
  contents: read
  pages: write
  id-token: write
concurrency: { group: pages, cancel-in-progress: true }
jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --group docs
      - run: uv run --group docs mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        with: { path: site }
      - id: deployment
        uses: actions/deploy-pages@v4
```

Append a docs gate job to `.github/workflows/ci.yml`:
```yaml
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --group docs
      - run: uv run --group docs mkdocs build --strict
```

Enable Pages for workflow deployment:
```bash
gh api -X POST repos/abhinay/skepsis/pages -f build_type=workflow 2>/dev/null \
  || gh api -X PUT repos/abhinay/skepsis/pages -f build_type=workflow
```

- [ ] **Step 5: Gates, commit, verify deployment**

```bash
uv run ruff check . && uv run mypy skepsis && uv run pytest -q
git add -A
git commit -m "docs: mkdocs-material site with per-diagnostic explainers and Pages deploy

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
gh run watch --exit-status $(gh run list --workflow docs.yml --limit 1 --json databaseId --jq '.[0].databaseId')
curl -fsS -o /dev/null -w "%{http_code}\n" https://abhinay.github.io/skepsis/
```
Expected: docs workflow green; final curl prints `200` (Pages can take a minute — retry once).

---

### Task 8: Release engineering (v0.1.0)

**Files:**
- Modify: `pyproject.toml` (version), `skepsis/__init__.py` (version), `README.md` (badges, install)
- Create: `CHANGELOG.md`
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: everything (this is the packaging of it).
- Produces: release machinery armed; the actual release happens in Task 10 after the owner's one manual PyPI step.

- [ ] **Step 1: Bump version**

`pyproject.toml`: `version = "0.1.0"`. `skepsis/__init__.py`: `__version__ = "0.1.0"`.
Run: `uv run pytest tests/unit/test_version.py -q` → `1 passed` (asserts the `0.1.0` prefix).

- [ ] **Step 2: Write CHANGELOG.md** (use today's date from `date +%Y-%m-%d`)

```markdown
# Changelog

## 0.1.0 — <date of release commit>

First public release.

- Four overfitting diagnostics: Probabilistic & Deflated Sharpe Ratio
  (Bailey & López de Prado 2012, 2014 — reproduces the DSR paper's published
  numerical example to all four printed decimals), Probability of Backtest
  Overfitting via CSCV (Bailey, Borwein, López de Prado & Zhu 2015),
  stationary block bootstrap with Politis–White block length (1994, 2004),
  and k-NN parameter-sensitivity maps.
- `skepsis.evaluate()` with progressive disclosure, rule-based overridable
  verdict, strict-JSON `to_dict()`, and a fully self-contained single-file
  HTML report.
- Vectorized bootstrap index generation (~830× on the iid path). Note:
  seeded resample streams differ from pre-release development builds; no
  released version depended on them.
- Demo: "Deflating the Golden Cross" notebook on 11 years of BTC-USD.
```

- [ ] **Step 3: Write .github/workflows/release.yml**

```yaml
name: release
on:
  release:
    types: [published]
permissions:
  contents: read
  id-token: write
jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4
      - name: Validate tag matches package version
        run: |
          VERSION=$(grep -m1 '^version = ' pyproject.toml | cut -d'"' -f2)
          TAG="${{ github.event.release.tag_name }}"
          if [ "$TAG" != "v$VERSION" ]; then
            echo "release tag $TAG does not match pyproject version v$VERSION" >&2
            exit 1
          fi
      - uses: astral-sh/setup-uv@v5
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 4: README badges + install**

At the top of `README.md`, directly under the tagline blockquote, add:
```markdown
[![ci](https://github.com/abhinay/skepsis/actions/workflows/ci.yml/badge.svg)](https://github.com/abhinay/skepsis/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/skepsis)](https://pypi.org/project/skepsis/)
[![docs](https://img.shields.io/badge/docs-abhinay.github.io%2Fskepsis-blue)](https://abhinay.github.io/skepsis)

![skepsis report](docs/assets/report.png)
```
Replace the "Status: pre-release… install from git" block with a plain PyPI install fence:
```bash
pip install skepsis
```
Add under the quickstart: `Docs: <https://abhinay.github.io/skepsis> · Demo: [Deflating the Golden Cross](notebooks/deflating-the-golden-cross.ipynb)`.

- [ ] **Step 5: Gates, commit, push**

```bash
uv run ruff check . && uv run mypy skepsis && uv run pytest -q
git add -A
git commit -m "chore: v0.1.0 version, changelog, release workflow, README badges

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 9: Launch post drafts

**Files:**
- Create: `docs/launch/reddit-r-quant.md`, `docs/launch/hn-show-hn.md`, `docs/launch/linkedin.md`

**Interfaces:**
- Consumes: the verified demo numbers (plan header). All three posts quote ONLY numbers printed by the committed executed notebook.
- Produces: drafts the owner posts manually AFTER the PyPI release is live (Task 10).

- [ ] **Step 1: Write the three drafts** (complete text; owner may edit voice before posting)

`docs/launch/hn-show-hn.md`:
```markdown
Title: Show HN: Skepsis – statistical tests for whether your backtest is overfit

I built an open-source Python library that answers one question: is this
backtest result real, or luck?

You give it your strategy's returns — and, crucially, the returns of every
variant you tried — and it runs the diagnostics from the backtest-overfitting
literature: Deflated Sharpe Ratio, Probability of Backtest Overfitting
(CSCV), a stationary block bootstrap, and a parameter-sensitivity map. It
produces a single-file HTML report with a verdict.

The demo notebook builds the most-published strategy on the internet — the
moving-average crossover, on 11 years of daily BTC — sweeps all 34
parameter combinations honestly (no look-ahead, no costs), and finds a
Sharpe 1.46, 650x backtest. Then it deflates it: measured against simply
holding, 33 of 34 variants lose, and the best survivor's Deflated Sharpe
Ratio is 0.35 — worse than a coin flip. The Sharpe was beta wearing a
costume.

The implementations are verified against the published papers, including
reproducing the DSR paper's numerical example to all four printed decimals.
Apache-2.0. Not a backtester — it sits downstream of whatever you use.

Repo: https://github.com/abhinay/skepsis
Docs: https://abhinay.github.io/skepsis
Demo: https://github.com/abhinay/skepsis/blob/main/notebooks/deflating-the-golden-cross.ipynb
```

`docs/launch/reddit-r-quant.md`:
```markdown
Title: skepsis: open-source library that tells you whether your backtest is overfit (DSR, PBO/CSCV, block bootstrap, sensitivity maps)

Most backtests published anywhere — blogs, vendor decks, even papers — omit
the one number that determines whether the result means anything: how many
variants were tried. I built skepsis to make accounting for that trivial.

`skepsis.evaluate(returns, trials=..., params=...)` runs: Probabilistic &
Deflated Sharpe Ratio (Bailey & López de Prado), Probability of Backtest
Overfitting via CSCV (Bailey, Borwein, López de Prado & Zhu), a stationary
block bootstrap with Politis-White block length, and a k-NN parameter
sensitivity map. Output: a rule-based verdict (thresholds documented and
overridable) and a self-contained HTML report.

Receipts, because this sub is rightly hostile to claims:
- Golden tests reproduce the DSR paper's numerical example to all four
  printed decimals (SR0 0.1132, DSR 0.9004, and both secondary claims).
- The demo sweeps 34 MA-crossover combos on 11y of daily BTC with shift(1)
  timing and no costs: best combo Sharpe 1.46 / 650x. Against buy-and-hold,
  33/34 variants lose and the winner's DSR is 0.35. PBO says the parameter
  choice flips OOS rank on ~45% of CSCV splits.
- Everything fails loud: NaNs rejected, skipped diagnostics say why,
  strained assumptions warn (e.g., heavy autocorrelation vs PSR).

Apache-2.0, Python ≥3.11, numpy/scipy core. Deliberately NOT a backtester.
Would genuinely value this sub's abuse: what's wrong, what's missing, what
would make you trust it or bin it?

https://github.com/abhinay/skepsis · https://abhinay.github.io/skepsis
```

`docs/launch/linkedin.md`:
```markdown
I shipped an open-source project: skepsis — statistical diagnostics for
backtest overfitting.

The uncomfortable truth about backtests is that the headline Sharpe ratio
is meaningless without one more number: how many variants were tried before
picking the winner. skepsis makes that accounting one function call — the
Deflated Sharpe Ratio, Probability of Backtest Overfitting, block-bootstrap
confidence intervals, and a parameter-sensitivity map, with a shareable
one-file HTML report.

My favorite result from building it: the demo takes the most-published
strategy on the internet (the moving-average "golden cross"), runs it
honestly on 11 years of Bitcoin data, and gets a seductive Sharpe 1.46 /
650x backtest. Then it asks the fair question — does the timing beat just
holding? 33 of 34 parameter combinations don't, and the survivor's Deflated
Sharpe is 0.35. The edge was beta in a costume.

Implementations are verified against the published papers (the DSR paper's
example reproduces to all four printed decimals). Apache-2.0.

Repo: https://github.com/abhinay/skepsis
Docs: https://abhinay.github.io/skepsis

If you work on research infrastructure and this is the kind of tooling you
care about — I'd love to compare notes.
```

- [ ] **Step 2: Commit**

```bash
git add docs/launch/
git commit -m "docs: launch post drafts (HN, r/quant, LinkedIn)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 10: Screenshot, final verification, release runbook

**Files:**
- Create: `scripts/screenshot_report.py`
- Modify: `docs/assets/report.png` (replace Task 7's placeholder with the real screenshot)

**Interfaces:**
- Consumes: `notebooks/golden_cross_report.html` (from executing the notebook), playwright (demo group).
- Produces: the real README/docs screenshot; the go/no-go verification; the owner-gated release steps.

- [ ] **Step 1: Write scripts/screenshot_report.py**

```python
"""Capture docs/assets/report.png from the demo report (run the notebook first).

Regenerate: uv run --group demo playwright install chromium
            uv run --group demo python scripts/screenshot_report.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

html = Path("notebooks/golden_cross_report.html").resolve()
out = Path("docs/assets/report.png")
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 1400}, device_scale_factor=2)
    page.goto(html.as_uri())
    page.wait_for_timeout(2000)  # let plotly render
    page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": 1000, "height": 1400})
    browser.close()
print(f"wrote {out}")
```

- [ ] **Step 2: Generate the screenshot**

```bash
uv run --group demo jupyter nbconvert --to notebook --execute --stdout notebooks/deflating-the-golden-cross.ipynb > /dev/null
uv run --group demo playwright install chromium
uv run --group demo python scripts/screenshot_report.py
```
Expected: `docs/assets/report.png` is a real report image (verdict banner visible). If chromium download fails in this environment, report DONE_WITH_CONCERNS naming the exact failed command — the owner can run the two commands manually.

- [ ] **Step 3: Final verification sweep**

```bash
uv run ruff check . && uv run mypy skepsis && uv run pytest -q
uv run --group docs mkdocs build --strict
git add -A
git commit -m "docs: real report screenshot for README and docs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
gh run watch --exit-status $(gh run list --workflow ci.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```
Expected: everything green, screenshot renders on the GitHub README.

- [ ] **Step 4: OWNER-GATED — do not execute without explicit owner confirmation**

The release requires one step only the owner can do, then two commands:

1. **Owner:** on pypi.org (logged in as the account that will own the package):
   Account → Publishing → "Add a new pending publisher" → PyPI project name
   `skepsis`, owner `abhinay`, repository `skepsis`, workflow `release.yml`,
   environment `pypi`.
2. Then: `gh release create v0.1.0 --title "skepsis v0.1.0" --notes-file CHANGELOG.md`
3. Watch: `gh run watch --exit-status $(gh run list --workflow release.yml --limit 1 --json databaseId --jq '.[0].databaseId')`
4. Smoke test (`--no-project` so the local checkout cannot mask PyPI): `uv run --no-project --isolated --with skepsis python -c "import skepsis; print(skepsis.__version__)"` → `0.1.0`.
5. Only after PyPI install works: owner posts the three drafts from `docs/launch/`.

---

## Execution order and dependencies

Task 1 (repo) first. Tasks 2, 3, 4 are independent of each other (all touch the library; execute sequentially to avoid merge friction, any order). Task 5 (data) before Task 6 (notebook). Task 7 (docs) needs Task 4's formatting only for accuracy of examples — run after 4. Task 8 after 2–7 (version bump last). Task 9 after 6 (quotes its numbers). Task 10 last, with the owner gate before the release itself.

## Post-plan follow-ups (NOT in this plan)

- Post-launch roadmap: walk-forward decay, backtester adapters (vectorbt et al.), CLI, effective-trials (clustered) DSR option.
- Community plumbing when traffic arrives: issue templates, CONTRIBUTING.md, discussion channels.
