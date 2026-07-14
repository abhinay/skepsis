# skepsis launch — Design Spec

**Date:** 2026-07-14
**Status:** Approved (design review with project owner)
**Prerequisite:** v1 library merged to `main` (spec `2026-07-12-skepsis-design.md`, plan `2026-07-13-skepsis-v1.md`, all reviews clean)

## 1. Purpose

Take the finished skepsis v1 library public: real repo, real docs, a signature demo, a PyPI release, and launch posts. This phase implements spec §9 (launch & visibility) and milestones M3.5–M4 of the original design, plus the deferred backlog from the v1 final review. Visibility is half the project: the demo notebook and docs explainers are first-class deliverables, not extras.

## 2. Locked decisions

- **GitHub home:** `github.com/abhinay/skepsis`, created **public immediately** (`gh` is authenticated as `abhinay`). The launch is a v0.1 announcement, not a reveal.
- **Docs URL:** `https://abhinay.github.io/skepsis` (GitHub Pages; no custom domain).
- **Demo target:** a famous strategy *type*, no individual named — the optimized moving-average crossover on BTC ("the Golden Cross"). The villain is the strategy every blog posts, not a person.
- **PyPI:** version `0.1.0` via trusted publishing (tag → publish workflow).

## 3. Deliverables

### 3.1 Public repo (first — everything downstream depends on it)

`gh repo create abhinay/skepsis --public` with description and topics; push `main`; verify the CI workflow goes green on GitHub; enable GitHub Pages (deploy from the docs workflow). Repo metadata: description "Is your backtest result real, or overfit? Statistical diagnostics for backtest overfitting", topics `quant`, `backtesting`, `overfitting`, `sharpe-ratio`, `finance`, `python`.

### 3.2 Backlog fixes (from the v1 final review)

1. **Vectorize `stationary_bootstrap_indices`** — replace the per-resample Python loop (~9s for defaults when block length = 1) with batched numpy generation; special-case `mean_block_length == 1.0` as a single `rng.integers` draw. **Benchmark (component-level, exact call):** `stationary_bootstrap_indices(n_obs=4000, mean_block_length=1.0, n_resamples=5000)` — local development target < 1.0s; the CI regression test asserts a deliberately loose bound (< 10s) so loaded runners don't flake. All existing property/behavior tests stay green. The seeded resample stream MAY change — acceptable now, before any user depends on it; note it in the changelog.
2. **Real URLs** — replace `github.com/abhinay/skepsis` placeholders (they are now correct by coincidence of the chosen handle, but verify every occurrence: README install URL, report template footer) and add docs-site links.
3. **De-duplicate `_MIN_BOOTSTRAP_OBS`** — `evaluate.py` imports the threshold from `skepsis.core.bootstrap` instead of redeclaring it.
4. **`chosen=` without `params`** — emit a `SkepsisWarning` that the argument is ignored, instead of silently discarding it.
5. **Report polish** — "1 trial(s)" grammar fixed properly (singular/plural); infinite stability score renders as "∞ (isolated spike)" instead of `inf` in ALL user-visible stability text: the report summary table, any figure caption, `Result.summary()`, and the verdict reason strings (`verdict.py`'s formatter currently produces "stability score inf > 2.0"). One shared formatting helper, not per-site fixes.
6. **Autocorrelation warning** — when the Politis-White block length (already computed in the bootstrap step) exceeds a documented threshold, warn that heavy autocorrelation strains PSR's assumptions. The threshold is a documented module-level constant in `evaluate.py` (`_AUTOCORR_WARN_BLOCK_LENGTH = 10.0`), not a new `evaluate()` parameter (YAGNI: overriding it is editing one constant, and the warning is informational).
7. **`moments.py` citation** — add Sharpe (1994), "The Sharpe Ratio", Journal of Portfolio Management, to the module docstring.
8. **DSR docs note** — the docs explainer (3.4) states explicitly that skepsis uses the raw trial count N (conservative: more deflation), not the paper's effective-trials clustering correction, and why.

### 3.3 Paper-reproduced golden test

Reproduce a published worked example from Bailey & López de Prado's DSR paper (2014) as a golden test in `tests/golden/`, verified against the paper's own printed numbers during plan-writing (not from memory). If the paper's conventions differ from ours (e.g., annualization or variance definitions), the test documents the mapping. The README's "verified against the papers" claim, plus the wording fix ("reference values" → the accurate description), lands in the same change. This is the credibility backbone: a skeptical quant will check exactly this.

### 3.4 Docs site

mkdocs-material, source in `docs/` (mkdocs.yml at repo root), deployed to GitHub Pages by a `docs.yml` workflow on push to main (`mkdocs build --strict` is also a CI gate). Pages:

- **index.md** — the pitch, report screenshot, 5-line quickstart, install.
- **explainers/** — one page per diagnostic (PSR/DSR, PBO-CSCV, bootstrap, sensitivity): the question it answers, the math in plain English, when it lies, a worked example. PSR/DSR page carries the raw-vs-effective-N note (3.2.8).
- **verdict.md** — every threshold, every rule, how to override; explicit "the verdict is a heuristic, read the diagnostics."
- **api.md** — mkdocstrings-generated reference for `evaluate`, `Result`, `Thresholds`, and `skepsis.core.*`.

The existing `docs/superpowers/` planning artifacts are excluded from the built site.

### 3.5 Demo notebook — "Deflating the Golden Cross"

`notebooks/deflating-the-golden-cross.ipynb`:

1. Load committed daily BTC-USD closes from `notebooks/data/btc_usd_daily.csv` (~11 years). Provenance: fetched once by `notebooks/data/fetch_btc.py` from a public exchange API; the script, source, and fetch date are committed alongside the CSV. The notebook itself needs no network.
2. Sweep the fast/slow MA window grid — fast ∈ {5, 10, 15, 20, 25, 30, 40, 50}, slow ∈ {20, 50, 100, 150, 200}, keeping pairs with fast < slow (34 trials: 3 for slow=20, 7 for slow=50, 8 each for 100/150/200) — long-when-fast-above-slow, daily rebalance, no costs (deliberately: the point is that even cost-free it dies).
   **Signal timing (no look-ahead):** the position held on day t is computed from closes up to and including t−1 — i.e., `position = (fast_ma > slow_ma).shift(1)` and `strategy_return_t = position_t × btc_return_t`. Warm-up: drop the first `max(slow windows) = 200` rows from EVERY trial so all 34 return series share one aligned, rectangular date index. The notebook states this explicitly — a look-ahead bug in the signature demo would be fatal to credibility.
3. Show the seduction: best combo's in-sample equity curve and headline Sharpe.
4. The turn: feed the full sweep to `skepsis.evaluate(returns, trials=..., params=...)`; walk through DSR, PBO, sensitivity map, verdict.
5. Save the HTML report; capture the README screenshot from it.

CI executes the notebook end-to-end (nbconvert/jupyter execute via a dev dependency) so it cannot rot. The README screenshot (`docs/assets/report.png`) is captured once locally via Playwright headless Chromium and committed; regeneration instructions live next to it.

### 3.6 PyPI v0.1.0

- Version bump `0.1.0.dev0` → `0.1.0`; `CHANGELOG.md` with an honest v0.1.0 entry (including the bootstrap seed-stream note).
- `release.yml` workflow: triggered by **published GitHub release only** (a single authoritative trigger — a tag-push trigger alongside it could attempt to upload the same immutable version twice). The workflow validates that the release tag equals `v<package version>` from `pyproject.toml` and fails loudly on mismatch, then builds with `uv build` and publishes via PyPI trusted publishing (OIDC, no API token).
- **Manual step (owner only):** register the trusted publisher on pypi.org (project `skepsis`, owner `abhinay`, repo `skepsis`, workflow `release.yml`). The plan includes exact instructions.
- README gets install-from-PyPI plus CI and PyPI badges.

### 3.7 Launch post drafts

Committed under `docs/launch/` (excluded from the docs site): `reddit-r-quant.md`, `hn-show-hn.md`, `linkedin.md`. Each is a complete, honest draft in the owner's voice — what skepsis does, the Golden Cross numbers from the actual notebook run, what it is NOT (no alpha, not a backtester), link to repo/docs/notebook. The owner posts them manually after the PyPI release is live.

## 4. Order

3.1 repo → 3.2 fixes + 3.3 golden (parallelizable) → 3.4 docs + 3.5 notebook (parallelizable; notebook numbers feed 3.7) → 3.6 release → 3.7 posts. CI must be green on GitHub before the release tag.

## 5. Non-goals

- No walk-forward decay, backtester adapters, or CLI (post-v1 roadmap, unchanged).
- No naming or targeting any individual's published strategy or blog.
- No custom domain, no analytics, no paid services.
- No auto-posting anywhere; the owner publishes the launch posts himself.

## 6. Verification

- Full existing gate suite stays green throughout (71+ tests, ruff, mypy strict-on-core).
- New: performance regression test (3.2.1), paper golden test (3.3), `mkdocs build --strict` gate, notebook-execution CI job.
- GitHub CI observed green on the real repo before tagging v0.1.0; PyPI install smoke-tested (`pip install skepsis` in a fresh venv, run the quickstart) after release.

## 7. Risks

- **Paper example mismatch (3.3):** the DSR paper's example may use conventions that need mapping to ours; budget plan-time verification, and if truly irreproducible, document why and pick another published example (e.g., PSR from the 2012 paper) rather than shipping a hollow claim.
- **BTC data licensing:** use a public API whose terms permit redistribution of a derived daily-close CSV; record source and terms in the data README.
- **Seed-stream change (3.2.1):** must land BEFORE v0.1.0 so no released version's seeded outputs are ever broken.
