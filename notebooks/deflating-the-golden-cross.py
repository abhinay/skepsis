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
# Sharpe **1.46**. A **651×** total return. This is the chart that sells the
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
