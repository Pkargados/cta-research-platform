# Cross-sectional momentum (XSMOM) — implementation recipe

**Source:** Asness, Moskowitz, Pedersen (2013), "Value and Momentum Everywhere,"
*Journal of Finance*. `references/Value and Momentum Everywhere.pdf`, read
directly — same source paper as `value_implementation_recipe.md`'s subject
(value.py currently has no separate recipe doc, per CLAUDE.md — see that file's
own note).

Per direct instruction — a deliberate scope expansion, NOT one of this project's
original six signal families.

Derived notes only — this file is not an external source. Regenerated 2026-07-24
during repo reconstruction to match `src/signals/xs_momentum.py` exactly.

## MOM2-12 (paper Section I.B)

The past 12-month cumulative return, **skipping the most recent month**:

```
MOM2-12_t = price[t-1] / price[t-13] - 1        (in month-end-resampled terms)
```

Computed on month-end-resampled BACK-ADJUSTED close (the paper's own convention is
monthly data throughout). Skipping the most recent month is "standard in the
momentum literature... to avoid the 1-month reversal in stock returns" — matched
here even though the paper itself notes this isn't strictly necessary for liquid
futures, "to maintain uniformity across asset classes," the paper's own stated
reason. `signals.xs_momentum.mom2_12(close, lookback_months=12, skip_months=1)`.

## Rank-weighted cross-sectional signal (paper Eq. 1)

```
w_i = rank(S_i) - mean_rank(S)
```

Algebraically identical to carry's Eq. 19 — both AQR/KMPV papers use the same
rank-demean shape, so this reuses the shared `signals.transforms.
cross_sectional_rank` (carry's own `carry1m_signal` is a thin wrapper over the
same function — CLAUDE.md Rule 6: extract once a second consumer exists, XSMOM was
that second consumer).

**Sector-scoped, not the paper's full cross-section** — this project's universe
spans FX/rates/commodities/equities, so comparing AUDUSD's momentum against Corn's
has no clean interpretation (the same reasoning `cross_sectional_rank`'s own
docstring gives).

`signals.xs_momentum.xs_momentum_signal(close, sectors)`.

## No vol-scaling

The paper's own headline construction has none — unlike momentum/breakout/
crossover's `vol_targeted_sign_signal`. `xs_momentum_signal`'s signature is
exactly `(close, sectors)`, nothing else.

## ONE Book, not a multi-horizon grid

The paper is explicit that it deliberately avoids testing several lookbacks "to
minimize the pernicious effects of data snooping" (Section I.B footnote), unlike
TSMOM/breakout/crossover/reversal's own multi-spec Book counts. No lookback grid
is built or reported for this signal.

## Cadence

Monthly rebalancing is a BACKTEST-layer choice (`backtest.engine`'s own monthly
resampling), not baked into the signal module itself — same convention as every
other signal family.

## Documented result (train / validation / test Sharpe)

-0.34 / -1.39 / +0.04 gross, -0.39 / -1.42 / -0.01 net. Weak-to-negative overall,
with validation (spanning the 2020 COVID crash) sharply negative — consistent with
the well-documented "momentum crash" phenomenon around violent V-shaped reversals.
Turnover ~3.3x annualized (net ≈ gross). Reported as found, not tuned (CLAUDE.md
Rule 1/2). See CLAUDE.md's "Cross-sectional momentum (XSMOM)" row.
