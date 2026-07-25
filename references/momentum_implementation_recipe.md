# Time-series momentum (TSMOM) — implementation recipe

**Source:** Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum," *Journal of
Financial Economics*. `references/Time Series Momentum.pdf`, read directly — not
reconstructed from memory or from CLAUDE.md's own summary.

Derived notes only — this file is not an external source. Regenerated 2026-07-24
during repo reconstruction to match `src/signals/momentum.py` exactly.

## Headline formula (paper Section 3.2, Eq. 5)

```
r^TSMOM,s_{t,t+1} = sign(r^s_{t-12,t}) * (40% / sigma^s_t) * r^s_{t,t+1}
```

Go long/short purely on the **sign** of the trailing k-month return; size the bet
to a constant **40% annualized ex-ante volatility target**, divided by the asset's
own current vol estimate. Headline spec: `k = 12` months lookback, `h = 1` month
holding, **no skip month** (the skip-month convention belongs to cross-sectional
momentum/XSMOM, not TSMOM — see `xs_momentum_implementation_recipe.md`).

## Mapping onto this codebase

| Paper concept | Code |
|---|---|
| `sign(r_{t-12,t})` | `signals.momentum.raw_momentum(close, lookback_months=12, skip_months=0)`, sign taken inside `vol_targeted_sign_signal` |
| `40% / sigma^s_t` | `signals.transforms.vol_targeted_sign_signal(raw_signal, vol, target_vol=0.40)` — generic across signal families, not momentum-specific |
| Full Eq. 5 position | `signals.momentum.tsmom_signal(close, vol, lookback_months=12, target_vol=0.40)` |
| Holding period `h` | NOT a parameter of the signal itself — applied at the backtest layer via `backtest.engine.backtest_signal(..., frequency="monthly", holding_months=h)`, which reproduces the paper's own overlapping-vintage-averaging construction |

## Implementation details worth remembering

- **Lookback computed in trading-day terms** (`lookback_months * 21`, the standard
  trading-days-per-month approximation) directly off DAILY closes — not a
  month-end resample at the signal-construction step. Resampling/holding-period
  blending happens downstream in `backtest.engine`, so the same raw daily signal
  can be backtested at any `holding_months` without recomputing it.
- **Input curve:** the BACK-ADJUSTED continuous curve
  (`data.continuous_curve.load_continuous_backadjusted`) — momentum needs clean
  percentage returns across futures rolls.
- **Vol estimator is NOT baked into the signal module.** Two candidates are
  compared on train evidence by the caller (`research/momentum.py`): the paper's
  own EWMA-of-squared-returns (`data.ewma_volatility`) and this project's
  Yang-Zhang (`data.volatility`). Decoupled from lookback `k` — one fixed-horizon
  vol estimate regardless of which `k` is tested, matching the paper (an earlier
  build of this codebase had an incorrect implicit 1:1 coupling between vol
  horizon and momentum lookback — fixed, not reintroduced here).
- **`vol` must already be annualized** — both vol estimators default to
  `annualize=True`. The 40% target itself is an annualized vol target (paper
  Section 4.1); passing a non-annualized vol series silently corrupts leverage
  (documented failure mode: SwissFranc's ~9% annualized vol implied 11-26x
  leverage the one time this constant was dropped for a diagnostic view).

## Specs reported (Table 2 grid — descriptive, not a spec-picker)

- **Headline, fixed a priori:** `k=12mo`, `h=1mo`, `target_vol=0.40`. Never chosen
  by looking at which grid cell scores highest (CLAUDE.md Rule 1/2).
- **Full 8×8 lookback×holding grid** (`GRID_MONTHS = (1, 3, 6, 9, 12, 24, 36, 48)`
  for both k and h, via `backtest.engine`'s own `holding_months`), reproduced as a
  train-period robustness view only, matching the paper's own Table 2.

## Documented result (train / validation / test Sharpe)

0.239 / 0.475 / 0.402 (Yang-Zhang vol estimator, the train-evidence winner over
EWMA's 0.200). All positive, no NaN, no data-snooping. See CLAUDE.md's
"Time-series momentum" row for the full history of bugs found/fixed while
building this (four real, distinct bugs, logged in `WORKFLOW.md` Phase 1).
