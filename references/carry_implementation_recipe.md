# Carry (roll yield) — implementation recipe

**Source:** Koijen, Moskowitz, Pedersen, Vrugt (2018), "Carry," *Journal of
Financial Economics*. `references/Carry.pdf`, read directly.

Derived notes only — this file is not an external source. Regenerated 2026-07-24
during repo reconstruction to match `src/signals/carry.py` exactly (the rebuild
that matches the paper's own construction, not the original non-paper-matching
first attempt).

## Underlying carry formula (unchanged from the original build)

```
Carry = (F_near - F_far) / F_near × 365 / (days to F_far expiry)
```

Computed by `data.term_structure.build_carry_panel` (37 assets from real
exchange-quoted calendar spreads; the 4 ICE softs present in this project's
universe — Coffee, Sugar, Cocoa, Cotton — via a back-differenced proxy,
`is_proxy=True` flagged end-to-end, never blended unlabeled with the real-quote
assets). `signals/carry.py` CONSUMES that panel — it does not recompute carry
itself.

## Four parallel specs, no headline pick

### 1. carry1m — "current carry" (paper's Eq. 19)

```
rank(C_i) - (N_t + 1) / 2
```

Rank-weighted cross-section of the current-month carry level, sector-scoped
(rather than the paper's own single global cross-section — "high carry vs. the
full 42-asset universe" has no clean interpretation, CLAUDE.md Rule 7). Reuses
`signals.transforms.cross_sectional_rank`. **Monthly rebalancing** (the paper's
own stated cadence, p.207) — the single biggest driver of this rebuild's
~120x-to-~1x annualized turnover collapse versus the original, non-paper-matching
first attempt. `signals.carry.carry1m_signal`.

### 2. carry1-12 — trailing 12-month average (paper footnote 14)

Trailing 12-month moving average of the RAW carry level, computed BEFORE ranking,
to smooth seasonal components (equity dividend-payment months, commodity harvest
seasons) a single month's carry can otherwise pick up as noise. Reuses
`backtest.engine.holding_period_positions` directly — that function's own
construction (month-end resample, rolling `holding_months`-month mean, reindexed
forward) turns out to be exactly this smoothing operation, not a new
implementation. `signals.carry.carry1_12_raw` / `carry1_12_signal`.

### 3. carry_timing, reference=0 (paper's Eq. before Table VI)

```
2 × I(C_i > 0) - 1
```

Simple ±1 direction, **no vol-scaling** — the paper's own construction has none (a
deliberate, logged exception to CLAUDE.md Rule 5's general binary-underperforms
finding, made specifically to reproduce this paper's exact published
methodology). Strict `>` test, so a tie goes short, not flat.
`signals.carry.carry_timing_zero_signal`.

### 4. carry_timing, reference=sector-pooled mean (paper's Table VI alternative)

```
2 × I(C_i - C_bar > 0) - 1
```

`C_bar` = "the average carry across all securities... up to that point in time"
(Table VI), adapted to this project's sector-scoped design: one pooled,
expanding-in-time threshold PER SECTOR (not per individual asset — the paper's
single global cross-section has no sector concept to begin with; sector-pooling is
this project's own adaptation, not a literal reading of the paper).
`signals.carry.sector_pooled_expanding_mean` + `carry_timing_mean_signal`.

## Mapping onto this codebase

| Spec | Function |
|---|---|
| carry1m | `carry1m_signal(carry_panel, sectors)` |
| carry1-12 | `carry1_12_signal(carry_panel, sectors, months=12)` |
| carry_timing (ref=0) | `carry_timing_zero_signal(carry_panel)` |
| carry_timing (ref=sector mean) | `carry_timing_mean_signal(carry_panel, sectors)` |
| All four | `build_all_carry_signals(carry_panel, sectors)` |

## Corrections made during the paper-matching rebuild (all logged in CLAUDE.md)

1. Daily rebalancing → monthly (p.207).
2. Raw-magnitude cross-sectional demean → rank-weighted (Eq. 19) — insulates
   weights from outliers, the paper's own stated reason for ranking.
3. Added carry1-12 (footnote 14) alongside carry1m.
4. Carry timing's reference point corrected from each asset's own individual
   expanding mean to a sector-pooled reference (a misreading of an earlier recipe
   doc, not the paper itself).
5. Carry timing's sizing corrected from vol-targeted to a simple ±1 direction —
   the paper's own construction has no per-asset vol-scaling at all.

## Documented result (train / validation / test Sharpe)

- carry1m: -0.01 / -0.58 / -0.68
- carry1-12: +0.33 / -0.36 / -0.06
- carry_timing (ref=0): -0.30 / -0.98 / +0.33
- carry_timing (ref=mean): +0.02 / -0.56 / +0.26

Turnover collapsed from ~120-125x to ~0.7-3.3x annualized once rebalancing matched
the paper's monthly cadence, and net-of-cost is now nearly identical to gross for
every spec — the first build's net-of-cost wipeout was overwhelmingly a
rebalancing-frequency problem, not a transaction-cost-realism problem. Genuinely
mixed, not a clean win: no spec is robustly positive across train/validation/test,
though carry timing turns solidly positive in the test period specifically
(consistent with the paper's own documented finding that carry underperforms in
global recessions — validation spans the 2020 COVID shock). Backtest window capped
at 2026-07-13 (real spread data's frozen end). See CLAUDE.md's "Carry" row.
