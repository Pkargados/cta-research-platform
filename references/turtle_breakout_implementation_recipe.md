# Donchian-channel breakout (Turtle Trading Rules) — implementation recipe

**Source:** the 1983 Turtle Trading Rules (Richard Dennis / William Eckhardt via
Curtis Faith's later published account) — a trading-practice convention, not an
academic paper. No PDF anchors this the way momentum/carry/XSMOM/value have one;
the mechanics below are the well-documented, widely-republished original rules.

Derived notes only — this file is not an external source. Regenerated 2026-07-24
during repo reconstruction to match `src/signals/breakout.py` exactly.

## Two historical systems, both reported as parallel specs

The original Turtles traded BOTH systems simultaneously as a blended book — no
"headline pick" here, matching that precedent:

| System | Entry channel | Exit channel |
|---|---|---|
| System 1 | 20-day | 10-day |
| System 2 | 55-day | 20-day |

## Mechanics

A **dual-channel, path-dependent state machine** (long / flat / short), not a
single vectorized `sign(close - rolling_max)` expression — that can't express
"stay long until price crosses the (looser) EXIT channel," which is the entire
point of the dual-channel design: the exit channel is deliberately looser than the
entry channel so a position isn't stopped out by the same noise that would block a
fresh entry.

- **Entry** (from FLAT): go long on a new `entry_window`-day high, short on a new
  `entry_window`-day low.
- **Exit** (from LONG): flatten on a new `exit_window`-day LOW. (from SHORT):
  flatten on a new `exit_window`-day HIGH.
- A position never flips directly long-to-short or short-to-long in one step — it
  always passes through flat first (`exit_window < entry_window` in both systems,
  so the exit channel triggers before a fresh entry could even be evaluated on the
  same bar).
- Rolling max/min are computed over the PRIOR `window` days, excluding today
  (`close.shift(1)` before the rolling window) — a breakout is a break of the
  prior range, not today's own bar compared to itself.
- First `entry_window` days of any asset's history are NaN — not enough history to
  compute either channel yet.

## Mapping onto this codebase

| Concept | Code |
|---|---|
| Per-asset state machine | `signals.breakout.donchian_state_machine(close_series, entry_window, exit_window)` — explicit walk-forward loop (same pattern as `data.continuous_curve.assign_front_contract`), not vectorized, since state depends on the prior day |
| Applied across the full universe | `signals.breakout.breakout_direction(close, entry_window, exit_window)` -> (T×N) DataFrame of {-1, 0, 1, NaN} |
| Full signal (direction × sizing) | `signals.breakout.breakout_signal(close, vol, entry_window, exit_window, target_vol=0.40)` |
| System 1 / System 2 | `signals.breakout.system1_signal` / `system2_signal` |

## Implementation details worth remembering

- **Input curve:** the BACK-ADJUSTED continuous curve — the OPPOSITE choice from
  momentum's Yang-Zhang input (which needs the RAW curve). A raw roll-date price
  jump could spuriously register as a "new N-day high," a hazard this signal is
  far more exposed to than momentum (whose 12-month average dilutes a single-day
  artifact).
- **Direction updates DAILY — non-negotiable.** Monthly formation (momentum's
  convention) would delay reacting to a breakout until month-end, destroying the
  entire reason to build this signal. Position SIZE re-derivation cadence (daily
  vs. coarser) is a separate, measured question left to `research/breakout.py`,
  not assumed in this module — the module always emits a daily direction signal;
  the caller decides how often to resize it via `backtest.engine.backtest_signal(
  ..., frequency=...)`.
- **Sizing:** `signals.transforms.vol_targeted_sign_signal`, `target_vol=0.40` —
  same generic transform every other signal family uses. `direction` is already in
  {-1, 0, 1}, so the transform's own `sign()` is a no-op on it (`sign(0)=0`
  correctly produces flat, not a divide-by-zero or spurious ±1).
- **Scope note:** "authentic Turtle" here means the entry/exit price-TRIGGER
  logic, not the original 1983 ATR-based money-management system — this project's
  existing vol-targeting supersedes that for sizing (CLAUDE.md Rule 7: re-derive
  mechanics per signal family, don't port a whole external system wholesale).

## Documented result

Real finding: measured turnover is genuinely high (~50-60x annualized) — daily
vol-resizing was tested and ruled out as the cause; the actual driver is pooling
many independently-triggered regimes under one daily gross-exposure-normalized
book, a portfolio-construction-layer issue, not a signal-level one. Pooled Sharpe
is weak-to-negative gross and worse net-of-cost — reported honestly, not tuned
after the fact (CLAUDE.md Rule 1/2). See CLAUDE.md's "Breakout (Donchian)" row.
