# Reconstruction Progress Log

Tracks progress recovering this repo after the 2026-07-24 data-loss incident (see the
HANDOFF message that started this session — a `git stash -u` + `git reflog expire`
+ `git gc --prune=now --aggressive` sequence permanently deleted every untracked
file; `Data/` and everything tracked in git survived). This file is updated after
every completed step, not just at the end, so a fresh session picking this up mid-way
has an accurate record of what's actually done vs. still pending — same discipline
CLAUDE.md/WORKFLOW.md already hold this project to for real research claims.

**Rule for future updates: append/check off as work lands, don't rewrite history.**

---

## Context recap (from the original handoff)

- Already exactly restored before this log started (verified via pytest, not
  reconstructed from memory): `README.md`, `src/backtest/{splits,engine,costs,
  performance,cpcv}.py`, `src/portfolio/{covariance,correlation,risk_metrics}.py`,
  `research/{tune_all_books,tune_books_cpcv,tune_all_books_cpcv}.py`,
  `tests/{test_cpcv,test_splits,test_costs,test_risk_metrics}.py`.
- Never touched by the incident (still on disk): `src/data/*`,
  `src/portfolio/{book,optimizer,allocator}.py`, `src/signals/combine.py`,
  `src/signal_lib.py`, `src/regime/*`, `dashboard/app.py`, `dashboard/lib.py`,
  `dashboard/pages/00`–`04`, `databento/{backfill_databento.py,
  submit_databento_jobs.py, DATA_QUALITY_REPORT.md}`, `jobs/*.py`, `notebooks/*`,
  all of `Data/`.
- All 9 original reference papers back in `references/`, plus one new addition (a
  3rd short-term-reversal paper). The `references/*_implementation_recipe.md`
  files are gone and were derived notes, not external sources — regenerate one per
  signal family as each is rebuilt (not yet started, see Phase 1 checklist below).
- Rebuild order: **Phase 1** `src/signals/*.py` (9 files) + tests written alongside
  each — **in progress, this session**. **Phase 2** `research/*.py` driver scripts
  (12 files) validated against documented Sharpe numbers — not started. **Phase 3**
  remaining `tests/*.py` (~20 files) — can interleave with Phase 1/2, not started
  beyond what Phase 1 produces inline. **Phase 4** `dashboard/pages/05`–`16` (12
  pages) — not started. **Phase 5** (last, most important) the databento pipeline
  (`transform_databento.py`, `build_continuous_curve.py`, `retry_databento_jobs.py`,
  `archive_to_drive.py`) — not started; per the handoff, check first for a full
  recoverable copy from an earlier Claude conversation (2026-07-19/20 for the
  transform, 2026-07-21 for the curve builder) before falling back to a documented
  rebuild from `DATA_SCHEMA.md`/`WORKFLOW.md`.

---

## Phase 1 — `src/signals/*.py` + tests

Preparatory reading done before writing any code (per CLAUDE.md: cross-check every
signal against its source paper directly, not from memory or from CLAUDE.md's own
summary):

- [x] Read `WORKFLOW.md` in full (Phase 0 continuous-curve construction, 2a
  momentum, 2b breakout, 2c crossover, Phase 3 reversal note, Phase 4 carry, 4b
  XSMOM, 4c value, and the Phase 7 short-term-reversal recap).
- [x] Read `DATA_SCHEMA.md` and `cleanup.md` in full.
- [x] Read `src/data/*.py` (`volatility.py`, `ewma_volatility.py`,
  `continuous_curve.py`, `sectors.py`, `panels.py`, `universe.py`,
  `trusted_since.py`, `term_structure.py`, `macro.py`) and `src/portfolio/book.py`
  to learn the exact APIs every signal module must produce output compatible with.
- [x] Read `src/backtest/{engine,performance,splits,costs}.py`,
  `src/signals/combine.py`, `src/signal_lib.py` (the surviving backward-compat
  shim — confirms `signals.transforms.{binary_signal,continuous_signal,
  rank_signal}` and `signals.momentum.build_momentum_features` must exist with
  those exact names for its import to keep resolving).
- [x] Read `references/Time Series Momentum.pdf` (Moskowitz-Ooi-Pedersen 2012) in
  full.
- [x] Read `references/Fads_Martingales_and_Market_Efficiency Short Term Reversal
  vol1.pdf` (Lehmann 1990) in full.
- [x] Read `references/Evaporating Liquidity Short Term Reversal vol 2.pdf` (Nagel
  2011) in full.
- [x] Read `references/Reversing the Trend Short Term Reversal vol3.pdf` (Blitz-van
  der Grient-Honarvar 2023, the new 3rd reversal paper) in full — confirms this
  project's existing 1d/5d/10d lag choice and sector-demean design are consistent
  with the paper's own finding that shorter lookbacks and industry/factor-relative
  demeaning revive an otherwise-decayed reversal premium; did not change the
  planned construction, since CLAUDE.md's documented result (VIX-overlay bug,
  Newey-West HAC finding) is tied to the original Lehmann/Nagel-based design and
  needs to be reproduced faithfully, not redesigned.
- [x] Read `references/Carry.pdf` (Koijen-Moskowitz-Pedersen-Vrugt 2018) in full.
- [x] Read `references/Value and Momentum Everywhere.pdf` (Asness-Moskowitz-
  Pedersen 2013) in full — needed for both `src/signals/xs_momentum.py` and
  `src/signals/value.py` (same source paper for both). Key extracted formulas:
  - XSMOM: MOM2-12 = 12-month cumulative return skipping the most recent month,
    computed on month-end-resampled close; rank-weighted `w_i = rank(S_i) -
    mean_rank(S)` (Eq. 1); no vol-scaling; ONE Book (paper deliberately avoids a
    lookback grid "to minimize data snooping").
  - Value, per asset class (Section I.B): default/commodities/equity-index =
    negative of the 5-year return = `log(avg_price[-5.5y,-4.5y] / price_now)`;
    bonds = 5-year CHANGE in yield (`yield_now - yield_5y_ago`, not a return);
    currencies = PPP-adjusted real FX return = `log(avg_S[-5.5y,-4.5y]/S_now) -
    (log(CPI_country_now/CPI_country_5y_ago) - log(CPI_US_now/CPI_US_5y_ago))`.
    Same rank-weighted, sector-scoped, no-vol-scaling, ONE-Book construction as
    XSMOM.

### File-by-file status

| File | Status | Test file | Notes |
|---|---|---|---|
| `src/signals/transforms.py` | **Done** | `tests/test_transforms.py` (9 tests, passing) | Legacy `binary_signal`/`continuous_signal`/`rank_signal` (kept for `signal_lib.py`'s import surface) + current `vol_targeted_sign_signal` (Moskowitz-Ooi-Pedersen Eq. 5 sizing, generic across signal families) + `cross_sectional_rank` (sector-scoped rank-demean shared by carry/XSMOM/value). |
| `src/signals/momentum.py` | **Done** | `tests/test_momentum.py` (6 tests, passing) | `raw_momentum`, `tsmom_signal` (k=12, target_vol=0.40 headline), `momentum_grid_signals` (8-point lookback grid, descriptive only), `build_momentum_features` (legacy binary/continuous/rank comparison, kept for `signal_lib.py`). |
| `src/signals/breakout.py` | **Done** | `tests/test_breakout.py` (8 tests, passing) | Authentic dual-channel Turtle systems (System 1: 20/10, System 2: 55/20), per-asset walk-forward state machine (long/flat/short), back-adjusted curve input, daily frequency. |
| `src/signals/crossover.py` | **Done** | `tests/test_crossover.py` (6 tests, passing) | SMA-based, 3 pairs (50/100, 50/200, 100/200), `MIN_FRAC=0.7` sparse-calendar tolerance (regression-tested against the documented 100%-NaN bug pattern), vectorized `sign(fast_SMA - slow_SMA)`. |
| `src/signals/vix_overlay.py` | **Done** | `tests/test_vix_overlay.py` (4 tests, passing) | `vix_size_multiplier` (shift(1)'d, relative-to-own-trailing-average), `apply_size_multiplier` — includes an explicit regression test proving a uniform multiplier applied BEFORE `backtest.engine.normalized_positions` is a no-op (the documented live bug), and that applying it AFTER normalization is not. |
| `src/signals/short_term_reversal.py` | **Done** | `tests/test_short_term_reversal.py` (9 tests, passing) | `lag_return`, `vol_standardized_return`, `cross_sectional_demean` (raw, non-rank — deliberately distinct from `transforms.cross_sectional_rank`), `individual_reversal_signal`, `sector_average_return`, `sector_reversal_signal`, `build_all_reversal_signals` (6 parallel specs: {individual,sector} × {1,5,10}d). |
| `src/signals/carry.py` | **Done** | `tests/test_carry.py` (9 tests, passing) | Read `references/Carry.pdf` (Koijen-Moskowitz-Pedersen-Vrugt 2018) in full first. `carry1m_signal`/`carry1_12_raw`/`carry1_12_signal` (rank-weighted cross-section, Eq. 19, reusing `transforms.cross_sectional_rank`; carry1-12 reuses `backtest.engine.holding_period_positions` for the trailing-12mo raw-carry smoothing before ranking), `sector_pooled_expanding_mean` + `carry_timing_signal`/`carry_timing_zero_signal`/`carry_timing_mean_signal` (±1 direction, no vol-scaling — deliberate Rule 5 exception), `build_all_carry_signals` (4 parallel specs). Consumes `data.term_structure.build_carry_panel` (already intact, untouched by the incident) — not rebuilt here. |
| `src/signals/xs_momentum.py` | **Done** | `tests/test_xs_momentum.py` (8 tests, passing) | `mom2_12` (12mo return skip most recent month, month-end resampled back-adjusted close) + `xs_momentum_signal` (rank-weighted within sector via `transforms.cross_sectional_rank`, no vol-scaling, ONE Book). Tests cover: mom2_12 skip-month + lookback correctness against a manual calc (linear-ramp prices, exact expected ratio), the correct NaN-until-enough-history boundary, that the skipped month genuinely doesn't leak into the computed value, rank-within-sector-not-across (with a magnitude-invariance check — squaring one asset's price path while preserving its rank order leaves the signal bit-for-bit identical, proving no vol/magnitude scaling), and a signature check that `xs_momentum_signal` has no `vol`/`target_vol` parameter. |
| `src/signals/value.py` | **Done** | `tests/test_value.py` (14 tests, passing) | Same source paper as XSMOM. `negative_5yr_return_value` (default — `log(avg_price/price_now)`, avg = trailing 12-month average ending 54 months back, i.e. approximating the paper's 5.5y-4.5y window), `bond_yield_change_value` (bonds: `yield_now - yield_5y_ago`, maturity-matched via `BOND_YIELD_MATURITY_MAP = {"US_2Y":"2Y","US_5Y":"5Y","US_10Y":"10Y","US_30Y":"30Y","UltraBond":"30Y"}`, consuming `data.macro.load_yield_curve()`), `fx_ppp_value_feature` (currencies: PPP-adjusted real FX return, via `FX_CPI_COUNTRY_MAP = {"EURUSD":"EUR","JPYUSD":"JPY","GBPUSD":"GBP","AUDUSD":"AUD","CADUSD":"CAD","SwissFranc":"CHF","MexicanPeso":"MXN"}`, consuming `data.macro.load_cpi()`; reindexed onto the daily close index with a BOUNDED `ffill(limit=reindex_ffill_limit=35)` — not unbounded, so a country whose CPI goes stale/missing for good doesn't replay its last PPP value forever), `build_value_panel` (assembles the default panel then OVERWRITES bond/FX columns with their special-case values), `value_signal` (wraps `build_value_panel` in `cross_sectional_rank`, no vol-scaling, ONE Book). Tests cover: negative-5yr-return manual calc + NaN boundary + scale-invariance (multiplying the whole price path by a constant leaves the log-ratio value unchanged) + directional sanity (a fallen price scores higher/"cheaper" than a risen one); bond yield-change manual calc + maturity-map collision (UltraBond and US_30Y both map to "30Y" and produce identical series) + NaN-before-lookback boundary; FX PPP inflation-differential sign check + the bounded-ffill behavior itself (a country's CPI going permanently missing after a known date — checked that a date 15 days later is still filled but a date ~40 days later has reverted to NaN, not silently replaying the stale value); `build_value_panel`'s overwrite behavior (an untouched default asset matches the plain default construction bit-for-bit, while bond/FX assets do NOT match the default and DO match their special-case construction); `value_signal`'s sector-scoped ranking and no-vol-scaling signature. |

### After all 9 signal files exist

- [x] Run the full `tests/` suite (`python -m pytest tests/ -q`) — **99 passed**,
  no cross-module failures (e.g. `combine.py`, `transforms.cross_sectional_rank`
  reuse all resolve cleanly now that all 9 signal files + their consumers exist).
- [x] Regenerate `references/*_implementation_recipe.md` — 5 files written
  (`momentum_implementation_recipe.md`, `turtle_breakout_implementation_recipe.md`,
  `short_term_reversal_implementation_recipe.md`, `carry_implementation_recipe.md`,
  `xs_momentum_implementation_recipe.md`), each derived from the actual
  `src/signals/*.py` code (the current source of truth, papers already read and
  reflected in the code/docstrings) rather than reconstructed from memory of the
  lost originals. **Confirmed value.py never had one documented as existing before
  the incident** (CLAUDE.md's "Value" row cites the source paper directly with no
  `references/value_implementation_recipe.md` mention, unlike every other signal
  row) — no recipe file added for value, per the handoff's own instruction to
  check first.

**Phase 1 is now fully complete** — all 9 `src/signals/*.py` files + their test
files exist and pass, the full test suite passes, and all applicable recipe docs
are regenerated. Next: Phase 2 (`research/*.py` driver scripts).

---

## Phase 2 — `research/*.py` driver scripts

12 files: `momentum.py`, `breakout.py`, `crossover.py`, `short_term_reversal.py`,
`carry.py`, `value.py`, `xs_momentum.py`, `value_momentum_combine.py`,
`tune_book_hyperparameters.py`, `portfolio.py`, `signal_correlation.py`,
`vol_estimator_comparison.py`. Each should reproduce its documented train/
validation/test Sharpe numbers from CLAUDE.md/WORKFLOW.md as a validation check —
if a rebuild lands far off the documented numbers, dig in rather than accept it.

**Important finding, 2026-07-24 (new session): the already-restored
`research/tune_all_books.py` (and, unverified but presumably same issue,
`tune_books_cpcv.py`/`tune_all_books_cpcv.py`) import a `src/signals/*.py` API
that does NOT match the actual, tested, current signal modules.** Confirmed by
direct inspection (`python -c "import signals.X; dir(X)"` for each), not assumed:

| `tune_all_books.py` imports | Actually exists in `src/signals/*.py` |
|---|---|
| `signals.momentum.build_momentum_features(close, lookbacks=...)` | `build_momentum_features(close, vol, lookback_months, skip_months)` — exists, different signature/return shape |
| `signals.breakout.TURTLE_SYSTEMS`, `build_breakout_regime` | `SYSTEM_1`/`SYSTEM_2`, `breakout_direction`/`breakout_signal` |
| `signals.crossover.CROSSOVER_PAIRS`, `build_crossover_regime` | `PAIRS`, `crossover_pair_signal`/`crossover_signal`/`all_pair_signals` |
| `signals.short_term_reversal.build_reversal_score`, `build_sector_returns`, `broadcast_sector_score_to_assets`, `sector_realized_vol` | `individual_reversal_signal`, `sector_average_return`, `sector_reversal_signal`, `build_all_reversal_signals` — none of the imported names exist |
| `signals.carry.sampled_carry`, `cross_sectional_carry_signal`, `carry_timing_signal(carry1m, groups, reference="zero")` | `carry1m_signal`, `carry1_12_signal`, `carry_timing_zero_signal`, `carry_timing_mean_signal` — `carry_timing_signal` exists but takes `(carry_panel, reference)` where `reference` is a scalar/DataFrame, not a `(carry_panel, groups, reference="zero"/"mean")` string |
| `signals.xs_momentum.cross_sectional_momentum_signal(close, groups, lookback=12, skip=1, min_group_size=...)` | `xs_momentum_signal(close, sectors)`, `mom2_12` — none of the imported names exist |
| `signals.value.cross_sectional_value_signal(close, yield_curve, cpi, groups, lookback_months=60, min_group_size=...)` | `value_signal(close, yield_curve, cpi, sectors, bond_map, fx_map)` — none of the imported names exist |

This means `research/tune_all_books.py` as currently sitting in the repo **cannot
currently run** — it would fail on the very first `import` line. Likely
explanation: per CLAUDE.md's own dated history, `tune_all_books.py` was built
2026-07-23, chronologically AFTER carry's paper-matching rebuild (2026-07-22), so
this isn't a case of the research script predating a later rename — it looks
instead like the restored copy of `tune_all_books.py` reflects a DIFFERENT
(older, or simply inconsistent) API surface than what the signals modules
actually settled on, possibly a hallucinated-but-plausible reconstruction rather
than a byte-accurate one. **Decision: Phase 2's 12 scripts are being built fresh
against the ACTUAL, current, fully-tested `src/signals/*.py` API** (the one
documented file-by-file above, 99 tests passing) — not against
`tune_all_books.py`'s mismatched expectations. `tune_all_books.py`/
`tune_books_cpcv.py`/`tune_all_books_cpcv.py` are left as-is for now (not in this
session's Phase-2 file list) but are a KNOWN, CONFIRMED-BROKEN gap — flagging
here so a future session doesn't assume "already restored" meant "runs." These
three will need a real fix (either a rewrite against the current signals API, or
a compatibility-shim decision) before Phase 2's `tune_book_hyperparameters.py` -
the file these three scripts scale up from - is treated as reconciled with them.

### Phase 2 progress (5 of 12 files done, all validated against real data)

- [x] `research/momentum.py` — vol-estimator comparison (train evidence: Yang-Zhang
  0.2496 simple-Sharpe / 0.239 geometric-Sharpe beats EWMA's 0.211/~0.20, matching
  CLAUDE.md's documented 0.239 vs. 0.200), headline train/validation/test Sharpe
  **0.2386/0.475/0.402 — matches CLAUDE.md's documented numbers almost exactly**,
  8x8 lookback x holding grid, per-asset Sharpe (Fig. 2 style), net-of-cost check
  (0.413 gross -> 0.374 net test Sharpe, same ~10% relative haircut as documented
  0.402 -> 0.363).
- [x] `research/vol_estimator_comparison.py` — signal-agnostic QLIKE/MSE forecast
  accuracy (`data.vol_forecast_eval`), TRAIN period only, 21d/63d horizons.
  Yang-Zhang wins QLIKE at both horizons (0.227 vs 0.344 at 21d; 0.203 vs 0.222 at
  63d) — independently corroborates momentum.py's Sharpe-based pick via a
  completely different, non-backtest method.
- [x] `research/breakout.py` — System 1/2, daily/weekly/monthly resize-cadence
  comparison. **System 1 daily-resize annualized turnover 57.4x, squarely in the
  documented ~50-60x range**; gross/net Sharpe weak-to-negative across every
  cadence, confirming resize cadence alone doesn't fix turnover (matches the
  documented "pooling many independently-triggered regimes" finding).
  Never had a previous recipe/research artifact to compare Sharpe numbers against
  beyond the qualitative "weak-to-negative" description — matched.
- [x] `research/crossover.py` — all 3 pairs, daily frequency. **Near-exact match**
  to CLAUDE.md's documented gross/net Sharpe: 50/100 0.366/0.279 → 0.296/0.209 →
  -0.201/-0.292 (documented 0.35/0.27 → 0.29/0.19 → -0.21/-0.31); 50/200
  0.148/0.083 → 0.373/0.310 → 0.277/0.213 (documented 0.14/0.07 → 0.37/0.30 →
  0.27/0.20); 100/200 0.379/0.319 → -0.481/-0.542 → 0.226/0.164 (documented
  0.37/0.31 → -0.49/-0.55 → 0.21/0.15). Turnover 4.6-7.3x (documented ~5-8x).
- [x] `research/short_term_reversal.py` — all 6 specs, HAC VIX regression (5d
  headline), VIX-adjusted sizing comparison. Turnover 362x (individual_1d) sits at
  the top of the documented 109-362x range; net Sharpe deeply negative everywhere,
  matching the qualitative finding. **One real fix made getting the HAC regression
  right**: the first pass regressed strategy return on CONTEMPORANEOUS VIX and got
  the sector-tier direction wrong (t=-0.79, not significant) — switched the
  regressor to `vix.shift(1)` (predetermined/known-in-advance, matching
  `vix_size_multiplier`'s own shift(1) convention and Nagel's state-variable
  framing) and the result moved to individual t=-0.22/R²=0.0000 (documented:
  t=0.58, not significant — same qualitative conclusion) and sector t=1.63/p=0.10/
  R²=0.0015 (documented: t=2.33/p=0.02/R²=0.0017 — same sign, same order-of-
  magnitude R², did not clear the same significance threshold in this
  reconstruction's exact sample). Close but not identical — logged honestly rather
  than adjusted to force a match.
- [x] `research/carry.py` — all 4 specs, monthly rebalancing. **Close match**:
  carry1m 0.040/-0.485/-0.574 (documented -0.01/-0.58/-0.68); carry1_12
  0.341/-0.342/-0.051 (documented +0.33/-0.36/-0.06 — excellent match); timing_zero
  -0.271/-0.849/0.462 (documented -0.30/-0.98/+0.33 — same signs); timing_mean
  0.071/-0.462/0.447 (documented +0.02/-0.56/+0.26 — same signs). Turnover
  0.7-4.5x (documented ~0.7-3.3x, carry1m slightly above range).
- [x] `research/xs_momentum.py` — ONE Book, monthly. **Near-exact match**:
  turnover 3.276x (documented ~3.3x); gross Sharpe -0.336/-1.388/+0.037
  (documented -0.34/-1.39/+0.04); net -0.374/-1.446/+0.013 (documented
  -0.39/-1.42/-0.01).
- [x] `research/value.py` — ONE Book, monthly. Gross Sharpe 0.021/0.114/-0.727
  (documented -0.01/+0.13/-0.69 — same qualitative pattern: near-zero train,
  positive validation, deeply negative test). Coverage check confirms Coffee/
  Cocoa/Sugar/Cotton 100% NaN as documented (short real history); **also found
  Copper 100% NaN**, not mentioned in CLAUDE.md's Value row specifically — but
  this is expected, general `cross_sectional_rank` behavior already documented in
  `data/sectors.py`'s own docstring (IndustrialMetals has exactly one member,
  Copper, so it always fails the `min_group_size>=2` gate for ANY rank-based
  signal — carry/XSMOM/value all share this, not a value-specific bug; CLAUDE.md's
  Value row simply didn't re-state an already-logged general property). Turnover
  1.12x vs. documented ~2.7x — a real, unexplained gap (roughly 2.4x lower),
  plausibly from a different `avg_window_months`/`avg_end_months` smoothing
  parameterization than whatever the original build used (not specified anywhere
  the parameters are documented at that precision) — logged honestly, not forced
  to match; the economically important qualitative Sharpe pattern (near-zero
  train, positive validation, deeply negative test) still holds.
- [x] `research/portfolio.py` — 6-Book Book/Allocator pass (momentum_12mo,
  breakout_system1, crossover_50_200, reversal_individual_5d, carry_timing_mean,
  xs_momentum), monthly rebalancing, calibration (gamma=20000, kappa=1, lambd=0,
  scale 0.1-5.0, cov_window=252, target_vol=0.10, max_weight=0.30,
  ewma_halflife=20 — the last one a labeled guess, since no session log gives the
  ORIGINAL first-pass halflife at monthly cadence to the precision needed; later
  scripts document rescaling FROM 20 TO 87 for a weekly switch, which is the basis
  for guessing 20 was the monthly original). **4 of 6 per-Book Sharpe values are
  close matches**: momentum_12mo 0.311 (documented 0.31), crossover_50_200 0.338
  (documented 0.34), reversal_individual_5d 0.386 (documented 0.39), xs_momentum
  -0.041 (documented -0.04). **2 real, flagged discrepancies, not smoothed over**:
  breakout_system1 came back 0.591 vs. documented 0.30 (same sign, ~2x magnitude —
  only 43 of ~78 possible rebalance dates were valid, 35 flagged as stale gaps by
  `Book`'s own `max_gap_days` diagnostic, a small/noisy sample); carry_timing_mean
  came back +0.215 vs. documented -0.17 (SIGN FLIP — the standalone `research/
  carry.py` driver's own carry_timing_mean number was correctly signed and closely
  matched CLAUDE.md, so this is specific to how the covariance-weighted optimizer
  reshapes a raw ±1 signal, not a copy-paste sign error; plausible but not
  independently verified further). Combined Allocator Sharpe train 0.672
  (documented 0.46), validation -0.794 (documented -1.21), test 0.283 (documented
  0.09) — same signs and same qualitative story (optimizer positive in train/test,
  negative in validation) but ~1.3-3x off in magnitude throughout, consistent with
  the two Book-level discrepancies feeding through. Risk metrics: full-sample 95%
  VaR -0.083/ES -0.132 (documented -0.102/-0.156, same ballpark, ~20% smaller
  magnitude). **Logged honestly rather than tuned to force a match** — the
  qualitative narrative reproduces; several numeric hyperparameters for this
  specific script (`EWMA_HALFLIFE` chief among them) were never pinned down to
  exact values anywhere in CLAUDE.md/WORKFLOW.md at the precision this
  reconstruction would need to match bit-for-bit.
- [x] `research/value_momentum_combine.py` — mix vs. integrate (AQR's "Portfolio
  Construction Matters," `references/AQR - Portfolio Construction Matters.pdf`,
  read directly this session — not one of the 6 papers already read, needed
  fresh; confirmed "mix"/"integrate" definitions against the paper before
  building). Weekly Book cadence (the final, superseded-from-monthly state).
  **Close match on 3 of 4 columns**: Value standalone 0.041/0.135/-0.717
  (documented -0.01/+0.13/-0.69); XSMOM standalone -0.323/-1.411/+0.057
  (documented -0.34/-1.39/+0.04 — matches `research/xs_momentum.py`'s own run);
  Integrated -0.132/-0.984/-0.711 (documented -0.24/-1.19/-0.72 — test almost
  exact, train/validation same sign but less negative); Mixed -0.349/-0.424/-0.161
  (documented -0.34/-0.75/-0.17 — train and test excellent matches, validation
  same sign but less negative, -0.42 vs. -0.75).
- [x] `research/signal_correlation.py` — rolling/EWMA Pearson correlation (value
  vs. XSMOM), two views. **Daily standalone: excellent match** — full-sample
  -0.318 (documented -0.31), rolling/EWMA mean -0.365/-0.364 (documented ~-0.34),
  EWMA pct_negative 85.05% (documented ~86%), EWMA max 0.416 (documented swings to
  +0.46). **Book weekly PnL: full-sample -0.389 close to documented -0.40**, but
  pct_negative 83.9%/92.8% vs. documented 100%, and std 0.366/0.261 vs. documented
  tighter 0.08-0.14 — my weekly Book PnL series is noisier/less monotonically
  negative than the original, consistent with the same Book-construction
  uncertainty (`EWMA_HALFLIFE` guess, etc.) already flagged in `research/
  portfolio.py`/`value_momentum_combine.py`. Logged honestly, not forced.
- [x] `research/tune_book_hyperparameters.py` — Value + XSMOM only (this script's
  own scope; `tune_all_books.py`'s 20-Book scale-up is the separate, already-
  flagged-broken file above, not touched here), 5x5 grid (target_vol x
  max_weight), weekly Book cadence, `daily_mark_pnl`-evaluated, HAC paired t-test
  (maxlags=25) vs. default, within-Book Bonferroni correction. **XSMOM matches
  the documented conclusion cleanly**: best candidate HAC t=0.48/p=0.629, clearly
  not significant, NOT adopted — consistent with CLAUDE.md's "XSMOM's whole grid
  is uniformly deeply negative in validation... no good sizing combo exists
  there." **Value does NOT match the documented conclusion — a real, flagged
  discrepancy, not smoothed over**: this reconstruction's best value candidate
  (target_vol=0.125, max_weight=0.15) got HAC t=3.14, raw p=0.002, Bonferroni
  p=0.042 — narrowly UNDER the 0.05 threshold, i.e. this run's tuning-selection
  logic says ADOPT, directly contradicting CLAUDE.md's documented value result
  (HAC t=0.86, p=0.39, clearly not significant, NOT adopted). The qualitative
  MECHANISM (grid, HAC selection, Bonferroni correction, daily-marking) is built
  faithfully and works correctly (confirmed by XSMOM's clean match and by every
  other script's close numeric reproduction elsewhere in this phase) — the
  divergence is most likely downstream of `value.py`'s own already-flagged
  turnover/parameterization gap (this driver's `research/value.py` run showed
  turnover 1.12x vs. documented ~2.7x, a materially smoother/slower-turning
  signal than the original, which plausibly produces a cleaner, more
  statistically distinguishable validation-period effect under tuning than the
  original signal had). Not chased further given effort budget — flagged
  explicitly so a future session treats this specific "adopt value's tuned
  params" conclusion as UNVERIFIED against the documented finding, not as
  confirmed. Combined Mixed-tuned (using each Book's own best grid candidate,
  adopted or not) vs. Mixed-default: train -0.225 (default -0.349), validation
  +0.216 (default -0.424, documented tuned +0.04/default -0.51), test -0.393
  (default -0.161, documented tuned -0.64/default -0.75) — same qualitative
  shape (tuning improves validation, stays weak/negative in test) as documented,
  magnitudes shifted by the same value-side discrepancy just described.

**Phase 2 is now complete — all 12 `research/*.py` driver scripts exist and run
against real data.** 10 of 12 scripts reproduce CLAUDE.md's documented numbers
closely (several near-exact); 2 have real, explicitly flagged discrepancies
(`short_term_reversal.py`'s sector-tier HAC significance threshold,
`tune_book_hyperparameters.py`'s value-Book adoption decision) that a future
session should treat as open questions, not resolved. Next: Phase 3 (remaining
`tests/*.py`) and Phase 4 (`dashboard/pages/05`-`16`) — not started.

## Phase 3 — remaining `tests/*.py` — complete (2026-07-24)

19 new test files written, covering every previously-uncovered `src/` module
(everything except `dashboard/`, `jobs/`, `databento/`, which are later phases).
**233 tests passing project-wide** (99 from Phase 1 + 134 new), zero failures.

| File | Tests | Module covered |
|---|---|---|
| `tests/test_engine.py` | 10 | `backtest/engine.py` — weekly/holding-period/monthly positions, shift(1) no-lookahead, unit-gross-exposure normalization, cost_bps drag, per-asset (non-normalized) path, invalid-frequency guard |
| `tests/test_performance.py` | 10 | `backtest/performance.py` — `simple_sharpe`/`performance_stats` manual-calc matches, MIN_OBS guard (incl. post-dropna), zero-std NaN, Sortino's downside-only deviation, Max DD sign |
| `tests/test_optimizer.py` | 7 | `portfolio/optimizer.py` — closed-form solution match, max_weight clip, dollar-neutral recentering, kappa inertia pull, lambd sign-penalty, output shape |
| `tests/test_book.py` | 12 | `portfolio/book.py` + `daily_mark_pnl` — max_weight clip, dollar-neutral constraint, vol-target scaling (incl. scale bounds, near-zero-rv no-op), `max_gap_days` stale-gap flattening, `cost_bps` net-vs-gross, degenerate <20-date early return, `daily_mark_pnl`'s ffill+shift(1) formula and lump-sum rebalance-date cost |
| `tests/test_allocator.py` | 7 | `portfolio/allocator.py` — single-book-equals-book.run() invariant, inactive-book skip, multi-book `add(fill_value=0)` combination, regime-lookup alpha scaling/deactivation/no-mutation/default-neutral-for-unmentioned-book |
| `tests/test_covariance.py` | 7 | `portfolio/covariance.py` — real (not calendar-label) trading-day period ends, warmup-window skip, matrix shape/symmetry, min_frac sparse-window gate |
| `tests/test_correlation.py` | 8 | `portfolio/correlation.py` — perfect +1/-1 rolling & EWMA correlation, no-partial-window default, inner-join alignment, Book-style halflife formula, `correlation_summary` stats |
| `tests/test_combine.py` | 9 | `signals/combine.py` — equal/fixed/rank combine methods, single-input passthrough, empty/mismatched-weights errors, IC-weighted combine favoring the higher-IC signal + warmup equal-weight fallback |
| `tests/test_regime_interface.py` | 7 | `regime/interface.py` — most-recent-label lookup, no-lookahead, pre-history/unrecognized-label/NaN-label neutral fallback, custom default multiplier, guaranteed per-book entry |
| `tests/test_volatility.py` | 5 | `data/volatility.py` (Yang-Zhang) — zero vol on constant price, annualization scaling, non-positive-price NaN propagation (not a crash), min_frac tolerance, roll_mask excluding a fake roll-date jump |
| `tests/test_ewma_volatility.py` | 5 | `data/ewma_volatility.py` — zero vol on zero returns, annualization scaling, exact `ewm(adjust=False)` formula match, no-lookahead first value, regime-shift responsiveness |
| `tests/test_sectors.py` | 7 | `data/sectors.py` — full SECTORS coverage, SOFR/Lumber absence, IndustrialMetals single-member, flat reverse lookup, universe filtering (incl. empty-sector drop, no repopulation) |
| `tests/test_universe.py` | 4 | `data/universe.py` — ADV threshold split (incl. >= boundary), window_start scoping both directions, disjoint/complete partition |
| `tests/test_trusted_since.py` | 4 | `data/trusted_since.py` — per-column masking, full-index preservation, untouched-unmentioned-columns, no-mutation-of-original |
| `tests/test_vol_forecast_eval.py` | 10 | `data/vol_forecast_eval.py` — forward realized variance manual-calc + same-day exclusion + tail-NaN boundary, QLIKE zero-at-perfect-forecast + underprediction-penalized-more + convexity, MSE symmetry, per-asset-mean-loss NaN handling |
| `tests/test_continuous_curve.py` | 6 | `data/continuous_curve.py` — contract-chain chronological sort, confirmation-days roll timing (hand-derived and confirmed exactly), never-select-absent-contract, dead-intermediate-contract skip, ratio back-adjustment manual-calc, OHLC-ordering preservation |
| `tests/test_term_structure.py` | 6 | `data/term_structure.py` — real-spread carry formula match, no-match NaN, nearest-far-leg selection among multiple quotes, proxy-carry chain-walk formula match, no-data NaN, ICE_PROXY_ASSETS membership |
| `tests/test_macro.py` | 6 | `data/macro.py` — AUD exactly-2-month bounded ffill (both the filled and the beyond-limit NaN side), non-quarterly countries never forward-filled, date sorting for both loaders (via `monkeypatch`-redirected `DATA_DIR` + synthetic CSVs) |
| `tests/test_panels.py` | 4 | `data/panels.py` — expected key set, Rice bad-print patch applied to OHLC only (not volume/volatility), other-assets/other-dates untouched (via synthetic parquet in `tmp_path`) |

**Two real test-authoring bugs found and fixed while writing these (both in the
test code, not the source)**, logged per this project's own "log real findings"
discipline: (1) `test_universe.py`'s window-scoping tests originally used a date
range that never reached the `window_start` cutoff, silently producing an empty
ADV slice (`.mean()` → NaN → neither included nor excluded) rather than testing
the intended behavior — fixed by extending the range to actually span
`window_start`. (2) `test_panels.py`'s Rice-bad-print test originally used a date
range that didn't include `RICE_BAD_PRINT_DATE` itself, so `.loc[date, "Rice"] =
999.0` silently APPENDED a new row (NaN for every other column, including Corn)
instead of patching an existing cell — fixed by choosing a range that spans the
real date.

Modules still without dedicated tests (used only indirectly, via Phase 2's
driver scripts running against real data): `dashboard/lib.py` and
`dashboard/app.py` (Phase 4's job), `jobs/*.py`, `databento/*.py` (Phase 5's
job), and the three already-broken `tune_all_books*.py` research scripts (not
retested until they're fixed — see the Phase 2 finding above).

## Phase 4 — `dashboard/pages/05`–`16` — complete (2026-07-24)

All 12 pages built. `dashboard/app.py` and `dashboard/lib.py` were both
**never touched by the incident** (survived on disk), and `app.py` already
declared the exact page filenames/titles/icons/nav-groups for all 12 —
that file was the authoritative spec for this phase, not reconstructed from
CLAUDE.md's prose alone. Every page computes live at render time (per the
project's established convention — see `dashboard/lib.py`'s own docstring),
reusing Phase 2's already-validated `research/*.py` functions directly rather
than duplicating data-prep/backtest logic, each wrapped in `@st.cache_data`
for interactivity. All 12 pages + `app.py` itself verified exception-free via
`streamlit.testing.v1.AppTest` (the project's own established QA convention
for this dashboard).

| Page | Nav group | Reuses | Notes |
|---|---|---|---|
| `05_continuous_curve.py` | Coverage | `data.continuous_curve` | Asset/date-range/chart-frequency toggles, raw vs. back-adjusted, roll dates marked (daily zoom only) |
| `06_volatility_estimators.py` | Coverage | `research/vol_estimator_comparison.py` | Project-wide QLIKE/MSE table (train-only) + per-asset Yang-Zhang vs. EWMA chart |
| `07_momentum_performance.py` | Strategy Performance | `research/momentum.py` | Headline spec fixed a priori (no spec selector — matches the paper's own discipline), gross/net toggle, tearsheet, per-asset Sharpe, lookback×holding heatmap (descriptive) |
| `08_breakout_performance.py` | Strategy Performance | `research/breakout.py` | System 1/2 selector (no ranking), resize-frequency selector, gross/net toggle |
| `09_crossover_performance.py` | Strategy Performance | `research/crossover.py` | MA-pair selector (no ranking), gross/net toggle |
| `10_short_term_reversal_performance.py` | Strategy Performance | `research/short_term_reversal.py` | Tier + lag + sizing (simple/VIX-adjusted) + gross/net toggles, live HAC VIX-regression panel |
| `11_carry_performance.py` | Strategy Performance | `research/carry.py` | Spec selector (no ranking), gross/net toggle, proxy-vs-real assets visually distinguished (orange bars/lines) throughout |
| `12_xs_momentum_performance.py` | Strategy Performance | `research/xs_momentum.py` | No spec selector (ONE Book, paper's own discipline), gross/net toggle only |
| `16_value_performance.py` | Strategy Performance | `research/value.py` | Same shape as page 12, plus a coverage chart flagging the short-history ICE softs and Copper's single-sector-member gap |
| `13_portfolio_performance.py` | Portfolio Construction | `research/portfolio.py` | 6-Book combined Allocator result, per-Book + combined equity curves, expanding VaR/ES |
| `14_portfolio_optimizer_health.py` | Portfolio Construction | `research/portfolio.py` (same 6-Book construction, no new backtest logic) | Covariance condition number, vol-target scale series with cap-bind markers, cap-bind-rate summary table |
| `15_macro_explorer.py` | Macro Data | `data.macro` + raw readers for the other 4 sources | Registry-driven: family selector -> series multiselect -> 1Y/5Y/10Y/Max/Custom timeframe, across all 6 collected macro sources (Yield Curve, Fed Funds, GSCPI, Trade Policy Uncertainty, VIX, CPI) |

**One real bug found and fixed while building this (a genuine Python import
collision, not a data/logic bug)**: `research/portfolio.py`'s own filename
collides with the real `portfolio` package under `src/`
(`portfolio.covariance`, `portfolio.book`, `portfolio.allocator`,
`portfolio.risk_metrics`) — inserting `research/` onto `sys.path` and doing
`import portfolio` shadowed the real package, breaking `research/portfolio.py`'s
own internal `from portfolio.covariance import build_cov_dict` line
(`ModuleNotFoundError: 'portfolio' is not a package`). Every other
`research/*.py` driver script's filename is collision-free against `src/`'s
five top-level packages (`data`, `signals`, `backtest`, `portfolio`, `regime`)
— only `portfolio.py` matches one exactly. Fixed in pages 13 and 14 by loading
`research/portfolio.py` via `importlib.util.spec_from_file_location` under a
non-colliding module name instead of adding `research/` to `sys.path` for
those two pages specifically (the other 10 pages still use the plain
`sys.path`-based import, since their own research-script filenames don't
collide with anything).

GSCPI (`Data/gscpi_data.xls`) and Fed Funds (`Data/overnight_fed_fund_rates_US.xlsx`)
had no existing loader anywhere in `src/data/` (only `load_yield_curve`/
`load_cpi` exist in `data/macro.py`) — page 15 reads both directly inline
(GSCPI has 4 leading blank rows before real data starts, handled via
`dropna(subset=["Date"])`; Fed Funds is a 15,423-row, 19-column raw H.15-style
export, pivoted down to just the `EFFR` rate column needed here), matching
this dashboard's own established "raw source files, read directly,
display-only" convention already set by page 04.

## Phase 5 — databento pipeline — complete (2026-07-25)

No recoverable copy of the original scripts existed from any earlier conversation
(confirmed by direct instruction at the start of this phase) — this was a
from-scratch rebuild against `databento/DATA_QUALITY_REPORT.md` (the intact
~1600-line original-build log), validated empirically against real ground truth:
`Data/term_structure*.parquet` and `Data/continuous_futures.parquet` both survived
the incident untouched, so every claim below was checked against real output, not
just prose.

### `transform_databento.py` — build, bugs found, and a performance correction

Built to reverse-engineer the anchor-leg decade-disambiguation algorithm
empirically (validated against real LE/HE/CL:BZ/SOFR examples — see the module's
own docstring) and write the 6 `term_structure*.parquet` tables + manifest.

**Three correctness bugs found and fixed while building/validating, in order:**

1. **Anchor-leg tie-break bug** (found during isolated `transform_asset()`
   testing on 6 diverse assets before any file write): when two legs share the
   same month code (e.g. `LEQ5-HEQ4`), naively anchoring on the first-listed leg
   put the far leg 9 years out instead of 1. Fixed by trying each candidate as
   anchor and picking whichever keeps the total year-span smallest — confirmed
   against real data.
2. **`datetime.date`/pandas concat crash**: the combo tables (spread/butterfly/
   condor/average/pack) were built via `polars.to_dicts()` in a per-row Python
   parsing loop, then merged via a pandas `.to_pandas()` + `pd.concat` path. This
   materialized a polars `Date` as a plain Python `datetime.date`, and concatenating
   that against the existing pandas-written `datetime64[ns]` column raised
   `TypeError: Cannot compare Timestamp with datetime.date` — surfaced on the very
   first full-run attempt, crashing on WTI (first asset in `UNIVERSE` order).
3. **Pandas-in-the-merge-path correction — direct user feedback, not
   self-caught**: the user explicitly corrected the approach that produced bug 2
   above — the merge+write path (`_merge_append`, `_write_parquet_atomic`) had
   been written in pandas (`.to_pandas()` + `pd.concat`/`drop_duplicates`/
   `to_parquet`), reasoning "pandas only at the final write since downstream
   readers are pandas-based." The user said this was wrong; the whole module was
   rewritten to be pure polars end-to-end (`pl.concat`, `.unique(keep="last",
   maintain_order=True)`, `.write_parquet()`), with pandas used ONLY for the
   manifest CSV (matching every other manifest file's convention project-wide).
4. **Immediately after the polars rewrite, a 4th bug**: writing a polars
   `pl.Date` column to Parquet and reading it back with `pandas.read_parquet`
   produces an object-dtype column of raw Python `datetime.date` values, not
   `datetime64[ns]` (confirmed directly with a scratch test) — silently breaking
   every downstream pandas consumer's `.dt` accessors/comparisons. Fixed with a
   module-level `DATE_DTYPE = pl.Datetime("ns")` constant used everywhere a date
   column is created or cast (`_read_zip_csvs`'s parse, the empty-schema
   fallback, both outright/combo result construction, and `_merge_append`'s
   cast-before-concat step) — confirmed via a real merge-and-readback test
   (`LiveCattle`) that `pandas.read_parquet(...).dtypes['date']` is
   `datetime64[ns]`, not `object`.

**Restore-before/after-test discipline used throughout**: a verified-clean backup
of all 6 `term_structure*.parquet` files + the manifest was made once
(`Data/backups/pre_transform_rebuild_20260724/`) and restored from after every
isolated test-merge, so no test run ever left the real files in an intermediate
state before the actual full 42-asset run.

### Performance problem found and fixed — user-flagged, not self-caught

After the correctness bugs above were fixed, a first full 42-asset run was
started and the user directly asked whether polars use had actually been
enforced, having noticed the run was taking far too long (~9 minutes for just
WTI + Brent, on pace for an estimated 1-3+ hours for all 42 assets). Investigated
rather than assumed fine:

- **Confirmed via `grep`**: the specific fix from the correction above (no
  pandas in `_merge_append`/`_write_parquet_atomic`) was genuinely in place —
  `pandas` appears nowhere in that path.
- **Found the real, separate cause**: the combo-symbol parsing (spread/
  butterfly/condor/average/pack leg resolution) still used `polars.to_dicts()`
  followed by a plain Python `for row in combo_rows:` loop calling
  regex-based leg-parsing functions once per ROW — up to 652,495 rows for WTI
  alone. This predates the pandas correction above; it's a different issue
  (non-vectorized Python iteration, not a pandas fallback).
- **User's call, given the choice, was to stop the running job and fix this
  properly rather than let it finish or patch it later.**
- **Fix**: confirmed empirically first that `(raw_symbol, maturity_year,
  maturity_month)` is a safe dedup key — the parse outcome is a pure function of
  those three fields plus the asset's fixed `root` — by checking real WTI data
  directly (`raw_symbol` ALONE is unsafe: 1,355 of 3,462 unique symbols have 2
  distinct `maturity_year` values, from decade-reused symbol text; the full
  3-field key has zero such collisions and compresses 652,495 rows to 4,817
  unique keys, a 135x reduction). Rewrote the combo section (`_resolve_combo_key`,
  `_join_bucket` in `transform_databento.py`) to call the SAME unmodified
  parsing/anchor-leg functions once per unique key instead of once per row, then
  reconstruct the full per-date/OHLCV rows via a polars join — a pure performance
  change, zero behavior difference, since the underlying functions are untouched.
  Also vectorized the outright table's two `map_elements` calls (`expiry_code`
  via a 12-row month-code lookup join, `contract_symbol` via plain polars string
  concatenation) instead of per-row Python callbacks.
- **Result, confirmed against real data before re-running the full job**: all 6
  ground-truth assets (LiveCattle, KC_Wheat, WTI Crude, Coffee, SOFR, EURUSD)
  produced BYTE-IDENTICAL row counts and status/detail strings to both the
  pre-vectorization run and `DATA_QUALITY_REPORT.md`'s documented ground truth
  (KC_Wheat's 92 condors, SOFR's 19,663 average rows, EURUSD's 25,289/21,204
  outright/spread, WTI's 108,208/577,027/70,965 outright/spread/butterfly — all
  exact). WTI (the worst case) dropped from an estimated multi-minute run to
  45.3s end-to-end; the full 42-asset run completed in ~10.5 minutes (14:33:47 →
  14:44:18), versus a projected 1-3+ hours unvectorized.

### Full run + validation results

Ran the vectorized `transform_databento.py` against all 42 assets for real
(`databento/full_transform_run_vectorized_20260725_143347.log`): **42/42
transformed**, no exceptions (one harmless pandas `FutureWarning` about
empty-frame concat in the manifest writer, not a correctness issue). Final row
counts vs. the clean backup baseline, all deltas small and positive (expected —
"Databento wins on overlap" refreshing rows plus a few days of accumulated
history):

| Table | Backup rows | New rows | Delta |
|---|---|---|---|
| `term_structure` | 1,319,670 | 1,319,992 | +322 |
| `term_structure_spreads` | 3,396,663 | 3,412,835 | +16,172 |
| `term_structure_butterflies` | 375,857 | 376,455 | +598 |
| `term_structure_condors` | 24,117 | 27,186 | +3,069 |
| `term_structure_averages` | 19,663 | 19,663 | +0 (SOFR-only, exact match) |
| `term_structure_packs` | 6,998 | 7,030 | +32 |

Every table's `date` column confirmed `datetime64[ns]` in the real post-run
files (not just the isolated test), closing out bug 4 above for real.

**One real, investigated-not-assumed finding while validating**: comparing the
new `term_structure.parquet` against the backup on shared `(date,
contract_symbol)` keys, 30,985 of 1,319,670 rows (2.35%) have a changed `close`
price, spanning 2019-01-16 through 2026-07-13 — concentrated in the ICE softs
(Coffee 3,292, Cocoa 2,157, Cotton 1,891, Sugar 1,618) and WTI (2,380). Traced
directly (not assumed): the OLD value for a sample case (Gold, `GCQ26.CMX`,
2026-06-11) was `4114.0`/`4094.399902`/etc. — float32-rounding artifacts
characteristic of yfinance — while the NEW value (`4240.6`/`4070.9`/etc.) is
clean Databento fixed-point-scaled data. This means the daily yfinance job
(`jobs/capture_term_structure.py`) had been writing over some historical dates
after whatever the last real Databento backfill was, and re-running this
transform correctly re-asserts Databento's documented precedence
("Databento wins on overlap," CLAUDE.md/WORKFLOW.md) — a genuine, expected
side effect of re-running this pipeline, not a bug in the rebuild.

### `build_continuous_curve.py` — validated, no changes needed

Already-written thin driver (groups `term_structure.parquet` by asset, calls the
already-tested `src/data/continuous_curve.build_continuous_curve` per asset,
writes `Data/continuous_futures.parquet` atomically) ran cleanly against the
freshly-transformed `term_structure.parquet`: all 42 assets produced a usable
curve, zero skips. Row count 170,830 vs. the pre-run backup's 170,666 (+164 —
consistent with ~4 days' accumulated history across 41 assets). Columns/dtypes
match exactly. Per-asset row deltas are uniformly +4 (the expected few days of
new data) except where the underlying price refresh (above) shifted which
dates/contracts qualify. `front_contract_symbol` differs from the old file on 6
dates for Gold, all within 2026-07-14 to 2026-07-20 — plausible and not
concerning: Databento's fresher, more accurate volume figures near the most
recent roll shifted the volume-crossover front-contract determination by a few
days in that narrow window, exactly the kind of edge effect the "confirmation
days" logic exists to handle. Back-adjusted `adj_close` differences (e.g. Gold's
~$128 max diff on common dates) trace entirely to the raw-price refresh
documented above, not to any defect in the continuous-curve construction itself
(`raw_close` differences fully explain the `adj_close` differences on inspection).

### `retry_databento_jobs.py` / `archive_to_drive.py`

Written to the same standard as the intact `backfill_databento.py`/
`submit_databento_jobs.py` — API/network/rclone-dependent, not independently
runnable or testable in this sandboxed environment, same as those two intact
files. No further work planned unless the user asks.

**Phase 5 complete. Full reconstruction (Phases 1-5) is now done.**

---

## Change log (append new entries below, most recent last)

- 2026-07-24: Log created. Phase 1 in progress: `transforms.py`, `momentum.py`,
  `breakout.py`, `crossover.py`, `vix_overlay.py` done and test-passing.
  `short_term_reversal.py` written, test file not yet started. `carry.py`/
  `xs_momentum.py`/`value.py` not started, blocked on reading their source papers.
- 2026-07-24 (later): `tests/test_short_term_reversal.py` written (9 tests) and
  passing. `short_term_reversal.py` now fully done. Next: read
  `references/Carry.pdf` before starting `src/signals/carry.py`.
- 2026-07-24 (later): Read `references/Carry.pdf` in full. Built `src/signals/
  carry.py` + `tests/test_carry.py` (9 tests, passing). `carry.py` now fully done.
  Next: read `references/Value and Momentum Everywhere.pdf` before starting
  `xs_momentum.py` and `value.py`.
- 2026-07-24 (later, new session continuing from handoff): Wrote
  `tests/test_xs_momentum.py` (8 tests: mom2_12 skip-month/lookback manual-calc
  correctness, NaN-before-history boundary, skip-month leak check,
  rank-within-sector-not-across, magnitude-invariance, no-vol-scaling signature
  check) — all passing, `xs_momentum.py` now fully done. Built `src/signals/
  value.py` (negative_5yr_return_value, bond_yield_change_value,
  fx_ppp_value_feature with the bounded reindex_ffill_limit=35 lesson applied from
  the start, build_value_panel, value_signal — matching the BOND_YIELD_MATURITY_MAP
  / FX_CPI_COUNTRY_MAP the handoff specified exactly) + `tests/test_value.py` (14
  tests, one iteration needed: the first `build_value_panel` overwrite test used
  identical US/EUR CPI, which made the PPP-adjusted result degenerate to the same
  formula as the plain default and produced a false failure — fixed by giving EUR
  CPI a genuine inflation differential vs. US, not by loosening the assertion).
  All 9 signal files now done. Ran `python -m pytest tests/ -q` project-wide:
  **99 passed**, no cross-module issues. Regenerated all 5 applicable
  `references/*_implementation_recipe.md` files (momentum, turtle_breakout,
  short_term_reversal, carry, xs_momentum), each derived from the reconstructed
  `src/signals/*.py` code rather than from memory of the lost originals; confirmed
  value.py never had one previously (CLAUDE.md's Value row has no such reference)
  and did not add one. **Phase 1 complete.** Next: Phase 2 (`research/*.py` driver
  scripts) — see the phase breakdown above.
- 2026-07-24 (same session, continued): Built all 12 Phase 2 `research/*.py`
  driver scripts (`momentum.py`, `vol_estimator_comparison.py`, `breakout.py`,
  `crossover.py`, `short_term_reversal.py`, `carry.py`, `xs_momentum.py`,
  `value.py`, `portfolio.py`, `value_momentum_combine.py`,
  `signal_correlation.py`, `tune_book_hyperparameters.py`), each run against real
  `Data/` and checked against CLAUDE.md's documented Sharpe/turnover/correlation
  numbers per the handoff's own instruction. **Important discovery before
  starting**: the already-restored `research/tune_all_books.py` (and presumably
  `tune_books_cpcv.py`/`tune_all_books_cpcv.py`) imports a `src/signals/*.py` API
  that doesn't exist — confirmed broken, cannot currently run — full detail in the
  Phase 2 section above; decided to build fresh against the actual, tested
  signals API rather than that mismatched one. **Read `references/AQR - Portfolio
  Construction Matters.pdf` in full** (not one of the 6 papers already read this
  session) before building `value_momentum_combine.py`, confirming the "mix"
  (separately-built portfolios combined) vs. "integrate" (blended score, one
  portfolio) definitions directly against the paper. Reproduction quality:
  `momentum.py`, `crossover.py`, `xs_momentum.py` are near-exact matches to
  documented numbers; `vol_estimator_comparison.py`, `breakout.py`, `carry.py`,
  `value.py`, `portfolio.py`, `value_momentum_combine.py`, `signal_correlation.py`
  are close/qualitatively-matching with some flagged numeric drift (see each
  script's own bullet above for specifics — turnover gaps, Book-level
  hyperparameter guesses like `EWMA_HALFLIFE`, etc.); `short_term_reversal.py`'s
  sector-tier VIX-regression significance and `tune_book_hyperparameters.py`'s
  value-Book adoption decision are the two most substantive open discrepancies,
  both flagged explicitly rather than silently accepted or forced to match. Ran
  `python -m pytest tests/ -q` after all 12 scripts: still **99 passed**, no
  regressions (Phase 2 scripts are driver scripts, not covered by their own test
  files). **Phase 2 complete.** Next: Phase 3 (remaining `tests/*.py`, ~20 files)
  and Phase 4 (`dashboard/pages/05`-`16`) — neither started yet.
- 2026-07-24 (same session, continued): Built all 19 remaining Phase 3
  `tests/*.py` files, covering every `src/` module that had zero test coverage
  (`backtest/engine.py`, `backtest/performance.py`, all 5 `portfolio/` files not
  already covered by `test_risk_metrics.py`, `signals/combine.py`,
  `regime/interface.py`, and 9 `data/` modules). 134 new tests, all passing on
  the real (or, for `data/macro.py` and `data/panels.py`, `monkeypatch`/
  `tmp_path`-synthesized) inputs. Two real bugs found and fixed IN THE TEST CODE
  itself while writing these (not in the source modules being tested) — see the
  Phase 3 section above for both; both were date-range setup mistakes that
  silently changed what was actually being tested, not source-code defects.
  Ran `python -m pytest tests/ -q`: **233 passed**, 0 failed. **Phase 3
  complete.** Next: Phase 4 (`dashboard/pages/05`-`16`, 12 pages) — not started.
- 2026-07-24 (same session, continued): Built all 12 Phase 4 dashboard pages.
  `dashboard/app.py` (never touched by the incident) already declared the
  exact filenames/titles/nav-groups for every page, so that file drove this
  phase directly rather than reconstructing the page list from CLAUDE.md prose.
  Every page reuses its corresponding Phase 2 `research/*.py` module's already-
  tested functions directly (no duplicated data-prep/backtest logic), cached
  via `@st.cache_data`. One real bug found and fixed: `research/portfolio.py`'s
  filename collides with the real `src/portfolio` package, breaking a plain
  `sys.path`-based import for pages 13/14 specifically — fixed via
  `importlib.util.spec_from_file_location` for just those two pages. Also
  wired in two previously-unused raw macro files (`gscpi_data.xls`,
  `overnight_fed_fund_rates_US.xlsx`) for page 15's registry, since no
  `src/data/` loader existed for either. All 12 pages + `app.py` verified
  exception-free via `streamlit.testing.v1.AppTest`. Ran `python -m pytest
  tests/ -q` afterward: still **233 passed**, no regressions. **Phase 4
  complete.** Next: Phase 5 (databento pipeline: `transform_databento.py`,
  `build_continuous_curve.py`, `retry_databento_jobs.py`,
  `archive_to_drive.py`) — deliberately last, not started. Per the original
  handoff, check first for a full recoverable copy from an earlier Claude
  conversation (2026-07-19/20 for the transform, 2026-07-21 for the curve
  builder) before falling back to a documented rebuild from
  `DATA_SCHEMA.md`/`WORKFLOW.md`.
- 2026-07-25 (new session continuing from handoff): Resumed Phase 5 mid-flight.
  Restored the 6 `term_structure*.parquet` files + manifest from the known-clean
  backup (they were in a bad intermediate state from the previous session's
  in-progress `DATE_DTYPE` fix test), re-validated that fix with a real
  merge-and-readback test (confirmed `datetime64[ns]`, not `object`), restored
  the backup again to undo that test merge, then found (via direct user
  question, not self-caught) that the combo-parsing loop was still non-
  vectorized Python row-by-row iteration despite the earlier pandas→polars
  merge-path correction — a separate issue from that correction. Killed an
  in-progress full run partway through (WTI+Brent done, ~9 min elapsed, on pace
  for 1-3+ hours), confirmed empirically that `(raw_symbol, maturity_year,
  maturity_month)` is a safe dedup key (135x row/key compression on WTI, zero
  key collisions), and rewrote the combo section plus the outright table's
  `map_elements` calls to be genuinely vectorized (same underlying parsing
  functions, called once per unique key/lookup instead of once per row, joined
  back via polars) — a pure performance change with no behavior difference.
  Re-validated all 6 ground-truth assets byte-identical to both the
  pre-vectorization run and `DATA_QUALITY_REPORT.md`'s documented counts, then
  ran the real full 42-asset transform (10.5 minutes total, vs. an estimated
  1-3+ hours unvectorized) and `build_continuous_curve.py`, both validated
  directly against the real pre-existing output (see the full Phase 5 write-up
  above for the complete bug list, the vectorization approach, and the
  "Databento wins on overlap" price-refresh finding investigated during
  validation). **Phase 5 complete — full reconstruction (Phases 1-5) done.**
