# Workflow / Roadmap

Living roadmap for the Systematic Commodity & Macro Futures Research Platform. This
supersedes `project-review.html` as the source of truth for what's planned — that file
is a dated snapshot (2026-07-08), this file is the one to update as work lands. See
`CLAUDE.md` for the hard rules and current-state summary this roadmap assumes.

Status tags: 🟢 done · 🟡 partial/in progress · ⚪ not started · 🔴 blocked

---

## 0. Sequencing principle

Ordered for a solo researcher who wants one honest, complete, end-to-end pipeline before
a wide signal library. Within the near-term signal scope (momentum, breakout, crossover,
classic cointegration spreads — see `CLAUDE.md`), methodology hygiene comes first,
because every phase after it compounds whatever discipline is or isn't in place. Full
portfolio construction (vol targeting, optimizer, Book/Allocator) is deliberately
sequenced after 2+ signal families exist, not before — there's no value in building an
optimizer for one signal.

---

## Phase 0 — Data engineering 🟢

Core ingestion pipeline is done. The continuous-curve construction (below) is also
now built — `Data/continuous_futures.parquet`, all 42 assets — and, as of 2026-07-21,
consumed by a real signal (`research/momentum.py`, Phase 2a). Two real
data-quality items are logged against the core panel, not blocking but worth reading
before trusting it without qualification — full detail in `DATA_SCHEMA.md` section 1:
a 2026-07-14 low-confidence theoretical caveat (raw-splice roll discontinuity, never
confirmed either way), and a 2026-07-21 **confirmed** finding (3.84% of core-panel
cells fail basic OHLC consistency, a separate mechanism from the splice caveat) that
led to the trusted-era/legacy-era split described below.

**Trusted-era / legacy-era split decided 2026-07-21, not built yet beyond the
reference file.** Prompted by evaluating whether the dataset built this session is
"flawless" (it isn't) and by deciding how to weigh 16 years of clean, Databento-backed
history against 26 years of `yfinance`-only history with confirmed noise. Full
rationale, numbers, and the rejected alternative (blanket deletion) are in
`DATA_SCHEMA.md` section 1 — summary: **don't delete either dataset.**
`Data/asset_trusted_since.csv` (new) gives the real, per-asset date from which
`term_structure.parquet` has genuine Databento coverage (2010-06-06/07 for 33 of 42
assets; later, documented exceptions for KC_Wheat, Russell2000, SOFR, Lumber, and the
5 ICE softs). That date forward is the trusted basis for signal calibration going
forward; everything before it in the core `yfinance` panel is kept but demoted to an
out-of-sample robustness check only, never primary calibration. Real cost: the
existing train/held-out split (`CLAUDE.md` Rule 1, train ≤2019/held-out 2020+) was
built against the full pre-2010 history and shrinks to as little as a 9-year train
window under this split. **Update, same day: this was re-decided, not left open** —
see Phase 2a below for the resulting three-way train/validation/test split
(validation 2020-2021, test 2022+) and the production `src/` architecture migration
that now implements it in `research/momentum.py`.
**Built 2026-07-21** — `src/data/continuous_curve.py` (pure functions, unit-tested
against synthetic data in `tests/test_continuous_curve.py`) + `databento/
build_continuous_curve.py` (driver, loops all 42 assets, writes `Data/
continuous_futures.parquet`). Prompted by finding the hard way (see Phase 2a's
momentum re-run) that trusted-era masking wasn't a fix for the core panel, just a
smaller dose of the same problem: the post-`trusted_since` portion of `close.parquet`
etc. still had a 0.53% OHLC-violation rate (checked directly — 755 of 143,495 cells),
still Yahoo's raw, unaudited splice.

**Scope — general-purpose infrastructure, not momentum-specific.** Anything that
trades a single per-asset price level (momentum, breakout, crossover) needs this;
classic RV/cointegration spreads and carry don't, since they already consume
`term_structure.parquet`'s real multi-contract data directly and were never blocked
on this. One dedicated construction script, built once, consumed by every
price-level-based signal family going forward — not re-derived per signal.

**Roll rule** (per asset, from `term_structure.parquet`'s real contract-level data):
- **Target method, once available: open-interest crossover** — roll when the next
  contract's OI overtakes the front's. This is the actual industry-standard rule and
  the one to switch to once OI is purchased (Databento's `statistics` schema,
  ~$50-100 estimated, already logged as a known gap in `DATA_SCHEMA.md` §1 — not
  bought yet).
- **Interim method: volume crossover**, using the real per-contract volume already in
  `term_structure.parquet` today. Roll when the next contract's daily volume exceeds
  the front's — the standard fallback when OI isn't available, and close to as good
  in practice (some practitioners prefer it for reacting same-day rather than on a
  one-day-lagged OI print).
- **Backstop: fixed days-before-expiry.** Layered under the volume-crossover rule for
  thin contracts where volume may never cleanly cross over before expiry — forces a
  roll regardless, so the series never rides a contract into its illiquid final days
  or (for physically-settled products) past first notice day.

**Adjustment method — build both, not one:**
- **Raw/unadjusted series** — real prices, no correction, for anything that needs an
  actually-tradeable historical reference: TCA/slippage work (Phase 6), the
  backtest-vs-live comparison log (Phase 10d), dashboard curve display, and as the
  ground truth to validate the roll-detection and adjustment logic against.
- **Back-adjusted series (ratio/proportional method — corrected 2026-07-21, see
  below; originally additive)**, for signal construction and backtesting
  specifically — historical prices scaled by the roll-date ratio so returns across
  a roll are clean, avoiding the fake return spike an unadjusted series creates at
  every roll (the mechanism behind the still-open roll-splice caveat above).

  **Correction (2026-07-21, found while building `research/momentum.py`,
  Phase 2a):** originally additive, on the reasoning that a ratio method requires
  dividing through a price and this dataset contains the real WTI April-2020
  negative print (verified real, not an artifact — `databento/
  DATA_QUALITY_REPORT.md` Asset 17, `CLK20.NYM`'s `close=-2.67` on 2020-04-20).
  That per-contract fact is correct, but the conclusion drawn from it wasn't checked
  against *this specific series* — the continuous curve's own roll rule (backstop:
  fixed days-before-expiry) rolls out of each contract before expiry, so by
  2020-04-20 the front-continuous series had already moved past `CLK20` onto a
  later contract that never went negative. Checked directly: `raw_close` is never
  non-positive for **any** of the 42 assets, including WTI (min 12.26) — the
  front-contract series simply never captures that per-contract event. Meanwhile
  additive turned out to have a real, live bug of its own: it pushed HeatingOil,
  RBOB, Brent, WTI Crude, and Oats through zero in old, heavily-shifted segments
  (250-3663 rows each with non-positive `adj_close`) — worse than "can go
  negative," a percentage-return singularity right at the crossing (`pct_change()`
  on RBOB hit -26,740% on one day, corrupting every downstream return-based
  calculation for that asset, not just display). Ratio adjustment is a product of
  positive numbers, so it's always positive by construction — verified 0 rows with
  non-positive `adj_close` anywhere after rebuilding — and has a second,
  independent benefit: it preserves percentage returns consistently across the
  whole history, which additive technically doesn't in old, heavily-shifted
  segments even where they stay positive. `tests/test_continuous_curve.py`'s
  back-adjustment test rewritten to ratio semantics; `dashboard/pages/
  05_continuous_curve.py`'s copy updated to match.

**Two real algorithmic bugs found and fixed by testing against the actual 42-asset
dataset, not just synthetic fixtures — worth knowing about if this ever gets
re-derived.** The synthetic unit tests all passed on the first attempt; running
against real data still surfaced two real gaps the synthetic cases didn't cover:

1. **"Next contract" can't always mean the immediately-adjacent chain entry.**
   MexicanPeso got permanently stuck on `6MZ10.CME` for 96.9% of its entire 16-year
   history — its immediate chain successor, `6MG11.CME`, turned out to have only 2
   real rows total (a thin serial-month FX contract with a brief early print and
   nothing since), so neither leg ever had an overlapping row to compare or roll
   into again, and the algorithm had no way to look further down the chain. Fixed by
   scanning forward from the front's position for the nearest chain entry that
   actually has a real row on the current date, rather than assuming `chain[idx+1]`
   is always viable. Also fixed the same way: Palladium (was 96.3% stuck), Platinum
   (82.8%), and partially the 5 ICE softs.
2. **Day-1 initialization could pick a contract that hadn't started trading yet.**
   `fillna(0.0)` before `idxmax()` on the first date made "not yet listed" tie with
   "a genuine zero-volume print," and the tie-break could land on a contract with no
   real history for months. Coffee's very first date (2023-08-01) initialized to
   `KCH25.NYB`, a March-2025 contract that doesn't actually start trading until
   2024-07-15 — 244 consecutive gap days, 32.6% of Coffee's entire curve, purely from
   a bad first pick. Fixed with `dropna()` before `idxmax()`, restricting the choice
   to contracts with a real row on day one. This alone fixed all 4 remaining
   ICE softs from ~27-33% gap rates down to under 0.5%.

**Residual, small, and honestly left as `NaN` rather than fabricated**: 1,085 of
170,666 total rows (0.6%) still have no valid raw price — checked directly, these are
scattered across dozens of different contracts and dates (Platinum alone has gaps
across 83 different front contracts, 1-2 days at a time, including recognizable
holiday dates like 2010-07-04 and 2012-07-04), consistent with genuine isolated
missing prints in the underlying feed, not a further algorithmic pattern. Left as
`NaN`, not interpolated or forward-filled — a signal built on this data will need to
handle that the same way every other gap in this project's pipelines already is.

**Spot-checked, not just trusted**: `raw_close` for sampled dates across WTI, Lumber,
and Coffee matches `term_structure.parquet`'s own quoted close for whatever contract
was assigned front that day, exactly, in every sample checked. WTI shows 198 rolls
over its 16-year history (~monthly), matching its real listing cycle. Adjusted-price
jumps at sampled roll dates are small, sane day-to-day moves, not the raw mechanical
gap - back-adjustment is doing its job.

**Not done yet, deliberately** (per the plan's own scope): `research/momentum.py`
has **not** been re-pointed at this new curve — that's the "take it from there" next
step, not part of this build. Also not done: promoting `build_continuous_curve.py` to
a scheduled `jobs/` script (it needs to re-run periodically to stay current as
`capture_term_structure.py` keeps appending new contract data daily) — deferred until
the output has actually been validated in real use.

- Core futures OHLCV pipeline (`get_data.ipynb` / `jobs/update_data.py`, `yfinance`,
  `period="max"`) → `open/high/low/close/adj_close/volume.parquet` + `metadata.csv` +
  `audit.csv` in `Data/`. Expanded 2026-07-20 to 41 of the project's 42-asset universe —
  SwissFranc (`6S=F`), MexicanPeso (`6M=F`), and Lumber (`LBR=F`) added (all three
  confirmed with full real history, live-tested, not assumed), matching their
  2026-07-17 addition to the term-structure universe (Phase 4). **SOFR is permanently
  excluded, not pending**: no usable Yahoo continuous ticker exists (`SR3=F`/`SR1=F`
  both resolve but return only 1 row, checked 2026-07-20) — it still has real
  term-structure/carry coverage (Phase 4), just not a place in this core signal-research
  panel. `jobs/update_volatility.py` picked up all 3 new assets automatically on its
  next run (pure function of the OHLC parquets); `jobs/update_dashboard_summary.py`'s
  `ASSET_CALENDAR` was updated in the same pass (SwissFranc/MexicanPeso → `CME_FX`,
  Lumber → `CME_Agriculture`, the closest real fit since no lumber-specific calendar
  exists in `pandas_market_calendars`) so the QA dashboard doesn't show them as
  `UNMAPPED`.
- Yang-Zhang multi-horizon volatility (`volatility.ipynb`, 21/63/126/252d) →
  `yang_zhang_features.parquet`.

Full schema and coverage detail in `DATA_SCHEMA.md`.

---

## Phase 1 — Research hygiene 🟢 done (2026-07-14)

Fix the two live methodology violations (`CLAUDE.md` Rules 1 and 2) and stop notebook
sprawl before building anything new on top of it.

- 🟢 Extracted signal construction, vol-scaling, and the backtest function into
  `src/signal_lib.py` (2026-07-14) — one `backtest_signal` (explicit `returns_data`
  param, no notebook-global dependency), replacing the two slightly different inline
  definitions. `feature_engineering.ipynb` now imports from it; re-executed end-to-end
  with zero errors and numerically verified identical to the pre-refactor logic.
- 🟢 Replaced the post-hoc energy/Natural-Gas universe edit (2026-07-14) with an
  ADV liquidity floor (≥1,000 contracts/day, trailing 2yr) + a train/held-out split
  (train ≤2019, held-out 2020+, chosen before looking at performance). Confirms Natural
  Gas/Brent are nowhere near illiquid — Energy stays in the universe by rule. Removed
  the violating cells from `feature_engineering.ipynb` (preserved verbatim in
  `deprecated/Scripts/feature_engineering_removed_energy_cells_2026-07-14.py`). Also
  caught and fixed a second, related issue while doing this: the notebook's "best
  signal" pick (`mom_252_skip`) was hardcoded and, under the corrected train-only view,
  is actually negative Sharpe in-sample — `mom_126_skip` is the real train-period winner
  now derived dynamically rather than assumed. Notebook re-executed end-to-end with zero
  errors after every change.
- 🟢 Rebuilt `Time_Series_Models.ipynb` (2026-07-14) — ADV-filtered 33-asset universe,
  rolling-window (not full-sample) Engle-Granger cointegration, tested on Corn/Wheat,
  Gold/Silver, Brent/WTI. Re-validating the legacy notebook's Corn/Wheat claim
  (full-sample p≈0.0003) properly **did not confirm it as a static fact**: cointegrated
  in only 33% of rolling 3-year windows, i.e. regime-dependent, not structural — exactly
  the distinction a full-sample test can't make. Brent/WTI is the most consistently
  cointegrated pair tested (45% of windows) and the strongest Phase 2d candidate of the
  three; Gold/Silver the weakest (20%) despite being "test first" in the original
  candidate table. Legacy notebook preserved at
  `deprecated/Scripts/Time_Series_Models_legacy.ipynb`.
- 🟢 Repo cleanup done 2026-07-14 — see `CLAUDE.md`'s repo-hygiene section and
  `deprecated/README.md`.

---

## Phase 2 — Signal library: the four in-scope families ⚪

Build/finish these four, in this order, each held to the Phase 1 methodology bar
(continuous + vol-scaled from the start, per `CLAUDE.md` Rule 5; rolling windows only,
per Rule 2). No portfolio optimizer yet — see Phase 5.

### 2a. Time-series momentum — done, rebuilt to match the paper exactly, final 🟢 (2026-07-21)

Rebuilt from scratch to match Moskowitz-Ooi-Pedersen (2012) exactly rather than the
previous binary/continuous/rank transform comparison. `references/
momentum_implementation_recipe.md` has the full recipe (signal formula, vol
estimator, position sizing, rebalancing, straight from the paper) and an explicit
diff against what the codebase had before — biggest gap found: neither `binary` nor
`continuous_signal` implemented the paper's actual position-sizing rule (sign-of-
momentum direction, sized to a constant per-instrument vol target). Recipe:
`TARGET_VOL(0.40) × sign(mom) / vol`, headline spec `k=12mo, h=1mo`, no skip month
(the skip convention already in `build_momentum_features` is a cross-sectional-
momentum import, not part of TSMOM, deliberately unused here). Full 8×8
lookback×holding grid (paper's Table 2, k and h both in {1,3,6,9,12,24,36,48} months)
reproduced as a **train-period, descriptive/robustness view — not a spec-picker**:
the headline spec is fixed a priori at the paper's own (12,1), never chosen by
looking at which grid cell scores highest here, the same look-ahead discipline as
`CLAUDE.md` Rule 1/2. Evaluation order matches the paper's own: per-asset Sharpe
first (Figure 2 style — does this work broadly, for how many assets), *then* the
naive equal-weighted diversified-portfolio backtest as the explicitly-labeled interim
combination step (real portfolio construction is a separate, later phase).

Two vol estimators built and compared, per direct request: `src/data/volatility.py`
(Yang-Zhang, ported from `volatility.ipynb`) and `src/data/ewma_volatility.py` (new,
matches the paper's own Eq. 1 exactly — EWMA of squared daily returns, `com=60`).
Both decoupled from the momentum lookback (one fixed-horizon vol estimate regardless
of which `k` is tested, matching the paper and correcting the old codebase's implicit
1:1 coupling between vol horizon and momentum lookback).

**Final result: train Sharpe 0.239, validation 0.475, test 0.402 — all positive, no
NaN, no data-snooping.** Yang-Zhang wins the vol-estimator comparison on train
evidence (0.239 vs. EWMA's 0.200). Supersedes the previous two-transform provisional
result entirely; `feature_engineering.ipynb` still holds that original result as
historical record, not overwritten.

**Real bugs found and fixed along the way, logged plainly (including one wrong
hypothesis), because getting to that final number took four separate, real fixes:**

1. **Yang-Zhang vs. back-adjusted prices.** Computing Yang-Zhang directly against
   `load_continuous_backadjusted()`'s OHLC (the obvious first choice, matching
   momentum's own signal source) pushed most of the universe to 70-100% NaN —
   the initial hypothesis was that back-adjustment structurally pushes
   backwardation-prone commodities (grains, meats, softs) non-positive, per Keynes'
   "normal backwardation." **Checked directly against the data and found false** —
   queried `adj_close <= 0` per asset and it was zero for every one of those assets.
   The real cause (see item 4) was the back-adjustment method itself, affecting a
   *different* group of assets (energy). Logging the wrong hypothesis here
   deliberately, not just the fix that followed it.
2. **Yang-Zhang vs. scattered panel gaps.** Moved Yang-Zhang to the raw curve
   (`load_continuous_raw()`, with `is_roll_date` masking the roll-date overnight
   artifact) — NaN dropped for most assets but stayed near-100% for Corn/Soybeans/
   Wheat/the meats. Root cause: the 42-asset panel's date index is the union of every
   asset's own trading calendar, so any asset with a sparser calendar than the panel
   (Corn: 849 of 5015 panel dates aren't real CBOT trading days, 779 separate 1-2 day
   gaps, not one missing block) shows up as scattered NaN. A default
   `rolling(window).var()` (min_periods=window) nulls almost every window once gaps
   are this dense. Fixed with `min_frac` (tolerate a fraction of the window missing,
   default 0.7).
3. **The overnight term needs its own tolerance.** Even after fixing 2, Cocoa stayed
   100% NaN *within its own well-populated recent history* — traced to the overnight
   component (`r_o`) being structurally sparser than the other two (it needs both
   today's open AND yesterday's close valid, roughly double the missingness, plus
   roll-masking on top): at Cocoa's most recent date, `r_c`/`rs` had 49/63 valid
   values (fine) but `r_o` had only 34/63 (NaN under a shared 44-value min_periods),
   nulling the whole sum through simple addition. Fixed with `overnight_min_frac`,
   a separate, more lenient tolerance (default 0.3) for that one component.
4. **The real back-adjustment bug** (see Phase 0 above) — additive back-adjustment
   pushed HeatingOil/RBOB/Brent/WTI Crude/Oats through zero, corrupting percentage
   returns directly (RBOB hit a -26,740% one-day `pct_change()`), not just Yang-Zhang.
   Corrected the curve construction itself to ratio adjustment, not a downstream
   patch — see Phase 0's change log for the full reasoning.
5. **A `max()`-over-possibly-NaN vol-estimator selection bug**, found alongside item
   4 (RBOB's corrupted Sharpe had poisoned the pooled `yang_zhang` result to NaN, and
   Python's `max()` silently picked it anyway since NaN comparisons are always
   False). Fixed to compare only among non-NaN candidates.
6. **The paper's 40% constant is Sharpe-invariant but not leverage-invariant** — a
   correction to this session's own earlier reasoning. Dropping it (on the argument
   that Sharpe ratios are scale-invariant) is true for the *pooled*, gross-exposure-
   normalized backtest, where it cancels — but not for the per-asset (unnormalized)
   diagnostic view, where it directly sets leverage. Without it, SwissFranc (~9%
   annualized vol) got routine 11-26x leverage, and one real 19.4% daily FX move
   against that implied a mathematically impossible "-494% daily return" (more than
   total loss), corrupting its per-asset Sharpe. Fixed by reintroducing the constant
   as an explicit `target_vol` parameter on `vol_targeted_sign_signal` (neutral
   default 1.0, since it's a generic transform reused across signal families;
   `research/momentum.py` passes the paper's own 0.40). One residual, understood,
   low-priority edge case remains: SwissFranc's EWMA-vol per-asset Sharpe is still
   NaN even at 0.40 target (EWMA's vol estimate dips low enough at one point that
   ~9.9x leverage against an adverse month still crosses the same threshold) — a
   known limitation of the per-asset diagnostic specifically (no hard leverage
   floor), not the core signal or the pooled/grid results (unaffected, since they're
   gross-exposure-normalized).

**Also done as part of this rebuild** (agreed 2026-07-21, all three now complete):
the dashboard's new continuous-curve page (`dashboard/pages/05_continuous_curve.py` —
asset/date-range/frequency toggles, raw vs. back-adjusted with roll dates marked,
plain-language roll-rule and adjustment-method explainer); reading
`references/Time Series Momentum.pdf` and writing the implementation recipe
(`references/momentum_implementation_recipe.md`); and this rebuild itself.

**Next up, agreed 2026-07-21 (dashboard follow-on) — three tasks, none started yet:**

1. **Fix page 05's stale display.** The underlying code and data are already
   correct (ratio adjustment, zero negative `adj_close` verified) — what's showing
   the old additive/Panama copy and negative-price chart is a stale dashboard
   session (`@st.cache_data(ttl=1200)` serving a cached read from before the
   rebuild, or the Streamlit process just needs a hard restart, not a browser
   refresh). Also add a plain-language paragraph explaining what a roll is, why it
   matters, and this project's methodology (volume crossover + fixed-days-before-
   expiry backstop) — a real content gap, not part of the stale-display issue.
2. **New Volatility dashboard page.** Toggle between Yang-Zhang and EWMA
   (`src/data/volatility.py` / `src/data/ewma_volatility.py`, computed off the
   corrected `continuous_futures.parquet` — Yang-Zhang off the raw curve with
   `is_roll_date` masking, EWMA off the back-adjusted curve's returns — not the
   legacy `yang_zhang_features.parquet`, which is still on the old yfinance panel
   and only has Yang-Zhang). Multi-asset overlay charting. Ranking of assets by
   volatility, **default basis = current (latest) value**, with a lookback
   selector (today / trailing 21d / trailing 252d / full-history average) to
   switch views — decided over "one fixed average" since "what's risky right now"
   and "what's structurally high-vol" are different, both useful questions.
3. **New "Strategy Performance" dashboard subcategory, starting with a Momentum
   page** (same nav-grouping pattern as today's "Monitoring"/"Coverage" — see
   `dashboard/app.py`). One page per sleeve going forward, as each signal family
   is finished and added. Momentum page scope, decided in discussion:
   - Methodology write-up (how it was built, decisions made, reference to
     Moskowitz-Ooi-Pedersen 2012) — adapt from `references/
     momentum_implementation_recipe.md` and this phase's narrative above, same
     style as page 05's "Why Both Series Exist" section.
   - Per-asset toggle + simultaneous multi-asset charting of performance.
   - Metric set: Ann Return, Ann Vol, Sharpe, Sortino, Calmar (return/maxDD), Max
     DD, win rate — a standard CTA tearsheet, not sprawling. Sortino/Calmar/win
     rate aren't in `backtest/performance.py` yet, so that module likely needs
     extending (or the page computes them itself — decide when building).
   - **Show train/validation/test explicitly, don't blend them** — and rank
     assets by the **test**-period value of whichever metric is selected, not a
     blended or in-sample number, to avoid the dashboard presenting an
     in-sample-flattering view as forward evidence (the same discipline this
     project already applies to signal research itself).
   - **Scope to the headline `k=12, h=1` spec only** — the full 8×8 grid is a
     research artifact (`Data/research/momentum/grid_sharpe_*.csv/png`), not
     something to re-expose as dashboard controls; this page is "the sleeve we'd
     actually trade," not every combination tested.
   - Vol-estimator toggle (Yang-Zhang vs. EWMA), mirroring the volatility page.
   - **Computed live at render time**, same pattern as pages 01/05 (read
     `continuous_futures.parquet`, rebuild the signal/backtest via existing
     `src/` functions) — not a new precompute job. Chosen for always reflecting
     current signal code with nothing to keep in sync; `research/momentum.py`'s
     own run time suggests this should be comfortably fast for a page load, but
     that's worth confirming empirically once building starts, not assumed.

### 2b. Breakout (Donchian channel) — building now 🟡

**Economic intuition:** commodities trade in tight consolidation ranges until a
structural supply/demand imbalance forces a re-pricing. A breakout above/below the recent
range reacts to that shift the moment a new extreme prints, rather than waiting for a
sustained move to accumulate the way momentum's 12-month average does — same underlying
"trends persist" belief as momentum, engaging much earlier in a new trend's life.
Standard component of institutional trend books since the Turtle Traders (Dennis/Parker).

**Design finalized 2026-07-21** (discussion + decision log; full mechanics in
`references/turtle_breakout_implementation_recipe.md`, playing the same role
`momentum_implementation_recipe.md` played for momentum — no academic paper anchors this
signal, its origin is trading practice, not a journal factor):

- **Authentic dual-channel Turtle systems, not the single symmetric channel originally
  sketched here** — a longer entry channel, a shorter exit channel, exactly the two
  historical systems (System 1: 20-day entry / 10-day exit; System 2: 55-day entry /
  20-day exit). No invented third horizon to match momentum's lookback set — that would
  undercut the authenticity the dual-channel design is preserving. Both reported as
  parallel specs (Turtles traded both simultaneously as a blended book historically;
  no "headline pick" needed or wanted).
- Genuinely path-dependent state machine (long/flat/short; which threshold applies
  depends on the position currently held) — implemented as a per-asset walk-forward,
  same pattern as `continuous_curve.py`'s `assign_front_contract`, not a single
  vectorized expression.
- **Channels built off the BACK-ADJUSTED curve**, the opposite choice from momentum's
  Yang-Zhang input (which needs the RAW curve) — a raw roll-date price jump could
  spuriously register as a "new N-day high," a hazard breakout is far more exposed to
  than momentum (whose 12-month average dilutes a single-day artifact).
- **Direction updates daily — non-negotiable.** Monthly formation (momentum's
  convention) would delay reacting to a breakout until month-end, destroying the entire
  reason to build this signal. Whether position *size* also needs daily re-derivation,
  or a coarser resize cadence cuts turnover without hurting Sharpe much, is measured
  (turnover + net-of-cost Sharpe via `backtest/costs.py`, built 2026-07-21) rather than
  assumed — starts daily/daily, decided from the actual numbers, not guessed upfront.
- Vol estimator (Yang-Zhang vs. EWMA) re-compared on train evidence per system, not
  assumed to inherit momentum's winner. Gross AND net-of-cost Sharpe reported from the
  start, since this signal is expected to be materially higher-turnover than monthly
  momentum.
- **Scope note:** "authentic Turtle" means the entry/exit price-trigger logic, not the
  original 1983 ATR-based money-management system — this project's existing
  vol-targeting (`target_vol=0.40`, same as momentum) supersedes that for sizing
  (CLAUDE.md Rule 7: re-derive mechanics per family, don't port a whole external system
  wholesale).

### 2c. Moving-average crossovers — build next ⚪

On `feature_engineering.ipynb`'s own feature wishlist (50/100/200-day) but never built.
A slower, smoother trend-confirmation signal than raw momentum or breakout — e.g. a fast
MA crossing above/below a slow MA, or price relative to a single long MA. Same per-asset
time-series structure as 2a/2b: build continuous (distance between MAs, or price-to-MA
distance, normalized by vol), not a binary cross flag, for the same reason as 2b.

### 2d. Classic cointegration / relative-value spreads — rebuild properly ⚪

Structurally different from 2a–2c: **convergent**, not divergent. Trend and breakout want
big directional moves and tend to be long convexity in a crisis (the well-documented
"crisis alpha" property of CTAs). RV spreads harvest a physically-anchored equilibrium and
produce steadier, carry-like returns in normal markets — but can suffer in the same
liquidation events that help trend, since physical-margin relationships can temporarily
blow through their normal range under forced selling. This is exactly why each spread
(or a combined RV book) should eventually be its own Book in the Phase 5 sense, with a
**rolling** — never full-sample — cointegration test to determine hedge ratios and
whether the trade is currently on, and a z-score entry/exit rule instead of continuous
momentum-style sizing.

Candidates, filtered to pairs with an actual physical relationship (not just a
statistically convenient correlation) and cheap to test against data already in hand:

| Spread | Legs & logic | Data status | Priority |
|---|---|---|---|
| **Gold / Silver ratio** | Both safe-haven precious metals, but silver carries real industrial-cycle exposure gold doesn't — ratio mean-reverts around that structural difference. | Have both (GC=F, SI=F) | **Rolling-window re-test (2026-07-14): weakest of the 3 candidates, cointegrated in only 20% of 3-year windows and not currently cointegrated (p=0.32).** Lowest priority now, despite the original "test first" label. |
| **Brent / WTI spread** | Global seaborne benchmark vs. US landlocked crude; spread driven by pipeline capacity and export-terminal bottlenecks, not sentiment. | Have both (BZ=F, CL=F) — **and, as of 2026-07-19, the real exchange-quoted spread itself: 66,291 CL-BZ rows in `Data/term_structure_spreads.parquet`, leg-decomposed, no back-differencing needed** (`databento/DATA_QUALITY_REPORT.md` Asset 17). | **Rolling-window re-test (2026-07-14): strongest of the 3 candidates, cointegrated in 45% of 3-year windows and currently cointegrated (p=0.036).** Highest priority for Phase 2d build-out — also the direct overlap point with the port congestion project's Track A/C |
| **3:2:1 Crack Spread** (WTI vs. RBOB vs. Heating Oil) | Refiner processing margin: 3 barrels crude → ~2 gasoline + 1 heating oil. Mean-reverts against physical refining operating costs. | Have all three (CL=F, RB=F, HO=F) — **and the real quoted 2-leg components: 24,065 RB-CL + 21,871 HO-CL rows in `Data/term_structure_spreads.parquet`** (2026-07-19, same source as above; the third leg needed for a true 3:2:1 weighting isn't a single quoted instrument and would still need constructing). | Test first |
| **Corn / Wheat spread** | Grain-complex substitution relationship (feed use, planting-acreage competition). | Have both — **and, as of 2026-07-20, the real exchange-quoted spread: 3,939 ZC-ZW rows in `Data/term_structure_spreads.parquet`, leg-decomposed, cross-checked against Wheat's own real outright universe with 0 mismatches** (`databento/DATA_QUALITY_REPORT.md` Assets 24-28). | **Rolling-window re-test (2026-07-14): the full-sample p≈0.0003 does not hold up as a static fact — cointegrated in only 33% of 3-year windows (currently cointegrated, p=0.036, but regime-dependent, not structural).** Middle priority of the 3 for Phase 2d. |
| **Soybean Crush** (Soybeans vs. Meal vs. Oil) | Processing margin for crushing raw soybeans into meal (feed) and oil (food/biofuel). Reported empirical half-life ~3 days — fast, high-turnover, needs its own execution cadence. | **Missing 2 tickers** — need Soybean Meal (`ZM=F`) and Soybean Oil (`ZL=F`) added to `get_data.ipynb` | Cheap to unblock, but not free |

---

## Phase 3 — Deferred / lower priority signals ⚪

Not in the current active scope (see `CLAUDE.md`), but on the roadmap:

- **Short-term reversal** — structurally cross-sectional/dollar-neutral like RV, not
  time-series like trend/breakout; needs its own book design. Lower priority than the
  four Phase 2 families.
- **Seasonality** — narrow, data-mining-prone; low priority.
- **Options VRP/skew, COT positioning, deep-learning EVaR portfolios** — out of scope.
  Sophisticated, but not what an institutional systematic manager would prioritize
  building first, and against the don't-add-complexity-for-its-own-sake principle behind
  this rebuild.
- **Curve curvature / butterfly richness** — candidate, not a task (logged 2026-07-15).
  The Databento definition/OHLCV exploration for carry (Phase 4) turned up
  exchange-listed butterfly instruments (three-legged, e.g. `LE:BF Z6-G7-J7` — long the
  two wing months, short the middle) alongside the calendar spreads Phase 4 uses for
  carry. A butterfly price is the market's direct quote on whether the belly of the
  curve is rich or cheap relative to the wings — a distinct signal family from anything
  currently in scope (not time-series-per-asset like momentum/breakout, not a two-leg
  RV pair like Phase 2d). Genuinely interesting, not currently needed by any in-scope
  signal — same treatment as the Feature Engineering Primer's idea list: a menu entry
  to revisit later, not a to-do. If ever built, needs the same decade-disambiguation
  fix as outrights/spreads (Phase 4) — the symbol string alone is not a safe contract
  key.

---

## Phase 4 — Carry 🟢 built 2026-07-22, honest weak/negative result

Carry (roll yield) needs the price of at least two contract months on the same
underlying at the same time:

```
Carry = (F_near − F_deferred) / F_near × 365 / (days to F_deferred)
```

The continuous, auto-rolled front-month series (`CL=F` etc.) used everywhere else in this
project shows only *one* price per day with the roll already stitched in — no way to
recover the deferred-contract price from it. This was originally logged as a hard block
(same honest reason the retired stat-arb project removed its own carry signal rather than
fake it with a residual-return proxy, `CLAUDE.md` Rule 4).

**Update — a free source was found and live-verified (not just researched):**
`yfinance` can pull individual contract-month prices directly, using the format
`{ROOT}{MONTH_CODE}{YY}.{EXCHANGE}` — e.g. `CLQ26.NYM` (WTI, August 2026, NYMEX). This is
**undocumented/unofficial behavior** (not in yfinance's public API surface — discovered by
testing, not by reading docs), but it was verified two ways:

- **Multiple simultaneous contract months work and show a real term structure.**
  `CLQ26.NYM` / `CLU26.NYM` / `CLZ26.NYM` / `CLF27.NYM` returned distinct closes
  (71.41 → 71.34 → 70.62 → 70.34 as of the test date) — a clean declining curve, exactly
  what the carry formula needs.
- **All 38 assets in the universe have a working individual-contract ticker.** Every
  asset was tested with its correct exchange suffix (`NYM`=NYMEX, `CMX`=COMEX,
  `CBT`=CBOT, `NYB`=ICE/NYBOT, `CME`=CME); all 38/38 resolved to real data. Two needed a
  different month code than the default tried (`CTZ26.NYB` for Cotton, `LEZ26.CME` for
  Live Cattle — not every product has the same listing cycle, e.g. livestock/some softs
  don't list a September contract) — a reminder that contract month codes have to be
  matched to each product's actual CME/ICE listing calendar, not assumed uniform.

**Update (2026-07-14) — both open caveats above have now been checked, and one is
disqualifying for backtesting:**

- **A contract's ticker stops resolving entirely once it expires.** Tested three WTI
  contracts that expired within the prior ~6 weeks (`CLK26.NYM`, `CLM26.NYM`,
  `CLN26.NYM`) — all three 404 ("possibly delisted"), vs. the not-yet-expired `CLQ26.NYM`
  which still resolves. **There is no way to retroactively pull an expired contract's
  history through this method.** This means the free path can only ever support a
  *forward-capture* archive (start recording today, build up history day by day) — it
  cannot backtest carry historically, contrary to how this section previously read.
- **The 2,129-row history for `CLQ26.NYM` back to 2018 is not usable, real data.**
  Inspected it directly: the pre-2024 portion has `Volume = 0` and flat
  `Open = High = Low = Close` for extended stretches — the signature of a synthetic/
  theoretical settlement print carried forward in the absence of real trading, not
  genuine market activity. Any consumer of this data needs a `Volume > 0` filter before
  treating a print as real.
- No "list available expiries" API, still true — contract symbols must be constructed
  from known CME/ICE month-code and listing-cycle conventions per product. Resolved this
  in practice by *discovering* listed months empirically (probe candidate tickers, keep
  the ones that resolve) rather than hardcoding a per-product listing cycle — see
  `jobs/capture_term_structure.py`.
- Unofficial behavior riding on Yahoo's internal API — still true, still a real risk
  (same risk class as the yfinance 0.2.50 → 1.5.1 breakage found during the original
  test session). **Pin/verify yfinance ≥ 1.5.1.**

**What was built as a result (2026-07-14):** `jobs/capture_term_structure.py` — a
daily job that discovers each asset's currently-listed contract months (empirically, by
probing, not a hardcoded calendar) and appends their full history to
`Data/term_structure.parquet`, deduplicated on `(date, contract_symbol)` so re-runs are
safe. Every run is logged to `Data/term_structure_manifest.csv`. Live-verified: 38/38
assets fully captured, 133,553 rows, zero duplicate rows on a repeat run (idempotency
confirmed). Wired into Windows Task Scheduler as `CTA_TermStructureCapture`, daily
6:15PM (15 minutes after the existing `CTA_DailyDataUpdate` job, to avoid overlap) — same
interactive-logon caveat as that job applies here. **This job runs under real time
pressure**: any day it doesn't run is a day of near-dated contract history permanently
lost for whatever expires before the next successful run, unlike `update_data.py` which
can always re-pull the continuous series from scratch.

**Free backfill search (2026-07-14) — checked and mostly ruled out:**
- **TurtleTrader (`turtletrader.com/hpd`)** — downloaded and inspected the actual data
  (not just the marketing page). Genuinely is individual contract-month files (e.g.
  `CL00M.txt` = June 2000 WTI) with real OHLCV *and open interest*, correct shape (thin/
  flat pre-liquidity prints, real volume approaching front-month, clean stop at expiry).
  **But the entire archive is frozen at 1999-12-20** — the most recent contract file
  across all 235 crude oil files ends there, before this project's own price panel even
  begins (2000-08-23). Free, correctly-shaped, zero usable overlap with any modern
  backtest window. Not viable.
- **API Ninjas / Commodities-API / OilPriceAPI** — checked; free tiers are delayed-quote
  or continuous-series only, none offer free historical individual-contract-month data.
- **Databento** — new accounts get **$125 in free credits**, but **this account's
  credit has already expired** (confirmed via the portal 2026-07-14): $12.33 was used
  against a March 2026 billing cycle, the remaining $112.67 expired unused on
  2026-04-24, current balance is **$0.00**. A reinstatement request has been drafted
  for the account owner to send; Databento work is parked pending a reply, per direct
  instruction — do not resume until there's a response or new direction. **Separately,
  even fully funded, `GLBX.MDP3` (the CME dataset covering NYMEX/COMEX/CBOT) only has
  data back to 2010-06-06 — it cannot backfill the full 2000-2026 panel, only the
  ~16 most recent years.** Actual costs, confirmed via their (free) cost-estimate
  endpoint for that 2010-2026 window: $7.54 for WTI's full contract history down to
  $0.17 for the 10-year note — extrapolated across the ~30 CME-cleared assets in the
  universe, a full historical pull is estimated at **$50-150** if paid out of pocket.
  Cheaper than any subscription alternative in the table below, but real money, not
  free — do not spend without explicit confirmation. **Scope decision (2026-07-14):
  pull open interest alongside price/volume if Databento's CME/ICE data includes it**
  (confirm when this resumes). Prompted by reading `CTA Feature Engineering Primer.pdf`,
  whose recommended schema includes open interest as a core field — we currently lack
  it entirely (`jobs/capture_term_structure.py` only gets OHLCV from yfinance, no
  OI), which is exactly why Phase 1b's roll rule fell back to a fixed-days-before-expiry
  schedule instead of the more standard OI-crossover
  method. Near-zero marginal cost since this pipeline is already being built.

**Update (2026-07-14): credit reinstated, $206.17 full-scope estimate exceeded the
$125 budget, scope trimmed — known gaps below.** Databento reinstated the account's
unused credit after the drafted email was sent. Priced the full pull (OHLCV +
`definition` schema, needed to map Databento's numeric `instrument_id` back to actual
contract symbols) across all 38 assets: **$81.88 for the 33 CME-cleared assets
(`GLBX.MDP3`, full 2010-06-06→present history) vs. $124.29 for just the 5 ICE-cleared
softs (`IFUS.IMPACT`, Coffee/Sugar/Cocoa/Cotton/OJ, full 2018-12-23→present)** — ICE
data is priced very differently per-asset, not proportional to asset count (Coffee and
Sugar alone each cost more than the entire CME energy complex). Total for everything:
$206.17, over budget.

**Known gaps from the scope trim — backfill later when there's more budget:**
- **ICE softs (Coffee, Sugar, Cocoa, Cotton, OJ) only got 2024-07-14→present (2 years),
  not the full 2018-12-23→present available on `IFUS.IMPACT`.** The 2018-12-23 to
  2024-07-14 stretch (~5.5 years) is not backfilled. Full ICE history would cost ~$124
  on its own — needs its own budget pass, not squeezed into this one.
- **Open interest is still not pulled at all** (separate from the above trim — this was
  already parked pending the OHLCV pull's cost/budget impact, per direct instruction).
  Lives in Databento's `statistics` schema (`stat_type=9` per the `databento_dbn.StatType`
  enum — confirmed authoritative, not scraped), which is event-level tick data, not a
  clean daily field, and needs real parsing work to isolate the official daily print.
  Estimated ~$3.54 for one asset's full statistics history (WTI) — comparable to OHLCV,
  so plausibly another $50-100 across the universe, not yet priced in full.
- CME-cleared assets (33 of 38) got their full available history (2010-06-06→present) —
  no gap there.

**Execution (2026-07-15) — two failed approaches, then a working one:**

- **First attempt: a single script (`databento/backfill_databento.py`) using the
  synchronous `get_range()` API, all 38 assets in one process, writing to
  `term_structure.parquet` only once at the very end.** Appeared to hang after ~40
  minutes with no output. Killed it. **This was a real, costly mistake, not just a lost
  attempt: the process had actually been working the whole time** — Databento's own
  portal usage graph showed $4.20 / 2.6GB genuinely downloaded during that window — but
  because the script buffered everything in memory and Python's `print()` output was
  buffered (not flushed) when not attached to a terminal, there was no visible progress
  to distinguish "slow" from "stuck." Killing it discarded that $4.20 of real,
  already-paid-for data, since nothing had been written to disk yet.
- **Root cause, isolated afterward:** `get_range()` with `schema="definition"` hangs
  reproducibly (confirmed with a 90-second timeout, no response) even for the cheapest
  single asset (UltraBond) and a short date range — while the identical request with
  `schema="ohlcv-1d"` completes in ~7 seconds. `get_range()`'s own docstring says as
  much: *"This method only returns after all the data has been downloaded... for large
  requests, consider using a batch download."* We'd used the wrong API method for the
  size of this request.
- **Working approach: Databento's batch API** (`client.batch.submit_job` /
  `list_jobs` / `download`) — submission returns immediately, processing happens
  server-side, no long-lived local connection to hang or misjudge. Built
  `databento/submit_databento_jobs.py`: submits one job per (asset, schema) pair — 38
  assets × 2 schemas (`ohlcv-1d` + `definition`) = 76 jobs — saving state to
  `Data/databento_jobs.csv` after every asset (not once at the end, learning directly
  from the first mistake). First run: **44/76 submitted successfully, 32 failed on
  HTTP 429 (rate-limited)** — the submission loop had no backoff between calls; the 32
  need a retry with delay/backoff, not a blind full re-run (the 44 successes are already
  queued and shouldn't be duplicated).
- **Batch jobs process slowly and entirely server-side** — observed ~4 jobs processing
  concurrently at any time, rest queued FIFO, no ETA field in the API. Observed
  throughput as of this writing: ~9.8 jobs/hour, ~23/44 done. This is normal Databento
  queue behavior, not a bug, and does not accrue cost while queued (`cost_usd`/`bill_id`
  are `None` until a job actually completes).
- **Still not built: the transform/join stage** — reading downloaded `ohlcv-1d` +
  `definition` files per asset, joining on `(date, instrument_id)` to recover each
  contract's true identity, and merging into `Data/`. The join logic itself already
  exists and is verified correct in `backfill_databento.py`'s `fetch_asset()` (built
  and tested against live `get_range()` data before the pivot to batch) — it needs to
  be re-pointed at the downloaded files instead of a live API call, and the design
  below (found via the first completed asset) changes what it keeps.

**Remaining 32-job submission gap closed (2026-07-17).** The 32 jobs (16 assets) that
failed to submit at all in the original run (`SUBMIT_FAILED: 429 Too Many Requests`,
`job_id` blank, never reached Databento's queue) sat unretried until now — found when
the user asked why UB's (UltraBond) definition file was missing. Checked
`Data/databento_jobs.csv` directly rather than assume: confirmed both of UB's jobs had
`submitted=False`. User already had UB's `ohlcv-1d` file in hand (acquired outside this
job-tracking system); only `definition` needed submitting. Before submitting anything
at scale, checked real cost via `client.metadata.get_cost()` (no balance-check endpoint
exists in the SDK — that's a portal-only check, confirmed by inspecting
`client.metadata`'s available methods) rather than guess: UB definition alone,
$0.0248. Account usage per the portal: $109.41 of $125 credit used, ~$15.59 remaining.
Priced the other 15 pending assets (30 jobs) the same way before submitting anything —
**$7.78 for all of them combined**, comfortably inside budget, no rationing needed
(cheapest: Russell2000 $0.12; priciest: LeanHogs $2.05 — the two livestock products are
the outliers, everything else under $0.65/asset). Built
`databento/retry_databento_jobs.py` — the retry-with-backoff step flagged as needed
but not built back on 2026-07-15 — reads `Data/databento_jobs.csv` for
`submitted=False` rows, resubmits with a 3-second delay between calls (the original
failure mode was zero delay between 76 back-to-back submissions), saves the CSV after
every single submission rather than at the end (same lesson as the earlier killed-
process/$4.20 incident). All 30 submitted cleanly with zero 429s this time — **76/76
jobs across the full 38-asset universe now submitted, 0 pending.** Processing is async
server-side as before; download/transform still needs to happen per asset as each
finishes, same pattern LE already proved out.

**Raw-pull off-machine archive built and run (2026-07-16/17).** Databento job outputs
expire 30 days after `ts_process_done` (confirmed both via the API's `ts_expiration`
field and the user's own portal) — real, ticking deadline per job, not indefinite
storage. Decided where the 2.3GB (and growing) of raw zips in `Data/databento_raw/`
should actually live: this project's own folder is nominally under the user's
`OneDrive` path, but `OneDrive.exe` was confirmed not running (checked directly, not
assumed from the folder name) — so despite appearances, nothing was being cloud-synced.
User is already paying for Google One (100GB, ~94.6GB free) and chose a scripted
upload over installing the Google Drive desktop client, to avoid adding a persistent
background sync process. Installed `rclone` (via `winget`), configured a
`drive.file`-scoped remote (`gdrive_cta` — least-privilege: rclone can only see/manage
files it creates, not the whole Drive) via its own OAuth flow. Built
`databento/archive_to_drive.py` (upload + checksum-verify via `rclone check`, logs to
`Data/databento_archive_manifest.csv`; deliberately does *not* delete local files —
that's a separate, explicit step, never automatic).

Two real bugs found running it, not assumed fixed on the first pass:
- **First run silently died partway (11/45 files) because it was backgrounded with a
  bare `&` inside a bash call rather than `nohup`** — when that shell session ended,
  the still-running upload process died with it. The script's own manifest write only
  happened once, at the very end of the full loop, so the local tracking file was left
  completely empty even though 11 uploads had already genuinely succeeded and were
  sitting correctly on Drive — same failure shape as the earlier killed-`get_range()`-
  process/$4.20 incident (Phase 4 above), a lesson that evidently needed re-learning at
  a second layer of this same project. Fixed the same way: manifest now saves after
  every single file. Reconciled the true state directly against Drive (`rclone check`
  on the whole directory, not trusted from the broken log) before resuming, rather than
  guess or blindly re-upload everything.
- **Resuming with `nohup`+`disown` this time, the previous (first) attempt's process
  turned out to still be alive** — both copies read "pending" near-simultaneously and
  raced on 6 files (SI/RB/PL/NG Definition+OHLCV), each independently uploading and
  verifying them. Not a data-loss bug - each upload was individually genuine and
  checksum-correct — but Google Drive allows multiple objects with identical filenames
  to coexist (unlike a real filesystem), so this left 51 objects on Drive for 45 real
  files, and a manifest with 90 rows for 45 files. Fixed with `rclone dedupe
  --dedupe-mode newest` (safe here since the duplicates are byte-identical copies of
  the same source file) plus a local manifest de-duplication. **Final reconciled
  state, verified three ways at once: 45 local zips = 45 manifest entries = 45 Drive
  objects, exact match.**

Not yet built: the actual local-deletion step (`cleanup_local_after_verified_upload()`
exists in the script, defaults to a dry run, never called automatically) — freeing
local disk space is still a deliberate, separate action pending the user's go-ahead,
not something this upload step does on its own. Also unresolved: rclone warned its
shared Google Drive OAuth client ID "is being retired... during 2026" — this setup may
need a dedicated client ID later this year to keep working; not urgent now, logged so
it isn't a surprise later.

**Universe expansion, 4 new assets (2026-07-17).** With ~$7.79 of Databento credit
left, evaluated candidates beyond the original 38 - VIX (`XCBF.PITCH`, confirmed
available but $109 full history / $12.70 even at 1yr, too expensive - Cboe lists both
monthly and weekly VIX futures under one parent query, driving up `definition` cost) and
palm oil (confirmed **not available at all** - checked the full Databento dataset
catalog directly, no Bursa Malaysia or equivalent Asia-Pacific commodity exchange in
it, not a cost problem) were ruled out first. Landed on **SOFR (`SR3`), Lumber (`LBR`,
the current physically-settled contract - not `LBS`, discontinued ~2022), Swiss Franc
(`6S`), Mexican Peso (`6M`)** - $5.90 total, chosen specifically because each also
passed a check the earlier candidates didn't get: **confirmed, not assumed, that
`jobs/capture_term_structure.py`'s free daily forward-capture mechanism actually works
for it**, by running its own `discover_contracts()` against each candidate before
committing. Two real candidates got dropped for exactly this reason despite being cheap
and Databento-available: **Ethanol** (`EH`, $2.51) - 0/6 candidate months resolved via
yfinance on both `CBT` and `CME` exchange-code guesses, not confirmed working - and
**old-contract Lumber** (`LBS`, $0.33) - 0/6, expected, since nothing currently lists
under a contract discontinued in 2022. Both would have been one-time historical
snapshots with no way to keep extending them, the same structural gap already logged
for spreads/butterflies (`yfinance` has no combo-instrument tickers at all) - not
worth adding regardless of price, per the standing principle that this project doesn't
carry data it can't honestly keep current. Databento batch jobs submitted (8 jobs, all
`GLBX.MDP3`, 2010-06-06→2026-07-14, matching the existing scope's dates) and all 4
assets added to `jobs/capture_term_structure.py`'s `UNIVERSE` (confirmed working
root/exchange pairs: SOFR 6/6 candidate months resolved, MexicanPeso 4/6, SwissFranc
3/6, Lumber 3/6 - sparser listing cycle than most, not a problem, just fewer months
out) and to `databento/submit_databento_jobs.py`'s `UNIVERSE` for consistency. **Update
2026-07-20: 3 of the 4 (SwissFranc, MexicanPeso, Lumber) added to `get_data.ipynb`'s
continuous-series pull and `jobs/update_data.py`, plus the dashboard's `ASSET_CALENDAR`
mapping** — see the Phase 0 note above. SOFR is not included: no usable Yahoo
continuous ticker exists for it (checked live, not assumed), so it stays out of the
core OHLCV panel permanently, not just for now. The still-open Soybean Meal/Oil
decision (#3 in the open decisions log) remains a separate, unrelated item.

**Category note (2026-07-20):** the four assets above were grouped as "new
additions" here purely because of *when* they were added, not what they are —
that grouping isn't a real asset category and shouldn't be treated as one going
forward. By actual category: **SwissFranc and MexicanPeso belong with FX**
(currency futures, mechanically identical to EURUSD/JPYUSD/GBPUSD/AUDUSD/CADUSD);
**SOFR belongs with rates** (an interest-rate future, cash-settled against a
reference rate rather than a Treasury note/bond, but still a rates instrument);
**Lumber has no natural category among this project's existing groupings**
(not a grain, metal, energy, livestock, FX, or rate) and stays its own category
of one.

**Rollout approach revised 2026-07-19, twice: asset-by-asset, then to deliberate,
fully-documented exploration with no target pace.** All 42 assets' raw Databento data
is now fully downloaded locally (84 files, `Data/databento_raw/`) and backed up to
Google Drive (`databento/archive_to_drive.py`, `gdrive_cta:CTA_Databento_Raw`, 84/84
reconciled) — the transform itself is validated against LE, KC_Wheat, and (as of this
update) **all 5 ICE-cleared softs: Coffee, Sugar, Cocoa, Cotton, OrangeJuice.** Coffee's
three findings (sentinel prices, `_Z`-variant instrument, incomplete-print rows)
confirmed general to `IFUS.IMPACT`, not Coffee-specific — Sugar/Cocoa/Cotton/OJ all show
the same shape at similar magnitude, no new anomaly types across any of them. Also
resolved, while running OJ: the pre-existing 3 non-positive `OrangeJuice` rows flagged
under Coffee turned out to be a genuine no-trade day for 3 far-dated contracts in
Databento's real feed (confirmed directly against the raw file, not assumed) that a
separate, pre-existing `jobs/capture_term_structure.py` (yfinance) bug wrote a spurious
zero for — out of scope to fix this pass, logged in `databento/DATA_QUALITY_REPORT.md`
Asset 6 so it isn't rediscovered as a mystery later. Full detail and the per-asset
comparison table: `databento/DATA_QUALITY_REPORT.md` Assets 3-6. **All 33 CME-cleared
assets remain unexplored** (2 of 33 done: LE, KC_Wheat) — next natural step per the
category-diversity goal below.

**Metals (2026-07-19): Gold done, clean, first real category jump.** `status=OK`, 0
anomalies — 52,398 outright rows, 100,425 spread rows, 0 butterflies, 0 condors.
Verified the zero butterfly/condor count directly against raw `definition` data
(all 12,638 unique spread symbols are plain 2-leg, no `:BF`/`:CF` marker anywhere)
rather than trusted at face value — a real COMEX fact, not a missed marker format.
Also 0 inter-commodity spreads, unlike LE's ~35% LE-HE cross-commodity pairs — not
assumed to generalize to the other 4 metals (Silver, Platinum, Palladium, Copper)
without checking each. **Process lapse, logged plainly:** this run executed without
the usual pre-run backup; confirmed safe after the fact by exact reconciliation
(51,124 = 51,124, nothing else moved), backup taken immediately after. Full detail:
`databento/DATA_QUALITY_REPORT.md` Asset 7.

**Silver, Platinum, Palladium done, clean, confirm Gold's pattern (2026-07-19).** All
three `status=OK`, 0 anomalies, 0 butterflies/condors/cross-root spreads, sane price
levels — checked individually rather than assumed, per the same discipline as the ICE
softs. Metals now 4-of-5 done and structurally simple as a category — no per-asset
surprises. **Copper (`HG`) is `BLOCKED`, not a transform bug**: its definition zip is
corrupted (truncated, missing the zip's End-Of-Central-Directory record), confirmed
reproducible on a second download (identical byte count and truncation point). Traced
via the Databento API directly: the underlying batch job completed successfully and
isn't expired, and Databento serves it as individual per-day files, never a zip — no
script in this repo builds that zip, so the truncation happened in an untracked
download/portal step, not in anything on our side. Logged to the manifest
(`status=BLOCKED`); user following up with Databento support directly rather than
guessing at a local fix. Full detail: `databento/DATA_QUALITY_REPORT.md` Assets 8-11.
**Metals: 4 of 5 done (Gold, Silver, Platinum, Palladium); Copper blocked pending
Databento support.**

**Rates: US_10Y (ZN) done, clean, first fixed-income category jump (2026-07-19).**
Verified three real structural differences *before* running the transform (fractional
32nds price-display fields genuinely populated but inert for our purposes; quarterly-
only listing cycle; contract size via `unit_of_measure_qty`, same field LE used).
`status=PARTIAL`, 10,503 outright rows, 4,955 spread rows, 0 butterflies, 0 condors —
both non-outright findings investigated and confirmed legitimate: (1) the one real
listed butterfly instrument has **zero** OHLCV prints across the entire 16-year pull
(real zero volume, not a parser miss — rates curve trades apparently go through
calendar spreads, not butterflies), (2) `spread_not_2_legs=957` is CME's
**User-Defined Spread** facility (bespoke, participant-negotiated combos, raw symbols
like `UD:ZN: TL 0825829457`) — confirmed CME's `GLBX.MDP3` definition schema has no
leg-decomposition fields for these (unlike ICE's schema), so there's no data-driven
way to resolve them; permanently out of scope, not a fix candidate. Also
`spread_unknown_root_exchange=31`: real spreads against CME's Ultra 10-Year Note
(`N1U`), a product outside our 42-asset universe — correctly skipped. Full detail:
`databento/DATA_QUALITY_REPORT.md` Asset 12.

**Rates: all 5 CME Treasury-complex assets now done (2026-07-19) — UltraBond, US_5Y,
US_2Y, US_30Y join US_10Y, confirmed not assumed.** Each checked individually: same
shape across all five (quarterly-only listing, 0-to-negligible butterfly volume,
`UD:` User-Defined Spreads as the only recurring non-outright anomaly, occasionally
paired with a real inter-commodity spread against an untracked root — `B1U`/`MWN`
for UltraBond, `F1U` for US_5Y, none for US_2Y/US_30Y). All merged clean (0
non-positive prices, 0 OHLC violations). **Rates fully explored as a category.**
Also recovered Copper (`HG`, metals) in parallel: the corrupted definition zip's
root cause was Databento's server-side zip-bundling endpoint timing out (504) on
this job — worked around for $0 by re-fetching all 5,042 individual files directly
(`filename_to_download`, bypassing the zip-bundle path entirely), each SHA256-
verified against Databento's own reported hash, then rebuilding a valid zip
locally. Re-ran clean: `status=OK`, 0 anomalies, same shape as the other 4 metals.
**Metals: all 5 done.** Full detail: `databento/DATA_QUALITY_REPORT.md` Assets
11 (update), 13-16.

**Energy started 2026-07-19 with WTI Crude (`CL`) — most consequential single asset
this session.** Caught a real correctness bug before it could propagate: the
ICE-derived "drop non-positive outright rows" filter would have silently discarded
the actual, historically documented April 2020 negative-price event
(`CLK20.NYM`, low=-40.32/close=-2.67 on 04-20, continuing negative into 04-21) -
found because the filter flagged it, checked before accepting the drop (per the
same discipline as Asset 3's negative-price verification), confirmed real via web
search (WTI's own well-known settlement event; most other commodities, including
coffee, have no comparable precedent). **Fixed the filter's signature from
"any field non-positive" to "any field exactly zero"** - preserves genuine negative
prints while still catching ICE's real bug. Verified this doesn't retroactively
break the 5 already-merged ICE assets (re-checked old vs. new filter side by side,
0 differences across all five). Also found and fixed a mechanical symbol-format gap
(WTI's far-dated legs need 2-digit year codes past a point - confirmed a real CME
convention, not ambiguous) and **built parsers for two new, real, liquid spread
grammars directly matching Phase 2d's roadmap** (below) rather than skip-and-log,
since the data was already there and the reuse case was immediate. Spread
resolution: 3,681/4,361 → 4,352/4,361 (99.8%). Full detail, including the exact
regex grammars and validation evidence: `databento/DATA_QUALITY_REPORT.md`
Asset 17.

**Brent (`BZ`) done next (2026-07-19) — cross-validated the WTI-Brent parser and
found Brent's crack spreads need a more general grammar than WTI's** (both legs can
have independent month codes, not always a shared month) - fixed, verified no
regression on WTI, confirmed on Brent: crack/spread resolution 2,128/2,725 →
2,725/2,725 (100%). **The real payoff: all 66,291 CL-BZ spread rows' `BZ`-side legs
now cross-checked against Brent's own freshly-pulled real outright universe - 0
mismatches** - closing the same validation gap LE's original inter-commodity
cross-check had (the partner leg's own data didn't exist yet at merge time). Full
detail: `databento/DATA_QUALITY_REPORT.md` Asset 18.

**Energy category complete (2026-07-19): RBOB, HeatingOil, Natural Gas join
WTI/Brent — all 5 done.** RBOB and HeatingOil closed the crack-spread validation
loop: both RB-CL (24,420 rows) and HO-CL (22,251 rows) legs, resolved during WTI's
own run before either asset had its own data, cross-checked against their own real
outright universes — **0 mismatches in both cases**, same clean result as Brent's
WTI-Brent legs. Natural Gas surfaced two more spread grammars (`:XS`, always paired
against Henry Hub — not a tracked future, so even parsed the leg would be
permanently unusable; `:SB`, genuinely tiny volume) — deliberately left as
log-and-skip rather than built, a real cost/benefit call rather than an oversight.
Condor logic (built for KC_Wheat, a grain) generalized to Natural Gas (energy) with
zero changes: 14/14 resolved, 0 mismatches. All 5 energy assets merged with 0
non-positive prices beyond the already-known/explained cases. Full detail:
`databento/DATA_QUALITY_REPORT.md` Assets 19-21.

**Fixed the same exact-zero bug in `jobs/capture_term_structure.py` (the live
scheduled yfinance job), not just the one-off Databento transform (2026-07-20).**
A new non-positive Cocoa row appeared mid-session with no Databento work anywhere
near Cocoa — traced to the separate `CTA_TermStructureCapture` scheduled task
firing on its own 6:15PM trigger while this session continued (confirmed via
`Get-ScheduledTaskInfo`, not assumed — real elapsed time was much longer than it
felt). A weekend-run hypothesis was raised and **directly tested, not just
assumed either way** — disproven by reproducing the identical zero-print live via
`yfinance` on the following Monday. Real cause: a far-dated, zero-volume contract
serving a flat carried-forward quote for an extended no-trade stretch (the same
"synthetic settlement print" pattern already documented in `DATA_SCHEMA.md`
section 1 for a different contract), glitching to an exact-zero
open/high/low on one isolated day within that stretch. **Fix**: the identical
exact-zero filter already validated for `transform_databento.py` (Asset 17), now
also in `pull_contract_history()`. Also cleaned the 4 already-known bad rows (3
OrangeJuice + this new Cocoa row) directly out of `Data/term_structure.parquet` —
confirmed safe first: the retroactive filter removed exactly those 4 rows and
nothing else, leaving the 2 genuine WTI negative-price rows untouched. Full
detail: `databento/DATA_QUALITY_REPORT.md`, "Cross-cutting fix" section (after
Asset 21).

**Livestock category complete (2026-07-20): LeanHogs and FeederCattle join
LiveCattle — all 3 done.** LeanHogs confirmed LE's own original cross-commodity
finding from the other side: the real LE-HE spread (11,766 rows) can now be
independently checked from both legs, not just LE's. FeederCattle surfaced
another generic fix: 385 rows (0.3%) had no `definition` row on the exact
matching date, though every instrument involved had one on an earlier date — CME
apparently stops republishing some instruments' definitions shortly before their
last real trade (near expiry/delisting). **Fixed generically**: the main join in
`transform_asset()` now falls back to the most recent *prior* definition entry
for the same `instrument_id` when the exact-date match fails — safe because an
instrument's identity fields never change over its life (established since
Asset 1/LE). Verified: recovered all 385 rows, 0 regression on WTI (already
0 unmatched, unaffected). Also notable: FeederCattle has **zero** cross-commodity
spread rows, unlike LE-HE — a real per-asset market-structure difference within
the same category, not assumed to generalize. Full detail:
`databento/DATA_QUALITY_REPORT.md` Assets 22-23 and the "Generic fix" section.

**Grains category complete (2026-07-20): Corn, Wheat, Soybeans, Rice, Oats join
KC_Wheat — all 6 done.** The real payoff: a validated grain-complex
cross-commodity spread network, directly serving Phase 2d's own candidate
table — **ZC-ZW (Corn/Wheat, 3,939 rows) is the literal Phase 2d "Corn/Wheat
spread" candidate**, now backed by real, exchange-quoted data instead of just a
cointegration test on outrights. Also found real KE-ZC and KE-ZW (KC_Wheat vs.
Corn/Wheat) spreads, the latter (29,636 rows) the single largest cross-commodity
pair found all session. ZC-ZW legs cross-checked against Wheat's own outright
universe: 0 mismatches. **Soybeans has zero cross-commodity spreads** — checked,
not assumed — because its natural partners (Soybean Meal/Oil, the Crush) aren't
in the 42-asset universe yet (open decision #3, still unresolved). Rice and Oats
were the cleanest assets of the entire session: `status=OK`, zero anomalies of
any kind. Full detail: `databento/DATA_QUALITY_REPORT.md` Assets 24-28.

**SOFR done (2026-07-20) — richest single-asset exploration this session,
completing the "rates" family by category (not by original when-added
grouping).** Explored the raw data thoroughly *before* running anything, per
direct correction mid-session (every prior category's first asset got this
treatment; this one initially didn't and should have). Found 46 simultaneously-
listed monthly contracts (vs. 3 quarterly for Treasuries) and **four spread
markers never seen before**: `:AB`, `:DF`, `:BB`, `:SB`. `:DF` (condor-shaped)
and `:BB` (butterfly-shaped) were free wins — 100% resolved reusing the existing
condor/butterfly logic unchanged, just recognizing the extra marker string.
`:AB` (CME **SOFR Average futures** — a real, single-quote instrument, 78.6M
contracts of volume, the largest combo type found all session) and `:SB` (CME
**Pack/Bundle spreads** — 674K contracts, two distinct raw-symbol grammars for
the same strategy) both needed genuinely new modeling — **two new output tables
added**, `Data/term_structure_averages.parquet` and
`Data/term_structure_packs.parquet`, both generic (not SOFR-specific) so any
future asset using the same markers picks them up automatically. `:AB`: 121/121
resolved, 0 mismatches. `:SB`: 90/90 resolved, 0 mismatches. Also hit the
FeederCattle join-fallback fix again (537 rows) — handled automatically, no new
code needed. **Rates family (by category) now fully done**: US_10Y, UltraBond,
US_5Y, US_2Y, US_30Y, SOFR. Full detail: `databento/DATA_QUALITY_REPORT.md`
Asset 29.

**Equity index started 2026-07-20 with SP500 and Nasdaq100.** SP500 was the
cleanest category start this session — `status=OK`, 0 anomalies of any kind
(cash-settled index futures don't have the physical-delivery/rate-curve
conventions that produce butterflies/packs/averages elsewhere). Nasdaq100
surfaced a real, pre-existing data-corruption bug unrelated to any code from
this session: its raw OHLCV zip actually contained **SP500 data**, not
Nasdaq100 (confirmed reproducible across two dates, and confirmed via the
Databento API that the original job was correctly submitted for `NQ.FUT` and
completed fine — the mixup happened in a local download/assembly step
predating this session). **Recovered the same way as Copper**: all 4,996
individual files re-downloaded directly from Databento, SHA256-verified, $0
cost, rebuilt into a clean zip. Re-ran clean: `status=OK`, 0 anomalies, price
range matches the real Nasdaq-100 index. Also fixed a real Windows-specific
file-locking bug found while diagnosing this — `term_structure_spreads.parquet`
grew large enough that polars' reader memory-maps it, and Windows then refused
to let the same process overwrite that path. **Fix**: every output-table write
now goes through a new `_write_parquet_atomic()` helper (write to temp, then
`os.replace`), protecting every future asset's merge+write. Full detail:
`databento/DATA_QUALITY_REPORT.md` Assets 30-31.

**Equity index category complete: Dow and Russell2000 join SP500/Nasdaq100, all
4 done.** Both checked directly for the same NQ-style symbol mismatch before
running anything (given it had just been found in this category) — both clean.
Both confirmed clean, `status=OK`, 0 anomalies of any kind. Dow's definition
data showed 2 real butterfly symbols (unlike SP500's zero), but neither ever
had a real OHLCV print — same benign pattern already seen for `ZN`/WTI/SOFR. 0
cross-index spreads found for any of the 4 (unlike grains/energy). **Equity
index was the cleanest category overall this session**, aside from the one
real corruption incident on Nasdaq100. Full detail:
`databento/DATA_QUALITY_REPORT.md` Assets 32-33.

**Lumber done (2026-07-20) — its own category, fully complete.** Checked first
for the same NQ-style symbol mismatch (clean). Confirmed a new physical unit
(`BDFT` = board feet, `unit_of_measure_qty` → 27,500 board feet, the real
modern post-2022 CME contract size already in use), the sparse odd-month
listing cycle already logged when Lumber was added to the universe, and a real
cross-commodity spread partner (`SYP`, correctly untracked) plus spreads
against the old discontinued `LBS` contract. `status=PARTIAL` (expected, fully
explained), 0 non-positive prices. **35 of 42 assets now transformed via
Databento** (verified against the manifest directly, not the outright table's
asset count, which also carries yfinance-only rows for not-yet-transformed
assets) — the remaining 7 are all FX. Full detail:
`databento/DATA_QUALITY_REPORT.md` Asset 34.

**FX complete — and with it, the entire 42-asset Databento transform rollout
(2026-07-20).** All 7 FX assets (EURUSD, JPYUSD, GBPUSD, AUDUSD, CADUSD,
SwissFranc, MexicanPeso) checked for the NQ-style mismatch first; found a
different, real bug on CADUSD - its Definition and OHLCV zips were simply
**swapped** (wrong filenames, not wrong/missing/corrupted data - confirmed
both sides held genuine CADUSD content before fixing by renaming, no
re-download needed). All 7 merged clean, `status=OK`, 0 non-positive prices.
0 cross-currency-pair spreads anywhere in FX, matching equity index's pattern.

**Final state, this staged rollout (started 2026-07-19, per direct instruction
to go asset-by-asset and document everything rather than race to completion):**
all 42 assets transformed, `term_structure.parquet` (outrights) now has all 42
represented, 1,318,677 rows. Two new output tables were added along the way
(`term_structure_averages.parquet`, `term_structure_packs.parquet`, both from
SOFR's Asset 29 exploration). Non-positive prices anywhere in the final
outright table: exactly 2, both the real WTI April-2020 negative settlement,
not an artifact. Every anomaly found this session — sentinels, incomplete
bars, decade ambiguity, new spread/butterfly/condor/pack/average markers,
join-fallback gaps, and three separate corrupted/mislabeled/swapped raw zip
incidents (Copper, Nasdaq100, CADUSD) — was diagnosed against real evidence
and either fixed generically (benefiting every future asset) or explicitly
logged as an understood, deliberate skip. Full detail across all 41 asset
write-ups: `databento/DATA_QUALITY_REPORT.md` Assets 1-41 plus the two
cross-cutting fix sections.

**Phase 2d update: real, exchange-quoted WTI-Brent and crack-spread data now exists
in `Data/term_structure_spreads.parquet`**, not just planned. 66,291 CL-BZ
(WTI-Brent) rows and 24,065 RB-CL + 21,871 HO-CL (crack spread legs) rows, all
leg-decomposed the same way every other spread in this table is. See the updated
Phase 2d candidate table below - this doesn't start Phase 2d itself (signal
sequencing per `CLAUDE.md` is unchanged, RV spreads are still 4th priority), it just
means the data is ready and directly-quoted (no back-differencing risk) whenever
that phase starts. Originally planned as a canary batch (one asset per `UNIVERSE` category
run together), revised to one-at-a-time after KC_Wheat's condor finding, then revised
again after Coffee: **not a rollout to finish, an exploration to learn from** — per
direct instruction, no rush, the goal is to understand and document each dataset's
characteristics as thoroughly as possible, using that understanding well, rather than
race to 42/42. Rationale accumulated fast: LE surfaced two real design gaps (anchor-leg
algorithm, cross-commodity-spread schema), KC_Wheat a third (condor spreads), and Coffee
a fourth *and* fifth *and* sixth in one pass (Databento's `UNDEF_PRICE` sentinel leaking
into `ohlcv-1d`, a second non-representative `_Z`-suffixed instrument per ICE contract
month, and incomplete daily bars on the genuine instrument) — four-for-four assets each
carrying undiagnosed peculiarities, several serious enough that they'd already been
silently merged into `term_structure.parquet` before being caught (see
`databento/DATA_QUALITY_REPORT.md` Asset 3 for the full Coffee writeup, including how
close this came to leaving fabricated near-zero/negative coffee prices in production
data). **Fixes found along the way are generic where the underlying issue is generic**
(the `UNDEF_PRICE` sentinel filter and the incomplete-bar drop apply to every future
asset; the ICE-specific `_Z`-variant exclusion is gated to `exchange ==
ICE_EXCHANGE_SUFFIX`) — so this approach still compounds into reusable infra, not
one-off patches, even at a slower pace. Every call remains explicit/named
(`transform_asset(asset, root, exchange)`), never bare `run()`.

**Per-asset findings are now logged in `databento/DATA_QUALITY_REPORT.md`, one section
per asset, not just summarized here** — restructured 2026-07-19 from a single-asset (LE)
report into a running multi-asset log, per direct instruction, specifically so the exact
evidence behind each fix can be traced back once this is all done, not just the
conclusion. `WORKFLOW.md` keeps the summary; that file keeps the detail.

Per-asset loop, no fixed pace, repeated for each remaining asset:

1. **Run `transform_asset(asset, root, exchange)` for exactly one named asset.**
2. **Inspect the result dict / manifest row directly** for new anomaly keys, not just
   `status`/exit code — a new anomaly *type* is the signal this asset has its own
   peculiarity worth understanding, not just skipping past.
3. **If anything looks off, stop and diagnose it from the raw extracted files before
   merging** — don't trust `transform_asset()`'s own write to gate correctness; it
   writes to disk unconditionally, before validation stats are even computed (a real
   design property discovered this session diagnosing Coffee, not a bug — but it means
   "diagnose before merging" has to be enforced by us, not the function). If a bad
   merge already happened, backups make it a one-asset revert, not a forensic recovery.
   Verify a suspicious pattern against real-world domain facts before concluding it's a
   data bug (e.g. Coffee's near-zero/negative prices were checked against coffee's real
   price history and coffee's lack of any negative-pricing precedent, not assumed).
4. **Document every finding in `databento/DATA_QUALITY_REPORT.md`**, whether it changed
   code or not — the point is building a complete per-asset record, not just fixing
   what breaks the pipeline.
5. **Backup `Data/term_structure*.parquet` (+ spreads/butterflies/condors) before each
   asset's run.**
6. **Spot-check the dashboard periodically** (via the `run` skill / Playwright) once a
   few assets are in — both real dashboard bugs found this session (the
   active-contract-window gap, the decade-ambiguous spread-selection bug) were found by
   actually looking at live output, not by reading code.
7. **No fixed order or batch size** — whatever surfaces the most category diversity
   next (still worth touching one metals/grains/ICE-softs/livestock/equity-index/rates/
   FX asset reasonably early, and Coffee already forces the question of whether the
   other 4 ICE softs share its `_Z`-variant/incomplete-bar characteristics), but no
   pressure to reach any particular count.

**`KC_Wheat` anomaly diagnosed and fixed (2026-07-19).** The prior (reverted) run's
`spread_not_2_legs=24` was not malformed data — it was 24 genuine CME **condor spreads**
(4-leg, e.g. `KE:CF H4K4N4U4`), a third exchange-listed spread type alongside the 2-leg
calendar spread and 3-leg butterfly, using a space-then-concatenated-leg-codes format
(`ROOT:CF <code1><code2><code3><code4>`, no dashes) that `resolve_spread_legs`'s
dash-split correctly recognized as not-a-2-leg-spread and logged+skipped — the same
"log and skip, don't guess" behavior already used for ICE's unsupported schema, not a
bug corrupting data. Materiality was tiny (92 rows / 24 instruments across the full
2014-2026 pull, vs. KE's 11,418 real butterfly rows across 83 instruments) but, per
direct instruction, condor support was added rather than left as a permanent skip,
generalizing the existing anchor-leg-plus-modular-offset algorithm (already used for
spreads/butterflies) to 4 legs, **generic in `transform_asset()` — applies to every CME
asset processed from here on, not just KC_Wheat.** New output table:
`Data/term_structure_condors.parquet` (`near`/`mid1`/`mid2`/`far` leg-decomposed,
mirroring the butterfly schema). Single-root assumption carried over from butterflies
(0 cross-root condors observed in KC_Wheat, same as butterflies in LE) — flagged, not
proven, for other assets.

**Exploration findings (2026-07-15) — first completed asset (LE / Live Cattle, both
schemas downloaded).** Full detail, evidence, and file-level checks in
`databento/DATA_QUALITY_REPORT.md`; summary here:

- **The raw `symbol` field is decade-ambiguous — do not use it as a contract key.**
  CME's raw symbology uses a single-digit year code (`LEZ5` = December, year ending in
  5), which is genuinely ambiguous over a 16-year pull (2010-2026 spans two 5s: Dec
  2015 *and* Dec 2025). Confirmed directly: `instrument_id=7780` covers 2014-07-02 →
  2015-12-31 (closes $118-160), `instrument_id=42041124` covers 2024-07-02 → 2025-12-31
  (closes $173-248) — both report `symbol="LEZ5"`. 47 of 60 outright symbols observed
  had 2 distinct `instrument_id`s for exactly this reason. **Fix:** derive the true
  4-digit contract year from the `definition` file's `maturity_year` field (joined via
  `instrument_id`, confirmed clean — `maturity_year=2015` vs `2025` for the two
  `instrument_id`s above), not from the symbol string. `instrument_id` itself is stable
  and never reused across contract instances — it's the *symbol string* that's lossy,
  not the join key.
- **Parent-symbol queries (`stype_in: "parent"`) return far more than outright
  futures.** Of 166,093 OHLCV rows for this one asset: 35,106 (21%) are outright
  futures, 108,885 (66%) are exchange-listed calendar spreads (two-legged, e.g.
  `LEV6-LEZ6`), and 22,102 (13%) are butterflies (three-legged, e.g.
  `LE:BF Z6-G7-J7`). This was already anticipated (see the original transform-stage
  note above) but not previously quantified — worth knowing before repeating this
  query pattern across the other 32 CME assets, since Databento almost certainly bills
  by data volume and a large majority of it is unusable for the outright term
  structure specifically.
- **Decision: keep all three instrument classes, not just outrights.** Originally
  planned to filter to `instrument_class == "F"` and discard the rest. Reconsidered:
  the calendar spread prices are directly usable, not just discardable noise — see
  below. Revised transform-stage output: three parquet files instead of one —
  `Data/term_structure.parquet` (outrights, as originally planned),
  `Data/term_structure_spreads.parquet`, `Data/term_structure_butterflies.parquet` —
  all keyed by the same `(date, instrument_id)`-joined, decade-disambiguated contract
  identity. No data discarded; each instrument class gets its own honestly-labeled
  table rather than being forced into a schema built for outrights only.
- **Calendar spreads are a better carry source than back-differencing two outrights.**
  The carry formula above needs exactly `F_near − F_deferred` — a calendar spread
  instrument (`LEV6-LEZ6`) *is* that quantity, already traded as one instrument with
  its own live market, not something to reconstruct from two separate legs. This
  avoids the leg-alignment risk of back-differencing (one leg stale or thin on a given
  day corrupts a synthetic spread in a way that has nothing to do with real carry) and
  is often more liquid than either leg individually, since spread trading is common in
  ag/livestock futures specifically so traders don't have to leg in manually. **Once
  `term_structure_spreads.parquet` exists, prefer it over back-differencing
  `term_structure.parquet`'s outrights for carry construction.**
- **Data quality otherwise clean** for this asset: 0 unread/malformed files across
  9,086 files (4,050 OHLCV + 5,036 definition), 0 non-positive outright prices, 0
  zero-volume outright rows, 0 duplicate `(date, symbol)` rows. File-count mismatch
  between the two schemas (4,050 vs 5,036) is expected, not a gap — `ohlcv-1d` only
  produces a file on trading days, `definition` snapshots publish on non-trading days
  too.

**Transform pipeline design (2026-07-16) — precise plan, not yet built.** LE is the
experimental first case; the pipeline below is designed to repeat unchanged for the
other 32 CME assets as their pulls arrive. Built in polars throughout (not pandas) —
this is the actual production path the earlier exploration was prototyping, not a
one-off script. Lives in `databento/transform_databento.py` (new file — a one-time/
occasional-run utility, not a scheduled job, so `databento/` not `jobs/`).

*0. Reused conventions — do not redefine these, import them:*
- `UNIVERSE` (asset → `(root, exchange)`) from `jobs/capture_term_structure.py` — the
  single source of truth for asset naming already used everywhere else in `Data/`.
- `MONTH_CODES = "FGHJKMNQUVXZ"`, same file.
- **Target `contract_symbol` format, confirmed by re-reading `capture_term_structure.py`
  directly rather than from memory:** `f"{root}{month_code}{yy:02d}.{exchange}"` — e.g.
  `LEZ26.CME`, *not* the 4-digit-year, no-suffix form used ad hoc during exploration
  (`LE` + `Z` + `2026`). The whole point of this transform is to land in the exact
  schema `Data/term_structure.parquet` already uses: `date, open, high, low, close,
  volume, asset, contract_symbol, root, exchange, expiry_code, expiry_year` (`asset`
  is the canonical name, e.g. `"LiveCattle"` — not `"LE"`, which is only the `root`).

*1. Discover & extract, per asset:*
- Match each `UNIVERSE` root against `Data/databento_raw/*.zip` (LE's pattern:
  `CME_Globex_MDP3.0_{root}_FUT_{Definition,OHLCV}.zip`). **Flagged, not assumed:**
  this exact naming pattern is confirmed only for LE (CME/GLBX.MDP3) — the 5 ICE assets
  use a different dataset (`IFUS.IMPACT`) and may name their exports differently; the
  discovery step will need verifying, not assuming, once a non-CME asset's files
  arrive.
- Extract to `Data/databento_raw/extracted/{root}/{definition,ohlcv}/` if not already
  present (skip re-extraction if the target folder already has files — idempotent).

*2. Read (polars, production-scale — not the exploration script's per-file Python
loop):*
- `pl.scan_csv(glob_pattern, ...)` per schema per asset — lazy, multi-threaded file
  reading across all daily files at once, rather than looping `pl.read_csv` per file
  and `pl.concat`-ing in Python (what the exploration did; fine at ~9,000 files for
  one asset, not the right pattern at 33-asset scale).
- `.select(...)` the 7 needed `definition` columns before `.collect()` — projection
  pushdown, confirmed in the exploration to cut real work (8 of 65 columns
  materialized, not all 65).

*3. Deduplicate `definition`* on `(date, instrument_id)`, `keep="first"` — confirmed
safe: duplicate rows carry identical values (`DATA_QUALITY_REPORT.md` Section 3), not
conflicting ones.

*4. Join* `ohlcv-1d` → deduplicated `definition` on `(date, instrument_id)` — left
join, confirmed 0% unmatched for LE. Log the unmatched count per asset to the manifest
(step 9) rather than assume it stays 0% for every asset.

*5. Derive the true contract identity* — `maturity_year` (definition) + `maturity_month`
→ `expiry_code` (via `MONTH_CODES`) → `contract_symbol = f"{root}{expiry_code}{yy:02d}.{exchange}"`,
`expiry_year` = full 4-digit `maturity_year`. Never derived from the raw `symbol`
string (Section 1, decade-ambiguous).

*6. Classify into 3 instrument classes* via `instrument_class` (`F`/`S`) +
`:BF`-marker split for butterflies within `S` (Section 2 of the report) — same logic
validated on LE, applied identically to every asset.

*7. Rescale prices* ÷1e9 → real decimals (confirmed, Section 4/6 of the report).

*8. Outputs — three tables, three different merge rules:*
- **Outrights → `Data/term_structure.parquet`.** Concat with the existing
  yfinance-forward-capture archive, `.unique(subset=["date","contract_symbol"],
  keep="first")` with Databento rows sorted first — **decided 2026-07-16: Databento
  wins** on the rare `(date, contract_symbol)` pair that exists in both sources
  (only possible right at the 2026-07-13/07-14 boundary). Licensed, complete CME data
  outranks yfinance's own documented "undocumented/unofficial behavior"
  (`DATA_SCHEMA.md` section 1). Forward-capture remains authoritative for every date
  after the boundary, where Databento has no coverage at all.
- **Calendar spreads → new `Data/term_structure_spreads.parquet`.** No existing
  counterpart to merge into (yfinance forward-capture has no spread-instrument
  tickers) — first write, not a merge. **Schema decided 2026-07-16, leg-decomposed:**
  `date, open, high, low, close, volume, asset, spread_symbol, near_contract_symbol,
  far_contract_symbol, near_expiry_year, near_expiry_code, far_expiry_year,
  far_expiry_code` — legs broken out explicitly (not just an opaque `LEZ26-LEV26`
  string) so a future carry signal can query "give me the spread between these two
  specific expiries" directly, matching how `contract_symbol` disambiguation works
  for outrights.
- **Butterflies → new `Data/term_structure_butterflies.parquet`.** Same treatment,
  3 legs instead of 2. Written since the data's already been paid for (`WORKFLOW.md`
  Phase 3 — a logged candidate, not a task) — kept, not consumed by anything yet.

*9. Manifest:* `Data/databento_transform_manifest.csv` — one row per (run, asset):
rows processed per instrument class, join-unmatched count, any anomalies found by
step 10, `status`. Same `run_date, ..., status, detail` shape as every other manifest
in this project.

*10. Validation, per asset, before merging:* the same checks the LE report ran by
hand — non-positive prices, zero-volume outrights, OHLC-relationship violations,
coverage vs. the real exchange calendar (`pandas_market_calendars`, already used in
`jobs/update_dashboard_summary.py`'s `ASSET_CALENDAR` mapping — reuse it, don't
redefine it). Logged to the manifest as warnings, not a hard gate — consistent with
how every other job in this project handles partial failures (log and continue, per
`jobs/update_data.py`'s own pattern), so one asset's anomaly doesn't block the other
31.

*11. Script structure:* one `transform_asset(asset_name, root, exchange) -> dict`
function, validated against LE first (the experimental case), then a thin runner that
either takes a specific asset name or auto-discovers every `UNIVERSE` asset with both
zips present in `Data/databento_raw/` and not yet in the manifest — so it naturally
picks up each asset as its pull arrives rather than requiring a "wait for all 33"
step, per the plan to repeat this per-asset once LE proves the design out.

**Built & validated against LE (2026-07-16) — two corrections to the plan above, found
by smoke-testing against real data rather than assumed from the single-example
exploration in `DATA_QUALITY_REPORT.md`.** `databento/transform_databento.py` now
implements all 11 steps; run against LE only (per the sequencing above), producing
35,143 outright rows (37 other assets' existing yfinance rows untouched, Databento
wins on the one real `(date, contract_symbol)` overlap per the merge rule), 108,885
spread rows, 22,102 butterfly rows. Validation step 10 matches
`DATA_QUALITY_REPORT.md` exactly: 0 join-unmatched rows, 0 non-positive prices, 0
zero-volume outrights, 0 OHLC violations, 16 missing sessions.

- **Step 5/6 correction: a spread/butterfly's own `maturity_year`/`maturity_month`
  identifies only one leg, not all of them — and not reliably the first-listed one.**
  `DATA_QUALITY_REPORT.md`'s recommendation ("needs the same `maturity_year`-based
  fix" as outrights) was an incomplete generalization from one example
  (`LEV6-LEZ6`, where the near/first-listed leg happened to be the one the
  definition row's maturity fields describe). At scale this doesn't hold: of LE's
  3,011 unique 2-leg spread instruments, 937 have the definition's maturity fields
  describing the *second*-listed leg instead of the first. **Fix, validated at full
  scale:** treat whichever leg's own (month code, single-digit year) matches the
  row's `maturity_month`/`maturity_year` as the "anchor" leg (confirmed via
  `activation`/`expiration` timestamps, not assumed); every other leg's year is the
  anchor's year plus the smallest non-negative decade offset that matches that leg's
  own digit. Still entirely `maturity_year`-derived, never the raw symbol string —
  the correction is to *which* leg that field describes, not a departure from
  Section 1's principle. Result: 3,011/3,011 spreads and 141/141 butterflies
  resolved with 0 unresolved legs; every LE-side leg (3,826 of them) cross-checked
  against LE's own independently-disambiguated 107-contract outright universe with
  0 mismatches. (The non-LE leg of an inter-commodity spread — see below — doesn't
  have that same independent cross-check available, since the other product's own
  outright universe hasn't been pulled; validated instead via
  `activation`/`expiration` consistency on a sample and the near≤far ordering
  holding for all 3,011 resolved pairs.)
- **Step 6 correction: not every 2-leg `instrument_class=S` row is a calendar
  spread.** 361 of LE's 807 unique raw spread symbols (~35% by unique symbol, 45%
  by row count) are inter-commodity pairs — `LEJ5-HEJ5` (Live Cattle vs. Lean
  Hogs), not `LEV6-LEZ6` (Live Cattle Oct vs. Dec). Only `HE` appears as a second
  root for LE; no third root observed. **Decision (direct instruction, 2026-07-16):
  keep both kinds**, generalizing the locked-in schema with explicit
  `near_root`/`far_root` columns rather than assuming a single root — the
  near/far (chronological, not string-position) framing still applies cleanly to
  a cross-commodity pair. Intent: feed the CTA dashboard and, longer-term, a
  synthetically-tracked version of these same spreads sourced from elsewhere once
  Databento's paid history isn't the only supply. Butterflies stayed single-root
  for every LE instance checked (0 cross-root), so their schema didn't need the
  same generalization.
- **ICE-cleared assets (`IFUS.IMPACT`) are explicitly out of scope for the
  spread/butterfly path above, not silently attempted.** Their `definition` schema
  carries explicit `leg_instrument_id`/`leg_raw_symbol`/`leg_count` fields — a
  structurally different (and, on the surface, easier) shape than CME/GLBX.MDP3's
  single-row-per-instrument-with-embedded-symbol-string shape the algorithm above
  was built and validated against. Outrights transform identically for both
  (the fields they need — `instrument_class`, `maturity_year`, `maturity_month` —
  are common to both schemas); ICE spread/butterfly rows are logged and skipped,
  not guessed at, pending its own validation pass.
- **Current state: validated against LE only.** A subsequent run against all 9
  then-ready assets (4 more CME grains/softs + 4 more ICE softs, whose zips had
  arrived mid-session) was reverted at the user's request to keep this pass
  scoped to LE — everything above describes LE's data specifically, not a
  multi-asset claim.

**Dashboard integration — bug found, plan logged (2026-07-16), building next.** Having
LE's data land in `term_structure.parquet` immediately broke an existing dashboard
query, found before it shipped to the scheduled job: `build_term_structure()` in
`jobs/update_dashboard_summary.py` took the latest observed price per
`(asset, contract_symbol)` with no active-contract filter — harmless while the archive
only held ~6 forward-capture contracts per asset, but once Databento added 101 more
(long-expired) LiveCattle `contract_symbol`s, a "today's curve" query for LiveCattle
came back with **107 points spanning 2010-2028 expiries**, most years-stale, would have
compared a 2010 close against a 2028 close for the contango/backwardation badge. **Fixed**
with `ACTIVE_CONTRACT_WINDOW_DAYS` (a contract with no print in this many days has
expired, not gone quiet) — tuned from an initial 21-day guess down to 10 after checking
LE's actual data: genuinely-active contracts were 0-6 days stale, a just-expired one
(`LEM26.CME`) was 16 days stale, so 10 cleanly separates them. Verified against the live
archive (107 -> 10 rows for LiveCattle, correctly reads Backwardation) and re-ran
`jobs/update_dashboard_summary.py` so the fix is live in `Data/dashboard_summary/`, not
just tested in isolation. All 37 other assets unaffected (none dropped by the tighter
window).

Plan for surfacing LE's new data in the dashboard (direct instruction, 2026-07-16) —
**scoped to LE only**, same discipline as the transform itself. Explicitly deferred until
every asset is integrated: reworking the term-structure page's now-partially-stale
"forward-capture only, not full history" caveat copy (still true for the other 37 assets,
not LE), and any cross-asset analysis (e.g. relative value across LE's own complex once
Feeder Cattle/Lean Hogs get their own Databento pulls) — this LE pass is the first of
many, not a one-off.

1. **Term Structure page — instrument-type toggle + per-instrument drill-down.** Add an
   Outrights / Calendar Spreads / Inter-Commodity Spreads / Butterflies selector. The
   existing "today's curve for an asset" view (now fixed) stays as the Outrights case.
   New: pick one specific instrument (a contract, or a spread/butterfly symbol) and see
   *its own* daily price history as a line chart — reading directly from
   `term_structure_spreads.parquet`/`term_structure_butterflies.parquet` filtered to
   that one instrument at render time, no new precompute needed (same pattern the page
   already uses for its manifest-based coverage chart, which also reads a raw `Data/`
   file directly rather than through `dashboard_summary/`).
2. **Contango/backwardation — a real history, not just today's badge.** New precompute:
   daily curve-shape classification across LE's full 2010-2026 date range (front vs.
   back close on each date the curve had >=2 active contracts), not just the current
   snapshot — forward-capture alone never had enough depth to make this meaningful.
3. **Richness badge for spreads/butterflies.** Today's price vs. its own trailing
   distribution (percentile/z-score), shown next to the selected instrument's price
   chart from item 1. Explicitly descriptive ("where does today sit in its own
   history"), not a trade signal — stays inside the QA-tool-not-signal-tool boundary
   `CLAUDE.md` sets for this dashboard. Computed at render time on the already-filtered,
   single-instrument slice (a few thousand rows at most) — lightweight display logic,
   not the kind of heavy signal-generation computation that has to go through the
   precompute job.
4. **Pipeline Health page — add the Databento transform manifest.** Alongside the 4
   existing scheduled-job manifests. Not subject to the same 30-hour staleness rule as
   the daily jobs, since `databento/transform_databento.py` is explicitly a one-time/
   occasional-run utility (`databento/`, not `jobs/`) — its entry reflects last-known
   per-asset status, not freshness.

Items 1-4 above are built and verified live (`streamlit run dashboard/app.py`).
**Deferred, per direct instruction:** reworking the term-structure page's asset-blanket
"forward-capture only, not full history" caveat copy — leave until every asset has
Databento depth, not just LiveCattle.

**Dashboard integration, round 2 (2026-07-16) — 5 more items, logged before building,
still scoped to LE only.** Prompted by actually looking at the round-1 build live:
carry — the literal stated reason Phase 4 prioritized capturing calendar spreads as
real quoted instruments rather than back-differencing outrights — has never actually
been computed anywhere, despite the data for it existing since this session's earlier
work.

1. **Carry / annualized roll yield**, for calendar spreads only (`near_root ==
   far_root` — inter-commodity pairs like LE-HE don't have a carry interpretation in
   the classic sense). Uses the exact formula from this same Phase 4 section:
   `Carry = (F_near - F_far) / F_near x 365 / (days between near and far expiry)`,
   with `F_near - F_far` taken directly from the spread's own quoted close (not
   back-differenced) and `F_near` joined from the near leg's own outright close in
   `term_structure.parquet` — days-between approximated at the month level (30.44
   days/month) since expiry is only known to month granularity in this schema, not
   the exact trading day.
2. **Curve steepness as a normalized %, not raw $.** A real flaw in round 1's Curve
   Shape History chart, found by looking at it live: it plots `back_close -
   front_close` in dollars, but LiveCattle's price level moved from ~$90 (2010) to
   ~$250 (2025-26) over the captured history, so a $5 gap means something completely
   different depending on when it occurs — not comparable across the panel. Fix:
   `(back_close - front_close) / front_close`.
3. **Synthetic-vs-real spread validation chart.** Plot the directly-quoted calendar
   spread against the back-differenced version (near leg close - far leg close, both
   from separate outright rows) for the same instrument, to actually show the
   leg-alignment-risk claim this project has made since the original LE exploration
   (`DATA_QUALITY_REPORT.md`) rather than leave it asserted but never demonstrated.
4. **Term-structure heatmap over time** (date x rank-from-front, colored by
   close-minus-front). A denser view than a single curve snapshot or a single
   front-minus-back line — shows the whole shape's evolution at once. Ranked by
   position along the curve (0=front, 1=next, ...) rather than absolute contract
   month, since absolute months roll off every day and would make the heatmap
   mostly-empty/diagonal.
5. **Liquidity/volume view** — a volume-over-time chart alongside the existing
   price chart for whichever instrument is selected (outright, spread, or
   butterfly), and switching the "most liquid first" instrument-picker sort key
   from row-count (a proxy) to total volume (the real thing) — both already
   collected in all three tables, currently unused beyond a `>0` filter.

Items 1-5 built. **Bug found live during review (2026-07-16), fixed same session: the
page selected spreads/butterflies by the raw `spread_symbol`/`butterfly_symbol` string —
the exact same decade-ambiguous identifier `DATA_QUALITY_REPORT.md` Section 1 already
documented and fixed for outrights (via `maturity_year`), and separately fixed correctly
in the transform's own output columns (`near_contract_symbol`/`far_contract_symbol`) —
but the dashboard page never used those disambiguated columns for selection, only for
display in the leg-detail expander.** Concretely: `LEM6-LEQ6` matches two real, unrelated
instruments — a Jun/Aug-2016 spread (310 rows, 2015-03-23 to 2016-06-30) and a
Jun/Aug-2026 spread (333 rows, 2025-03-04 to 2026-06-30) — found by the user looking at
the live chart and asking why there was a ~9-year straight diagonal line in the middle of
otherwise-noisy real data. It wasn't a data gap or an interpolation bug: grouping by the
raw string silently stitched the two unrelated real instruments into one series, and
Plotly drew a straight line connecting 2016's last point to 2026's first. Confirmed not
a rare edge case — 44 of ~128 unique LiveCattle butterfly symbols have the same collision,
and per the user, effectively every spread symbol does too (expected: the 16-year pull
window means most month-pairs occurring early in it recur with the same single-digit year
a decade later). **Fixed:** the page now keys instrument selection on the disambiguated
leg-contract-symbol tuple (`near_contract_symbol`/`far_contract_symbol`, or
`.../mid_contract_symbol/...` for butterflies), never the raw symbol string — confirmed
the fix isolates exactly one real instance per selection (LEM16.CME/LEQ16.CME alone now
returns 310 rows, 2015-03-23 to 2016-06-30, not the merged 643).

**Follow-on, logged not built (2026-07-16): the daily-forward-capture spread/butterfly
gap.** The calendar spreads and butterflies above are real, exchange-traded instruments
— not synthetic — but `yfinance` has no spread-instrument tickers at all, so once daily
forward-capture takes over from Databento at the 2026-07-13/14 boundary, real
spread/butterfly data stops; the free daily source structurally can't produce it.

**Free-source search (2026-07-16) — no free, systematically-automatable path found.**
Checked CME Group directly, Databento's live tier, and Barchart before concluding
this, not assumed:
- **CME Group's own free delayed quotes** (10+ min delay, includes spread/combination
  volume on their quote pages) stay free only for manual/human viewing. Automated or
  scheduled access — a script, a daily job — falls under CME's own **Non-Display Use
  policy** regardless of the delay, requiring a direct license (ILA). Their real-time/
  delayed API is separately priced at $0.50/GB + ILA fees. "Free" and "automatable"
  turn out not to overlap here.
- **Databento's live/streaming tier** — the natural extension of the historical batch
  pull already in use — had usage-based pricing discontinued April 2025; it's now a
  **$179/month minimum subscription** (Standard plan), not an incremental top-up of
  the $125 one-time credit already spent.
- **Barchart** has a well-structured page for CME calendar spreads and butterflies
  (`barchart.com/futures/quotes/{root}*0/futures-spreads`, explicit `SP`/`BF` spread
  types) that looks free in a browser, but their real API (OnDemand) starts at
  $500/month, and their Terms of Use restrict automated/bot access to the free page —
  a real ToS line, not just a paywall to route around.

**Conclusion: three honest options, no free fourth one.**
1. Pay for ongoing access (Databento live, $179/mo, or CME's own API) — a real
   subscription decision, not currently budgeted.
2. **Periodically re-run the historical batch pull** (the product already in use,
   e.g. quarterly) to refresh the spread/butterfly archive with a small one-time cost
   each time rather than a monthly subscription — not daily/real-time, but free of new
   ongoing commitment and reuses `databento/transform_databento.py` unchanged.
3. The computed/back-differenced proxy from the two outright legs already captured
   daily (`near_close − far_close`) — free, but the weaker, leg-alignment-risk-prone
   proxy `DATA_QUALITY_REPORT.md` already found inferior to a real traded spread quote.
   If ever built, must be clearly labeled computed/derived, never presented as
   equivalent to a real traded price (`CLAUDE.md` Rule 4) — e.g. a distinctly-named
   `term_structure_spreads_computed.parquet`, never merged into the real
   Databento-sourced spread table.

Not decided yet which of the three to pursue, if any — logged for when it matters
(once Phase 2d/carry work actually needs fresher spread data than the one-time
historical pull provides).

**Other paid alternatives, ruled less attractive than Databento (kept for reference):**

| Source | What you get | Cost |
|---|---|---|
| **Interactive Brokers API** (`ib_insync`) | Historical data for specific contract months via the TWS API. | **Corrected from earlier research — not actually free.** Needs a funded account (~$500 minimum in most cases) and US Futures market-data permissions requiring $30/month in generated commissions. Free paper accounts reportedly cannot pull historical data at all. |
| CME DataMine | Official End-of-Day/settlement data direct from the exchange. | **$105/month per exchange group up to $2,100/month** for the full End-of-Market-Summary package (confirmed pricing). |
| Norgate Data | Back-adjusted *and* individual-contract data with documented roll schedules, ~100 futures markets / 11 exchanges. | 6- or 12-month terms only (no monthly option); exact price not confirmed — check `norgatedata.com/prices.php` directly. |
| EIA (energy subset only) | Free daily/weekly settlement data for energy contracts — same EIA API already proven working in the port congestion project. | **Discontinued** — the EIA NYMEX futures-price series stopped publishing after 2024-04-05; not viable for ongoing/live use, only as a possible historical-backfill source. |

**Recommendation (updated 2026-07-14):** the free yfinance path is validated and running
(forward-capture only, per above). Historical backtesting of carry needs Databento's
batch pull, but the free credit has expired (see above) — parked pending the
reinstatement request. If reinstatement fails, the fallback is paying out of pocket
(~$50-150 estimated, 2010-2026 only) rather than a subscription vendor, since it's a
one-time cost either way and still cheaper than any alternative in the table above.

**Signal build planned, 2026-07-22 (per direct instruction — carry moves ahead of
cointegration in build order, logged before writing any code):** full design recipe
in `references/carry_implementation_recipe.md`, source Koijen-Moskowitz-Pedersen-Vrugt
"Carry" (*JFE* 2018). Key decisions already made there, summarized here so this
section stays the authoritative status pointer:

- Reuses the exact formula this same Phase 4 section already derived from the
  LiveCattle prototype (`Carry = (F_near - F_far)/F_near × 365/(days between near and
  far expiry)`, real quoted spread close over the near leg's own outright close) —
  not re-derived from the paper independently.
- **No spot price data needed, for any asset class** — the paper's own "synthetic
  spot" construction (nearest-to-expiry futures standing in for an unreliable/
  unavailable real spot print, explicit in its commodities section) is the same idea
  this project's adjacent-contract formula already embodies, and it generalizes
  uniformly here because every asset class in this universe trades via futures with
  multiple listed maturities.
- **Two Books, not one**, mirroring the paper's own two complementary constructions
  (cross-sectional carry trade and carry timing) and this project's own precedent
  (TSMOM vs. XSMOM being separate, not competing, signals) — cross-sectional reuses
  `short_term_reversal.py`'s within-sector demean machinery, carry timing reuses
  `vol_targeted_sign_signal`, no new position-construction logic needed for either.
- **Real caveat, flagged not hidden:** `term_structure_spreads.parquet` is a one-time
  Databento pull frozen at 2026-07-13 (daily forward-capture only gets outright
  prices, no spread tickers via yfinance) - the backtest is fully real-data-grounded
  through that date, but a live/forward-going version needs one of three
  already-logged options (subscription, periodic re-pull, or the labeled proxy below)
  resolved first.

**Decisions confirmed 2026-07-22, discussed live before settling (not defaulted):**

1. **Forward-data gap** — deferred. Backtest capped at 2026-07-13 now; the
   live-continuation path gets decided only if/when the backtest result makes that
   decision worth making.
2. **Cross-sectional tier** — sector-scoped (`data.sectors.SECTORS`), consistent with
   short-term reversal. A pooled global rank across all 37+5 assets is not built now;
   a vol-weighted cross-sector combination (the paper's own "global carry factor") is
   a candidate later portfolio-construction step, not part of the first pass.
3. **ICE softs (Coffee, Sugar, Cocoa, Cotton, OJ — zero real spread data)** — build
   the back-differenced proxy for these 5, not exclude them, but only after
   quantifying the proxy's real error first (below) rather than deciding on the "leg-
   alignment risk" assertion alone.

**Proxy error, measured directly (2026-07-22), not assumed:** compared the real
quoted spread against the back-differenced proxy (`near_outright_close -
far_outright_close`, two separately-timed prints instead of one simultaneous spread
quote) for four CME-cleared assets that have both, restricted to the front calendar
spread specifically (far-dated/thin pairs showed much worse, occasionally
sign-flipping errors and aren't representative of what carry actually trades).
Error as % of the near leg's own price:

| Asset | Median abs error | 90th pct | 99th pct | Max observed |
|---|---|---|---|---|
| LiveCattle | 0.04% | 0.24% | 0.82% | 3.2% |
| Corn | 0.06% | 0.23% | 1.06% | 6.3% |
| Gold | 0.07% | 0.39% | 1.41% | 6.5% |
| WTI Crude | 0.08% | 0.48% | 1.72% | 29.0% |

Typical case is fine (median error small - both legs usually trade actively enough
near the close); the tail is real (99th pct 0.8-1.7%, occasional multi-percent
outliers) and matters more than it looks because carry's `365/(days to expiry)`
multiplier (roughly 6-12x for a 1-2 month near/far gap) amplifies a spread-level
error into a larger error in the final annualized carry figure. **This measurement
is an extrapolation, not direct evidence for the ICE softs themselves** - there's no
real spread data for Coffee/Sugar/Cocoa/Cotton/OJ to validate against at all; softs
are generally thinner markets than WTI/Corn/Gold, so the true proxy error there is
plausibly comparable or worse, not better. Full detail: `references/carry_
implementation_recipe.md`. **Every proxy-derived carry value must carry an explicit
`is_proxy=True` flag through to the dashboard, never silently blended with the 37
real-quote assets** - CLAUDE.md Rule 4's labeling discipline, applied concretely.

**Standard process for this build, same as every prior signal family - noted
explicitly since it's a multi-session build:** pure functions in `src/data/` and
`src/signals/carry.py` first, with tests → `research/carry.py` driver script, run
against real data, results sanity-checked → only then `dashboard/pages/
11_carry_performance.py` → CLAUDE.md/WORKFLOW.md updated with the actual result →
full `pytest tests/` run clean. Dashboard integration happens *after* the signal is
built and tested, not before - same order as momentum, breakout, crossover, and
short-term reversal.

**Built and backtested 2026-07-22 (first attempt) - honest weak/negative result,
following the standard process above exactly. Corrected same day, see below - kept
here as the real record of what was tried first and why it changed, not overwritten.**

- `src/data/term_structure.py` - `load_front_contract_symbols()` reads
  `front_contract_symbol` straight off `Data/continuous_futures.parquet` (so carry's
  "front" matches every other signal's roll-rule definition, no second competing
  definition); `real_spread_carry()` restricts `term_structure_spreads.parquet` to
  the FRONT calendar spread specifically per date (near_contract_symbol == that
  date's front, nearest-expiry far leg where several are quoted the same day - not
  just any spread, since far-dated/thin pairs aren't representative of what carry
  actually trades); `proxy_carry()` walks `data.continuous_curve.build_contract_chain`
  forward from front to the nearest next-listed contract with a real outright print
  that day, for the 5 ICE softs. `build_carry_panel()` ties both paths together plus
  an asset-indexed `is_proxy` flag. 9 unit tests (`tests/test_term_structure.py`),
  including a synthetic-fixture reproduction of the nearest-far-leg selection and the
  proxy's forward-scan-skips-unavailable-candidate behavior.
- `src/signals/carry.py` - `cross_sectional_carry_signal()` (thin wrapper over
  `signals.short_term_reversal.cross_sectional_demean`, deliberately no sign flip -
  above-sector-average carry is already the long trade, unlike reversal's bet against
  the move) and `carry_timing_signal()` (`sign(carry - carry.expanding().mean()) /
  vol`, reusing `signals.transforms.vol_targeted_sign_signal`). Carry timing's "own
  historical mean" uses an **expanding**, not fixed rolling, window - a rolling
  lookback would be an extra, undiscussed hyperparameter with no basis in the paper
  or recipe; expanding avoids inventing one. 4 unit tests
  (`tests/test_carry.py`).
- **Real-vs-proxy error, re-measured live against current data
  (`research/carry.py`'s own `proxy_error_report()`), not just cited from the
  2026-07-22 design session:** median 0.01-0.04%, 99th pct 0.15-0.39%, max 1.3-3.6%
  (LiveCattle/Corn/Gold/WTI Crude) - same qualitative shape as the design-session
  table (small typical error, real tail) but not identical numbers, expected since
  this is an independent live re-measurement, not a copy-paste. Full table:
  `Data/research/carry/proxy_error_measured.csv`.
- **Universe actually used: 38 ADV-filtered assets** (same floor as every other
  signal), of which 33 get real-quote carry and 4 get the ICE-softs proxy (Coffee,
  Sugar, Cocoa, Cotton - OrangeJuice itself fails the ADV floor, so it never reaches
  either path here). ICE softs' term-structure history starts 2023-08 at the
  earliest (Cocoa: 2024-07-15) - well after `TRAIN_END` (2019-12-31), so their
  per-asset train-period Sharpe is correctly `NaN` (zero train-period observations,
  the same known, already-documented gap `backtest/performance.py`'s own `MIN_OBS`
  guard exists for) - not a bug, an expected artifact of when that data starts.
- **Result, gross/net Sharpe (train / validation / test, test capped at
  2026-07-13):**
  - Cross-sectional carry: **-0.31 / -0.31 / -0.34 gross → -1.01 / -0.62 / -0.66
    net.** Negative in every period, gross and net alike - not just a cost story.
  - Carry timing: **0.16 / -1.03 / 0.06 gross → -1.26 / -2.20 / -1.09 net.** Weakly
    positive gross in train/test, sharply negative in validation, and net-of-cost
    Sharpe is deeply negative in every period.
  - **Turnover measured at ~120-125x annualized for both Books** - in short-term
    reversal's high range (~100-360x), well above breakout's ~50-60x - which is why
    the net-of-cost picture is so much worse than gross for both Books.
- **Honest conclusion:** despite the design recipe's own "best-matched signal built
  so far" framing (real data built specifically for this signal, no proxy needed for
  most of the universe, one uniform formula across every asset class), the measured
  result does not bear that out - carry is the **second** outright-unprofitable-
  net-of-cost signal family after short-term reversal, not the strong result hoped
  for. Reported as found, not tuned to look better (`CLAUDE.md` Rule 1/2).
- **Dashboard** (`dashboard/pages/11_carry_performance.py`, registered in
  `dashboard/app.py` as the 5th "Strategy Performance" page): Book selector
  (cross-sectional / carry timing) + gross/net toggle, per-asset tearsheet with an
  explicit "Is Proxy" column, equity curves with proxy assets rendered in a distinct
  line style/color - never a silent merge. Verified exception-free via
  `streamlit.testing.v1.AppTest` across all 4 Book × cost-view combinations (no
  browser tool available in this environment - the project's substitute for live
  UI verification).
- Full `pytest tests/` run clean after this build: 92 passed.

**Rebuilt 2026-07-22, same day, to match the paper exactly - per direct instruction,
after being asked why the first attempt didn't work.** Rather than accept "carry
doesn't work here" from the first attempt, `references/Carry.pdf` was read directly
(not relied on from memory or the recipe doc's summary) to check the paper's actual
construction against what was built. Real gaps were found, not just a costs/tuning
question:

1. **Rebalancing: daily → monthly.** The paper states plainly "the portfolio is
   rebalanced every month" (p. 207). The first build's `frequency="daily"` (copied
   from short_term_reversal without re-deriving for carry's own slower-moving
   economics, the same CLAUDE.md Rule 7 lapse logged elsewhere in this project) was
   confirmed, empirically, to be the dominant driver of the ~120x turnover: testing
   daily/weekly/monthly side by side on the same signal showed turnover falling
   120x→15x→7x while gross Sharpe barely moved - the frequency was destroying returns
   through cost, not through the signal itself being wrong.
2. **Cross-sectional weighting: raw-magnitude demean → rank (Eq. 19).** `w_i =
   rank(C_i) - (N_t+1)/2`, not `signals.short_term_reversal.cross_sectional_demean`'s
   raw-magnitude subtraction. The paper is explicit about why: a signal-weighted
   scheme "can place considerable weight on the extremes... using ranks instead of
   the signals themselves... insulates weights from outliers." The first build's raw
   demean let whichever asset had the single biggest RAW carry level (e.g. LeanHogs,
   std ~0.61, vs. FeederCattle's ~0.13 in the same Livestock sector) dominate its
   sector's trade every day - the identical failure mode short-term reversal's own
   build already found and fixed once (vol-standardize before demeaning), not carried
   over to carry the first time. Measured directly: switching raw-demean→rank at
   monthly frequency moved train Sharpe from -0.32 to +0.33 on the same underlying
   carry data - the difference between "doesn't work" and "works," on this one
   change alone.
3. **carry1m and carry1-12, not a single unsmoothed reading (footnote 14).** The
   paper itself flags carry1m as high-turnover and builds carry1-12 - a trailing
   12-month moving average of the raw carry LEVEL, computed before any cross-
   sectional step - specifically to cut turnover (~50% on average) while keeping
   ~92% of the Sharpe. Both are now built, mirroring the paper's own Table 2 Panels A
   and C, reusing `backtest.engine.holding_period_positions` directly (built
   originally for momentum's overlapping-vintage blending - it turns out to be
   exactly carry1-12's own construction with `holding_months=12`).
4. **Carry timing's reference point: per-asset individual expanding mean → sector-
   pooled (Eq. 24, Section 3.7).** The paper's `C̄` is 0 or "the average carry across
   all securities in a given asset class up to that point in time" - one shared
   threshold per sector, not each security's own history. The first build's
   individual-expanding-mean construction was a misreading of the recipe doc's own
   paraphrase, not the paper itself - corrected after reading Section 3.7 directly.
5. **Carry timing's sizing: vol-targeted → simple ±1 direction.** The paper's own
   weights (`z_t(2·1(C_t^i - C̄ > 0) - 1)`) are not vol-scaled at all - `z_t` only
   fixes aggregate ($2) exposure. Replacing `signals.transforms.vol_targeted_sign_
   signal` with a plain `np.sign()` is a deliberate, logged exception to CLAUDE.md
   Rule 5's general "binary underperforms continuous/vol-scaled in this universe"
   finding - made specifically to reproduce this one paper's published construction
   exactly, per direct instruction, not a quiet reversal of that finding elsewhere.

**Result of the fix:** turnover collapsed to **~0.7-3.3x annualized** across all four
specs (cross_sectional_carry1m 3.3x, cross_sectional_carry1_12 0.66x,
carry_timing_zero 1.68x, carry_timing_mean 2.01x) - net-of-cost Sharpe is now nearly
identical to gross everywhere, confirming the first build's net-of-cost wipeout was
overwhelmingly a rebalancing-frequency problem, not a transaction-cost-realism one.

**Four parallel specs, gross Sharpe (train / validation / test, test capped at
2026-07-13):**
- `cross_sectional_carry1m`: -0.01 / -0.58 / -0.68
- `cross_sectional_carry1_12`: **+0.33** / -0.36 / -0.06
- `carry_timing_zero`: -0.30 / -0.98 / **+0.33**
- `carry_timing_mean`: +0.02 / -0.56 / **+0.26**

**Honest conclusion: genuinely mixed, not a clean win, but a real, structurally
sensible pattern - not noise.** Cross-sectional carry1-12 has a real positive TRAIN
Sharpe but doesn't generalize to validation/test. Both carry-timing variants are
weak-to-sharply-negative in train/validation (validation spans 2020-2021, the COVID
shock - consistent with the paper's own documented finding that carry strategies
underperform during global recessions/business-cycle stress across every asset
class it studies) but turn solidly positive in test (2022 onward - plausibly the
global rate-divergence/inflation-shock regime being unusually informative for carry
specifically). No single spec is robustly positive across all three periods - this
is the honest result of matching the paper exactly, reported as found, not tuned
(`CLAUDE.md` Rule 1/2).

- `src/signals/carry.py` rewritten in place: `cross_sectional_carry_signal` (now
  rank-based), `pooled_expanding_carry_mean` (new - the sector-pooled C̄), `carry_
  timing_signal` (now takes `groups` + `reference` instead of `vol` + `target_vol`),
  `sampled_carry` (new - thin, clearly-named wrapper around `backtest.engine.
  holding_period_positions` for carry1m/carry1-12). 8 unit tests
  (`tests/test_carry.py`, rewritten), including a synthetic check that rank-weighting
  treats "lowest of three" identically regardless of how close the actual values are
  (unlike a magnitude-based demean) and that the pooled mean weights every
  observation equally, not every day equally.
- `research/carry.py` and `dashboard/pages/11_carry_performance.py` both rewritten
  for the 4-spec structure and monthly cadence; dashboard re-verified exception-free
  via `streamlit.testing.v1.AppTest` across 6 spec/cost-view combinations.
- Full `pytest tests/` run clean after the rebuild.

---

## Phase 4b — Cross-sectional momentum (XSMOM) 🟢 built 2026-07-22

**Not one of the original six signal families in this project's scope (`CLAUDE.md`)
- added per direct instruction as a deliberate expansion**, alongside starting the
portfolio-construction pass (Phase 7) ahead of cointegration (still the next item on
the original six-family list, deferred again, not dropped).

Source: Asness, Moskowitz & Pedersen, "Value and Momentum Everywhere" (*Journal of
Finance*, 2013), `references/Value and Momentum Everywhere.pdf` (user-provided, read
directly) - full recipe in `references/xs_momentum_implementation_recipe.md`.
TSMOM's explicitly distinct sibling (both papers frame time-series and
cross-sectional momentum as correlated but different phenomena) and the third
cross-sectional signal in this project after short-term reversal and carry.

**Recipe, matched exactly:**
- `MOM2-12` (12-month return, skip most recent month, Jegadeesh-Titman convention) -
  applied uniformly even though the paper itself says the skip "is not necessary"
  for liquid futures, "to maintain uniformity across asset classes," explicitly
  accepting a *conservative* number as a result ("momentum returns for these asset
  classes are in fact stronger when we don't skip the most recent month"). Computed
  on the month-end-resampled back-adjusted close, not a daily day-count
  approximation.
- Rank-weighted (Eq. 1: `rank(S_i) - mean_rank`), within sector
  (`data.sectors.SECTORS`) - algebraically identical to carry's Eq. 19. Extracted
  into a new shared `signals.transforms.cross_sectional_rank(features, groups,
  min_group_size)`, since this is now needed by two independent signal families
  (`CLAUDE.md` Rule 6: extract once a second consumer exists) -
  `signals.carry.cross_sectional_carry_signal` is now a thin wrapper over the same
  function, no behavior change (all existing carry tests still pass unchanged).
- No vol-scaling (the paper's own headline construction - ex-ante-vol weighting is
  reported only as a robustness check, footnote 10).
- **One Book, not a multi-horizon grid** - the paper is explicit: "we are not
  interested in coming up with the best predictors of returns... rather to
  maintain a simple and fairly uniform approach... that minimizes... data
  snooping." A genuine departure from TSMOM/breakout/crossover/reversal's own
  multi-spec Book-count decisions, made because this paper gives no comparable
  precedent for testing several lookbacks in parallel.
- Monthly rebalancing (the paper's data is monthly throughout).

**Result:** measured annualized turnover ~3.3x (low, in momentum's range, net ≈
gross). Train/validation/test gross Sharpe **-0.34 / -1.39 / +0.04** (net: -0.39 /
-1.42 / -0.01) - weak-to-negative overall. Validation (spanning the 2020 COVID
crash) is sharply negative, consistent with the well-documented "momentum crash"
phenomenon around violent, V-shaped market reversals (trend-followers, including
cross-sectional momentum, are known to suffer disproportionately in exactly this
kind of regime). Reported as found, not tuned (`CLAUDE.md` Rule 1/2).

- `src/signals/xs_momentum.py` (`build_momentum_feature`,
  `cross_sectional_momentum_signal`) + 4 unit tests (`tests/test_xs_momentum.py`).
- `research/xs_momentum.py` - same conventions as every prior research script.
- `dashboard/pages/12_xs_momentum_performance.py`, registered in `dashboard/app.py`
  as the 6th "Strategy Performance" page (gross/net toggle only - no spec selector,
  since there's just one Book) - verified exception-free via `streamlit.testing.v1.
  AppTest`.
- Full `pytest tests/` run clean after this build.

---

## Phase 4c — Value 🟢 built 2026-07-23

**Not one of the original six signal families - added per direct instruction as
a further deliberate expansion, following XSMOM's own precedent** (same paper,
different factor). Sequenced after the Portfolio Construction/Risk/Macro-Data
work specifically because it needed two additional data inputs (yield curve,
CPI) that didn't exist until that work built them - not an arbitrary ordering.

Source: Asness, Moskowitz & Pedersen, "Value and Momentum Everywhere" (*Journal
of Finance*, 2013), `references/Value and Momentum Everywhere.pdf`, read
directly (Section I.B) rather than from memory. See that same session's
groupings-comparison and data-sourcing discussion earlier in this phase
(bonds/CPI) for the full research trail behind this build.

**Recipe, matched exactly** - no book-value measure exists for futures the way
it does for equities, so the paper substitutes the negative of the past 5-year
return everywhere, with two further asset-class refinements:
- **Commodities & equity indices**: `log(price_5yr_ago / price_today)` - LOG,
  matching the paper's own stated formula exactly, not a simple-pct-change
  approximation.
- **Bonds** (`US_2Y`/`US_5Y`/`US_10Y`/`US_30Y`/`UltraBond`): the 5-year CHANGE
  IN YIELD, maturity-matched from `Data/Yield_Curve_6M_to_30Y.csv` (already
  collected, previously unused) - richer than the paper's own single-10-year-
  yield-per-country simplification, since this project has the full curve.
  UltraBond has no exact maturity match and is mapped to 30Y as the closest
  available point - a labeled approximation.
- **Currencies** (the 7 FX-group members): PPP-adjusted real FX value - nominal
  5-year FX return minus the relative CPI inflation differential vs. the US,
  using the newly-built `Data/cpi_level_index.csv`. Not forward-filled where a
  country's CPI is stale/missing - those assets drop out of the cross-section
  on affected dates via `cross_sectional_rank`'s own `min_group_size` gating.
- Rank-weighted (reuses `signals.transforms.cross_sectional_rank` directly, the
  same shared function carry/XSMOM use), no vol-scaling, **one Book, not a
  multi-lookback grid** - same three departures already established for
  carry/XSMOM, for the same reason (the paper's own stated anti-data-snooping
  design). Ranked within this project's own sectors (`data.sectors.SECTORS`, 9
  groups), not the paper's coarser 5-asset-class split (which pools all 27
  commodities into one factor) - a real, considered difference, not an
  oversight (see this same session's groupings-comparison note).

**One real bug found and fixed live, before shipping, via a dedicated
regression test - not discovered after the fact**: the final daily-reindex step
in the FX PPP function (`feature.reindex(fx_close.index).ffill()`, unbounded)
would silently replay the last GOOD monthly PPP value forever once a country's
CPI went missing - reindex+ffill can't distinguish "haven't reached the next
real month yet" from "the source has gone missing," so an unbounded ffill
directly contradicted this signal's own "don't fabricate stale data" design
(the same convention already used for carry's ICE-softs proxy). Fixed with
`ffill(limit=35)` (a bit more than one month - bridges the ordinary monthly-to-
daily gap without indefinitely propagating a genuinely missing observation).
Verified directly: an unbounded version showed a stale non-NaN value 100 days
after a simulated CPI gap; the fixed version correctly shows NaN at that same
point.

**Result: weak-to-negative, reported as found** - measured annualized turnover
~2.7x (low, consistent with a slow 5-year-lookback signal). Train/validation/test
gross Sharpe **-0.02 / +0.12 / -0.68** (net: -0.05 / +0.08 / -0.70) - essentially
flat in train, mildly positive in validation, clearly negative in test. Per-asset
Sharpe (train) is genuinely mixed across all three asset-class treatments (FX
mostly positive: GBPUSD +0.99, AUDUSD +0.49, CADUSD +0.38; bonds mildly and
fairly uniformly negative, -0.03 to -0.19, no outlier; commodities/equities
widely dispersed, Dow +0.99 to Palladium -0.89) - nothing points at one
construction being broken rather than the signal itself being weak, and a
value-factor drawdown across the 2022+ regime (aggressive rate hikes, momentum-
dominated markets) is a well-documented real phenomenon, not just this
universe. **Coffee/Cocoa/Sugar/Cotton are 100% NaN** - real trusted price
history for these 4 softs only starts 2023-2024 (`DATA_SCHEMA.md`'s trusted-
since masking), short of the 60-month lookback this signal needs - a real,
labeled data-coverage gap, not a bug, that resolves itself as more history
accumulates.

- `src/data/macro.py` (new - `load_yield_curve`, `load_cpi`, raw loaders,
  signals/ modules take already-built panels as arguments per this project's
  usual separation, same pattern as `data.term_structure`).
- `src/signals/value.py` (`commodity_equity_value_feature`, `bond_value_feature`,
  `fx_ppp_value_feature`, `build_value_feature`, `cross_sectional_value_signal`)
  + 8 unit tests (`tests/test_value.py`) - formula correctness against hand-
  computed values for all three asset-class treatments, sign-convention checks,
  the missing-CPI-doesn't-leak-forward regression test, and a routing test
  confirming each asset gets exactly one treatment (no double-counting).
- `research/value.py` - same conventions as every prior research script.
- `dashboard/pages/16_value_performance.py`, registered in `dashboard/app.py` as
  the 7th "Strategy Performance" page (gross/net toggle only, no spec selector -
  verified exception-free via `streamlit.testing.v1.AppTest` across cost-view
  and ranking-metric changes).
- Full `pytest tests/` run clean after this build (141 passed, up from 133).

---

## Phase 5 — Portfolio construction: thin vertical slice ⚪

**Scaffolding built ahead of schedule (2026-07-15)** — `src/portfolio/`
(`optimizer.py`, `book.py`, `allocator.py`) and `src/signals/combine.py`, adapted from
the retired Cross Asset Stat Arb Engine and evaluated piece-by-piece (see
`cleanup.md` section 3 for the full keep/adapt/discard breakdown). This is scaffolding
only — not yet exercised against a real signal, still ⚪ for the actual phase work
below. Built early, ahead of the normal "2+ signal families exist" trigger
(`CLAUDE.md` Rule 6), because it's porting an already-evaluated engine rather than
guessing at a shape from scratch — logged as a deliberate exception, not a precedent.

Turn the best-validated Phase 2 signal(s) into an honest, complete mini-portfolio — not a
bigger signal library.

- Per-asset position sizing from a risk budget divided by Yang-Zhang volatility (already
  built).
- A single portfolio-level volatility target (scale gross exposure to, say, 10%
  annualized).
- Equal risk-budget weighting across the universe. **No covariance optimizer yet** —
  that's Phase 7, once there's more than one signal family to combine.
- Expected output: signal → vol-scaled position → portfolio-level vol target → backtest
  with turnover reported. This is literally "volatility targeting," the first
  unimplemented half of the resume bullet, and it reuses the volatility estimator that's
  already done.

---

## Phase 6 — Transaction cost realism 🟡

**Pulled forward, 2026-07-21 (deliberate exception, same pattern as the portfolio
scaffolding and `src/` split before it):** a simple, honest cost estimate is being added
to the single-signal backtest engine now, not held for the full Phase 5/7 portfolio
pipeline. Rationale: any high-turnover candidate signal (short-term reversal is next
on the roadmap after breakout/crossover/cointegration) needs an honest net-of-cost
Sharpe to be evaluated on its own merits, before a portfolio layer exists to net it out
— momentum's own reported 0.402 test Sharpe is still gross-of-cost today. The
turnover-*penalized optimizer* (trading off expected return against cost inside the
objective function, the pattern the retired stat-arb engine's turnover-penalized MVO
already validated) stays Phase 7's job; only a return-drag estimate is being added now.

**Design:**
- New module `src/backtest/costs.py` — pure functions, no optimizer dependency, same
  convention as every other signal/backtest module:
  - `turnover(positions)` — `|Δposition|` per asset per day.
  - `transaction_cost_drag(positions, cost_bps)` — turnover × each asset's own assumed
    one-way cost (bps), summed across the book into a daily return drag.
- `backtest_signal`/`backtest_signal_per_asset` (`src/backtest/engine.py`) get one new
  optional parameter, `cost_bps=None` (default preserves existing behavior/tests
  exactly) — turnover is computed on the *final*, already-normalized position array
  used to multiply against returns, not a separate recomputation.
- Cost assumption: **liquidity-tiered, derived from each asset's own trailing ADV**
  (`data.universe.get_liquid_universe` already computes this — reusing real data
  already in hand rather than inventing per-instrument numbers from nothing).

**Sourcing check (2026-07-21, before implementing):** searched for a real institutional
bps-by-asset-class table before assuming one. Findings, honestly caveated:
- Moskowitz-Ooi-Pedersen (2012) itself — the paper this whole signal reproduces — is
  **entirely gross-of-cost**; it contains no transaction cost table at all, so it isn't
  a usable source here despite being the natural first place to look.
- No authoritative, citable per-asset-class institutional bps table was found. What
  turned up is anecdotal but directionally useful: a GSCI-tracking commodity futures
  basket costs **>100bp/year** in transaction costs vs. **<10bp/year** for S&P 500
  futures (a 10x+ gap between commodities and the most liquid financials); e-mini S&P
  front-month bid-ask ≈ 1 tick (~0.5bp of notional) vs. its own far-month contract at
  10-12 points (~15-20x wider); Palladium spreads can reach 8 points in thin trading.
- **Conclusion: the specific bps-per-tier values below are a labeled placeholder
  assumption, not a measured fact** — same "label missing/uncertain rather than fake
  it" discipline this project already applies to carry (CLAUDE.md Rule 4). Tiering
  structure (liquid financials cheapest, thin ags/softs/metals most expensive) is
  well-supported directionally; the exact numbers should be recalibrated against a real
  broker cost sheet or fill-level data if that ever becomes available.

| Tier | Basis | Illustrative one-way cost |
|---|---|---|
| 1 — very liquid | Rates, FX majors, equity index futures | ~1bp |
| 2 — liquid commodities | Gold, WTI/Brent, Copper, mainstream energy/metals | ~3bp |
| 3 — moderate | Grains, most softs, livestock | ~6bp |
| 4 — thin | Lowest-ADV quartile of the included universe (Platinum/Palladium-type liquidity) | ~10bp |

Tiers assigned by ADV quartile within the ADV-filtered universe itself (not by
hardcoded instrument name), so the assumption travels if the universe or liquidity
conditions change.

---

## Phase 7 — Multi-signal portfolio construction ⚪

Once 2+ signal families exist (Phase 2 complete), exercise `src/portfolio/` (built
ahead of schedule, see Phase 5) for real: replace equal-risk-budget sizing with the
covariance-based optimizer — Ledoit-Wolf covariance on vol-scaled returns (built on
whatever return series the signals actually trade, not a beta-neutralized proxy — see
`cleanup.md` section 3 for why that distinction mattered in the retired project), a
turnover-penalized objective, one Book per signal family, an Allocator that combines
them. `src/signals/combine.py` gives the option to blend signals into one Book instead
of (or alongside) combining separate Books — a research decision, not an architectural
one. This is also the natural point to slot in a `build_congestion_signal` Book if the
port congestion project has validated a signal by then (see "Integration" in
`CLAUDE.md`), and to empirically test whether the DCC-GARCH package evaluated in
`cleanup.md` section 3 improves on rolling Ledoit-Wolf for this 41-asset signal
universe (the `get_data.ipynb` panel — SOFR has no OHLCV series to build a signal
from, see Phase 0).

**Book-count decisions, logged 2026-07-21 (before any optimizer is actually built, so
these are pre-committed, not chosen by looking at Phase 7 results):**

- **Breakout is 2 Books, not 1** — System 1 (20d/10d) and System 2 (55d/20d), fed in
  separately, not blended into a single averaged signal beforehand and not reduced to
  "whichever tested better." Rationale (from the discussion this session): picking one
  system because it backtested better would be the exact same spec-selection
  look-ahead Rule 1/2 already prohibits elsewhere, just relocated to the
  portfolio-construction stage; the historical Turtles traded both simultaneously as
  policy, never picking one; and any redundancy/correlation between the two is exactly
  what the Ledoit-Wolf covariance-based optimizer above already exists to handle
  properly (shrink the redundant pair's combined sizing), not something to pre-filter
  by hand. Keeping them as two separate Books (rather than pre-blended via
  `signals.combine.combine_alphas`) preserves that correlation structure for the
  optimizer to actually see, rather than hiding it behind an upfront average.
- **Momentum expands from 1 Book to 3, one per horizon** — **3mo, 12mo (unchanged
  headline spec), 24mo** (revised 2026-07-21 from an initial 1/12/36 proposal, per
  direct feedback - see below). Chosen from Moskowitz-Ooi-Pedersen's own
  already-published grid points (not invented), not by scanning which of the 8 grid
  lookbacks scored best on train Sharpe. Real-world precedent: short/medium/long
  multi-horizon trend ensembles are standard industry practice (AHL, Winton and
  similar CTAs blend trend speeds deliberately, for robustness across regimes where
  different horizons dominate) - distinct from breakout's situation (an
  externally-fixed, closed 2-system set with real trading precedent) but grounded the
  same way: decided for reasons independent of this project's own backtest results.
  **Revision history, both rounds checked against real data, not just argued
  abstractly:**
  - *Round 1 (1/12/36) rejected*: 1-month lookback paired with 1-month holding is too
    noisy a combination to make sense as a standalone spec. 36mo was checked against
    actual data (not just judgment): with ~11.8 years of train data, warmup alone
    consumes 25.4% of it - but the sharper problem is that a 36-month lookback only
    spans **~3-4 non-overlapping cycles** in train, a very small effective sample to
    trust that horizon's behavior on, versus 12mo's ~11-12 non-overlapping yearly
    cycles.
  - *Round 2 (6/9/12, proposed as an alternative) also rejected, on different
    grounds*: measured (not assumed) the average cross-asset correlation between each
    candidate lookback's momentum FEATURE (a structural/input diagnostic, not a
    backtested-performance peek, so this doesn't violate Rule 1/2) - 6/9/12 are
    correlated at **0.64-0.82** with each other, i.e. nearly redundant, defeating the
    entire point of a multi-horizon ensemble (genuine diversification across regime
    sensitivities). 3/12/24's pairwise correlations (0.44, 0.56, 0.33) are
    meaningfully lower - a real, evidence-based improvement, not just a compromise
    that "feels" more spread out.
  - **Final: 3/12/24mo.** 24mo's warmup consumes 16.9% of train (~9.8 usable years,
    ~5-6 non-overlapping cycles) - a smaller data-availability compromise than 36mo's,
    while still giving genuine separation from the unchanged 12mo anchor.
- **Crossover expands to 3 Books, one per MA pair** — 50/100, 50/200 (golden
  cross/death cross), 100/200, all three pairwise combinations of
  `feature_engineering.ipynb`'s own wishlist ("Moving Average Features: 50, 100,
  200"), not invented and not narrowed to one by backtest. Same discipline as
  breakout's 2 systems and momentum's 3 horizons - decided before building
  `research/crossover.py`, not after seeing the result (which turned out mixed: see
  CLAUDE.md's current-state table - 50/200 consistently positive, 50/100 fades in
  test, 100/200 sign-flips in validation - exactly the kind of spread across parallel
  specs this "report all of them honestly" discipline exists to surface rather than
  hide behind a single cherry-picked winner).
- **Short-term reversal is 6 Books, not 1** — two tiers (individual-asset,
  sector-level) x three lags (1d/5d/10d), all pre-committed before building
  `research/short_term_reversal.py`. The two-tier split mirrors Nagel (2011)'s own
  individual-stock vs. Fama-French-industry-portfolio comparison directly (his
  finding that an unconditionally-near-zero industry-reversal strategy still earns
  real, VIX-predictable returns in turmoil is the actual reason to build a
  sector-level tier, not a symmetry preference); the three lags mirror momentum's
  3-horizon and breakout's 2-system precedent (Lehmann's own Table I tested k=1wk
  through k=52wk and found k=1wk overwhelmingly strongest - 1d and 10d are included
  as the fast/noisy and slow/marginal boundary cases, not because either was expected
  to win). All 6 came back unprofitable net-of-cost (CLAUDE.md's current-state
  table) - reported as a full, honest spread across specs, not narrowed down after
  the fact to whichever looked least bad.
- **Carry is 4 Books** — cross_sectional_carry1m, cross_sectional_carry1_12,
  carry_timing_zero, carry_timing_mean (Phase 4b's paper-matched rebuild, mirroring
  Koijen-Moskowitz-Pedersen-Vrugt's own Table 2 carry1m/carry1-12 pair and Table 6's
  reference=0/mean pair). **Decided 2026-07-22, per direct instruction: "use them
  all"** — not narrowed to a subset despite the mixed result (no spec robustly
  positive across train/validation/test) logged for carry - the user's own explicit
  call, made knowing the individual-spec results already, on the reasoning that a
  covariance-based optimizer is the right place to let weak/uncorrelated specs
  compete for capital rather than pre-filtering them by hand (see the
  portfolio-scope discussion this session for the caveats attached to that
  reasoning - regime-dependent value still needs a regime classifier, which doesn't
  exist yet, and MVO on noisy near-zero-Sharpe inputs can amplify estimation error
  rather than filter it out).
- **XSMOM is 1 Book** — Asness-Moskowitz-Pedersen's own MOM2-12 measure, no
  multi-horizon grid (Phase 4b) - the paper is explicit it deliberately avoids
  testing several lookbacks "to minimize the pernicious effects of data snooping,"
  unlike TSMOM/breakout/crossover/reversal, which each have real multi-spec
  precedent from their own source papers or trading practice.
- Every Book-count decision above was made *before* touching `src/portfolio/`'s
  optimizer, per the same discipline as everything else in this project: fix the spec
  first, look at results after, never the reverse.

**Flagged, 2026-07-21, resolved same day: vol estimator selection was being done
inconsistently with everything above, and has been redone signal-agnostically.**

Both `research/momentum.py` (3 horizons) and `research/breakout.py` (2 systems) pick
"the winning vol estimator" (Yang-Zhang vs. EWMA) independently per spec, by
comparing **pooled train Sharpe** and taking whichever is higher - a performance-based
selection, done five separate times (3mo→EWMA, 12mo→Yang-Zhang, 24mo→EWMA;
System 1→Yang-Zhang, System 2→Yang-Zhang). Two problems with this, surfaced in
discussion rather than caught before building:

1. **Inconsistent with the standard applied everywhere else.** Lookback horizons and
   Turtle systems were deliberately *not* chosen by comparing our own train Sharpe
   (that would've been the same forbidden move as picking a universe by backtest
   performance, per Rule 1/2) - external reasoning (the paper's own designation, real
   Turtle historical precedent, feature-correlation checks) was used instead. Vol
   estimator choice never got held to that same bar; it's been train-Sharpe selection
   from the very first momentum build.
2. **Doesn't make economic sense on its own terms.** Volatility is a property of an
   asset's price history, not of whichever trading signal is consuming it - there's
   no economic reason Yang-Zhang should be the better estimator of Gold's true vol
   for a 12-month bet but EWMA better for the same asset's 24-month bet. A "winner"
   that flips depending on which signal is asking, across five closely-matched
   binary comparisons, is more likely to be train-period noise in a close two-way
   comparison than a stable, structural fact.

**The fix (built 2026-07-21):** `src/data/vol_forecast_eval.py` (pure functions -
`forward_realized_variance`, `qlike_loss`, `mse_vol_loss`, `per_asset_mean_loss`,
tested in `tests/test_vol_forecast_eval.py`) plus a new driver script,
`research/vol_estimator_comparison.py`, decouple vol-estimator selection from any
trading signal's Sharpe entirely - no signal, no position, no backtest anywhere in
either file. Design decisions actually made:

- **Ground truth**: forward realized variance built from the BACK-ADJUSTED curve's
  own simple returns (`close.pct_change()`) - the same series both research scripts
  actually trade, and neutral with respect to either estimator's own input curve
  (Yang-Zhang's is raw OHLC, EWMA's is back-adjusted returns), so using either
  estimator's own input as ground truth would have structurally favored it. RV_t =
  (252/h) * sum of the next h days' squared returns, i.e. genuinely forward-looking
  by construction (t+1...t+h) - correct for an ex-post forecast-accuracy check
  (nothing here trades on it), not a look-ahead violation of CLAUDE.md Rule 3 (which
  governs signals that trade "today").
- **Horizons**: two evaluated, not one - 21 trading days (PRIMARY, matches
  momentum's monthly rebalance, the dominant validated use case) and 5 trading days
  (robustness check, closer to breakout's daily cadence and a purer test of
  short-horizon forecast skill). Tying the primary horizon to actual downstream
  usage (rather than picking whichever horizon flatters a preferred estimator) kept
  this non-arbitrary.
- **Loss function**: QLIKE (Patton 2011), normalized to 0 at a perfect forecast,
  convex in realized/forecast, asymmetric (penalizes underprediction more) - the
  standard literature choice and the primary decision metric. MSE-on-vol reported as
  a symmetric robustness cross-check.
- **Period**: TRAIN only (formation date <= TRAIN_END), same discipline as every
  other spec decision in this project, applied here deliberately even though this is
  a data/methodology question rather than a strategy backtest.
- **Aggregation**: per-asset mean loss first (min 20 obs, else NaN - same MIN_OBS
  convention as backtest.performance), then MEDIAN across assets (robust to one
  noisy asset) plus a win-rate (fraction of assets where an estimator has the lower
  loss), not one pooled number.

**Result: Yang-Zhang wins, decisively and consistently.** Median QLIKE loss (train,
29 of 38 ADV-filtered assets had enough history to compare - the other 9, mostly
ICE softs and late-starting livestock/rates names, are NaN under the same MIN_OBS
gate used elsewhere):

| Horizon | Yang-Zhang median QLIKE | EWMA median QLIKE | Yang-Zhang win rate |
|---|---|---|---|
| 21d (primary) | 0.201 | 0.259 | - |
| 5d (robustness) | 0.407 | 0.470 | 72% (21/29 assets) |

MSE-on-vol was close to a coin flip at both horizons (win rates roughly 34-66%,
direction not consistent) - expected, since QLIKE and a symmetric squared-error loss
weight underprediction differently, and QLIKE is the metric that's actually relevant
here (this estimate sizes real positions, so underprediction is the costlier
direction). QLIKE agreeing at both the primary and robustness horizon, with a
decisive win rate at 5d, made this a clean call, not a coin-flip forced decision.
Full numbers: `Data/research/vol_estimator_comparison/decision.txt` and the
per-asset CSVs/PNGs alongside it.

**Yang-Zhang is now the single, project-wide vol estimator.** `research/momentum.py`
and `research/breakout.py` no longer compute EWMA at all (import removed) - the
`winning_vol = max(usable, key=usable.get)` pattern is gone from both files,
replaced by a fixed `VOL_ESTIMATOR = "yang_zhang"` constant. Both scripts re-run
clean against the current data (2026-07-21) with this change. Dashboard pages
07 (momentum) and 08 (breakout) keep their Yang-Zhang/EWMA radio toggle - still
useful for visual, exploratory comparison - but the Methodology copy on both no
longer implies a per-spec-picked "winner"; it explains the actual project-wide
decision and links to this section.

**Moving-average crossover built 2026-07-21** (signal scope item 3) -
`src/signals/crossover.py` (fully vectorized `sign(fast_SMA - slow_SMA)`, no
per-asset walk-forward loop needed, unlike breakout's genuinely path-dependent
state) + `research/crossover.py` + `dashboard/pages/09_crossover_performance.py`.
Three MA pairs (50/100, 50/200, 100/200 days), SMA not EMA, back-adjusted curve,
daily frequency, immediate flip with no confirmation filter, Yang-Zhang vol used
directly (built after the project-wide decision above, so there was no
`winning_vol` pattern to write in the first place) - see CLAUDE.md's current-state
table for the full design rationale and result.

**Real bug found and fixed live, same day:** the first version of
`build_crossover_regime` used a strict `min_periods == window` rolling mean (an
untested guess that a stable trend average shouldn't tolerate any gap tolerance).
Checked directly against real output: Corn, Soybeans, Wheat, Cocoa, Coffee, Cotton,
Sugar, and every other sparser-calendar asset came back **100% NaN across their
entire history** (0 of 5015 dates for Corn, even at the shortest 50/100 pair), not
just reduced coverage. Root cause is the same one already documented for
`data.volatility`'s `min_frac` and `signals.breakout`'s `DEFAULT_MIN_FRAC`: the
42-asset panel's date index is the union of every asset's own trading calendar, so
an asset with a sparser calendar than the panel (Corn: ~779 scattered 1-2-day gaps)
has at least one missing day inside essentially every 100-200-day window, over its
entire history - a strict rolling mean nulls the whole window the instant one day
inside it is missing, and with gaps this frequent, no window is ever fully clean.
Fixed by reusing the same already-validated `DEFAULT_MIN_FRAC = 0.7` tolerance and
rationale (not a fresh guess) - this is the third time this exact calendar-sparsity
issue has surfaced in three different signal families (Yang-Zhang's rolling
variance, breakout's rolling max/min, now crossover's rolling mean), which is worth
remembering as a standing hazard for any future rolling-window feature on this
panel, not just a one-off fix.

**Short-term reversal built 2026-07-21** (signal scope item 4, moved ahead of
cointegration per direct instruction - see CLAUDE.md's "Signal scope for now") -
`src/signals/short_term_reversal.py` + `src/signals/vix_overlay.py` +
`src/data/sectors.py` + `research/short_term_reversal.py` +
`dashboard/pages/10_short_term_reversal_performance.py`. **The first genuinely
cross-sectional signal in this project** - see CLAUDE.md's current-state table and
`references/short_term_reversal_implementation_recipe.md` for the full design
rationale (Lehmann 1990 mechanics, sector-scoped peer groups instead of the full
42-name universe, vol-standardization before demeaning, Nagel 2011's VIX-conditional
sizing overlay).

**Real bug found and fixed live, same day:** the VIX-adjusted sizing overlay was
first implemented by multiplying the raw cross-sectional SIGNAL by the day-level VIX
multiplier, then feeding the result through the normal `backtest_signal` pipeline
(which gross-exposure-normalizes to unit long/short every day). Checked directly
against real output: "simple" and "VIX-adjusted" Sharpe came back **bit-for-bit
identical in every single train/validation/test cell** - a uniform-across-assets
daily scalar multiplied into the signal before normalization factors out of both the
position sum and the `sum(|weights|)` normalization denominator identically, so it
has mathematically zero effect on the final book. This is the exact failure mode
CLAUDE.md's own Architecture section already documents in the abstract ("vol
targeting silently cancels out any post-solve position scaling") - it recurred here
in concrete form because the fix was applied to the wrong stage of the pipeline, not
because the warning was unknown. Fixed by applying the multiplier to the ALREADY
gross-exposure-normalized position array instead (`signals.vix_overlay.
apply_size_multiplier`, called after `backtest.engine.normalized_positions`, before
computing returns/cost) - "simple" and "VIX-adjusted" now produce genuinely different
position arrays and Sharpe numbers, as they should.

**Result, reported honestly:** unprofitable net-of-cost across all 6 specs (2 tiers x
3 lags) - turnover 109-362x annualized (far above breakout's ~50-60x), net Sharpe
-0.5 to -2.8 in train alone. This is the first of the four signal families built so
far (momentum, breakout, crossover, short-term reversal) with **no positive spec at
all** net-of-cost - breakout and crossover each had at least one.

**Second bug/gap, caught by direct question, same day: the VIX-conditioning
regression's first pass reported point estimates only (R² 0.0001 individual-tier,
0.0017 sector-tier), no standard error or significance test at all.** Asked directly
"how did you conduct the regression, this data seems bad for OLS" - correct
instinct: the regression's own inputs (a reversal book's daily return, mechanically
serially correlated because adjacent days' positions share most of the same
trailing-return window; VIX, itself highly persistent; return volatility that's
time-varying by construction, the whole phenomenon being tested) are exactly what
plain-OLS standard errors handle badly. Nagel's own paper already addresses this
with Newey-West HAC standard errors, 20 lags, for every daily regression he reports
(his Tables 2-5) - replicated that exactly (`research/short_term_reversal.py`'s new
`hac_regression_report`, `statsmodels.OLS(..., cov_type="HAC", cov_kwds={"maxlags":
20})`) rather than just asserting the low-R² finding was "essentially null."
**Result of doing it properly: individual-tier is genuinely indistinguishable from
noise (t=0.58, p=0.56) - that R² is not a real relationship. Sector-tier IS
statistically significant (t=2.33, p=0.02)** despite its own tiny R² (0.17%) - a
real, if economically small, effect, consistent with Nagel's own finding that
industry-level reversal's VIX-predictability is genuine even where individual-level
is muddier. Doesn't change the bottom line (net-of-cost Sharpe stays deeply negative
either way, and the sizing overlay it produces still helps in some out-of-sample
periods and hurts in others) but is a materially more precise finding than the
uncorrected first pass, and a real methodological gap worth remembering: report a
regression's significance, not just its point estimate and R², whenever the
dependent variable is a rebalanced strategy's own return series (mechanically
autocorrelated by construction, in almost every case in this project).

Full numbers: `Data/research/short_term_reversal/`. VIX itself newly added to the
data pipeline for this signal (`Data/vix_data.csv`, `jobs/update_macro_data.py`,
DATA_SCHEMA.md section 3) - CBOE VIX index via `yfinance`, 1990-present, no
publication-lag concern (same-day observation, unlike GSCPI/fed-funds).

**First small-scale pass run 2026-07-22, per direct instruction ("validate plumbing
on a small set, then continue")** - closed two real, pre-existing gaps in
`src/portfolio/` before running anything:

- **`src/portfolio/covariance.py` (new)**: `build_cov_dict(returns_df, window=252,
  freq="ME")` - rolling Ledoit-Wolf shrinkage covariance (`sklearn.covariance.
  LedoitWolf`), point-in-time correct (`returns.loc[:date].iloc[-window:]`), keyed
  by REAL trading dates (`real_period_end_dates` - resamples the index itself, not
  the returns, since `resample("ME").last()`'s own calendar-label index frequently
  isn't a real trading day and would silently fail to intersect with any signal's
  own date index downstream). Extracted from the retired Cross Asset Stat Arb
  Engine's own pattern (evaluated as "transfers" in `cleanup.md` section 3 back on
  2026-07-15, but never actually built as a reusable function anywhere - only ever
  inlined directly in that engine's own `run_baseline.py`/`run_engine.py`). Also
  handles scattered NaN within a window (drops NaN rows before fitting, gated by
  the same already-validated 0.7 `min_frac` tolerance used elsewhere in this
  project) - `LedoitWolf` itself has no native NaN tolerance, unlike this project's
  own rolling-window functions.
- **22 new unit tests** (`tests/test_optimizer.py`, `tests/test_covariance.py`,
  `tests/test_book.py`, `tests/test_allocator.py`) - `optimizer.py`, `book.py`,
  `allocator.py`, and `signals/combine.py` had zero test coverage before this,
  despite `book.py`'s own docstring already flagging it as "not yet exercised
  against real signals."
- **Book-count decisions logged for carry (4) and XSMOM (1)** - the two families
  missing from the decisions list above until now (see that list).

**First real exercise**: one representative Book per family (6 total - momentum
12mo, breakout System 1, crossover 50/200, short-term reversal individual/5d, carry
timing mean, XSMOM), explicitly NOT the full 19+-spec roster every family's own
Book-count decision would eventually call for - scaling up is the natural next step,
not attempted in this pass.

**Two real bugs found and fixed running actual data through this scaffolding for
the first time, not hypothetical:**

1. `Book._period_return_map` summed each rebalance window's daily returns via a
   plain numpy `.sum()` on `.values` - this propagates NaN across the ENTIRE
   period the moment a single scattered missing-return day falls inside it, which
   this panel's genuinely sparser-calendar assets (grains, softs, livestock, SOFR)
   guarantee happens routinely. Every book's PnL came back all-NaN before this was
   found. Fixed to `.sum(skipna=True)` (pandas' own default) - a missing day now
   correctly contributes zero to that period, matching every other NaN-handling
   convention already established in this project (e.g. `cross_sectional_demean`'s
   min-group-size gating), rather than corrupting the whole period.
2. `Book.run()`'s own PnL is genuinely MONTHLY-periodicity (one value per
   rebalance date) - comparing it against `backtest.performance.performance_stats`
   (hardcoded to daily/252 annualization, correct for every OTHER pnl series in
   this project, which are always daily even for monthly-formed signals) overstated
   annualized return and vol by roughly `sqrt(252/12) ≈ 4.6x`. Fixed with a
   period-aware stats helper parameterized by the series' own real periodicity, used
   only for the Book/Allocator side of the research script's own comparison.

**Also found and corrected, a dimensional-sanity issue, not a bug**: alpha units
differ wildly across families (rank-based scores vs. vol-targeted-sign scores), so
each Book's alpha is rescaled by its own train-period standard deviation before
reaching the optimizer. An initial `gamma=10` (an arbitrary-looking "default") left
the risk term (`gamma × Sigma`, with this universe's daily return variance ~1e-4)
utterly dominated by `kappa`, letting individual books reach 8-12x gross exposure
and single-month PnL swings past -50%. `gamma=20000` brings the risk term back to
the same order of magnitude as `kappa`/rescaled-alpha, producing a sane (though
still un-backtested-in-the-Sharpe-sense) 3-5x gross exposure range. Chosen to avoid
a degenerate optimizer regime, not selected by comparing backtest performance
(`CLAUDE.md` Rule 1/2 governs signal-spec selection, not sizing a regularization
constant to the actual scale of the numbers involved).

**Each Book restricted to assets with >=90% valid RETURN history**, not the full
ADV-filtered universe - `LedoitWolf` needs every column simultaneously non-null on
the same date, and the union of several assets' own gaps compounds fast (measured:
even a lenient inclusion threshold left the most recent 252-day window only 58%
jointly clean, below the covariance module's own 70% `min_frac`). Raising the bar to
90%-valid-per-asset (~25 of 38 qualify) restores a usable joint-clean-row rate. Drops
the ICE softs, SOFR, KC_Wheat, Russell2000, livestock, and grains from this pass
specifically - a real scope restriction for this first pass, not the full universe.

**Result: genuinely mixed, not a clean case for the optimizer yet.** The
Allocator-combined book's test-period Sharpe (0.23) beats the naive equal-weight
blend of the same 6 raw signals (-0.06), but validation (2020-2021, the COVID shock)
is meaningfully worse for the optimizer (-1.22) than the naive blend (-0.38).
Per-book Sharpes inside the Allocator run: momentum_12mo 0.61, breakout_system1
0.59, crossover_50_200 0.35, reversal_individual_5d 0.33, carry_timing_mean -0.36,
xs_momentum -0.25 - roughly consistent with each family's own already-known
standalone result. Test window capped at `SPREAD_DATA_END` (carry's real spread
data freezes there) - stated explicitly, same discipline as carry's own dashboard.

- `research/portfolio.py` (driver script) + `dashboard/pages/
  13_portfolio_performance.py` (registered in `dashboard/app.py`, verified
  exception-free via `streamlit.testing.v1.AppTest`) - a report page, not a
  per-asset explorer like pages 07-12, since the point of this pass is the
  combined portfolio-level result, not individual asset tearsheets.
- Full `pytest tests/` run clean after this pass.
- **Not attempted in this pass, deliberately**: scaling to the full 19+ Book
  roster; any regime-conditioning (`Allocator`'s `regime_lookup` stays unused -
  Phase 8's classifier still doesn't exist); tuning `gamma`/`kappa`/`lambd`/
  `max_weight`/`target_vol` beyond the dimensional-sanity fix above; resolving the
  monthly-cadence-for-fast-signals mismatch (breakout/crossover's own pages trade
  them daily).

**Dashboard reorganization — done 2026-07-22** (built in a fresh session per the
handoff above, to avoid mid-task context compression). Scope matched the agreed plan
exactly, no changes to the design:

1. **New top-level nav category** in `dashboard/app.py`, `"Portfolio Construction"` —
   mirrors the existing `"Strategy Performance"` category's pattern (a plain
   `st.navigation()` dict key with a list of `st.Page`s). `dashboard/pages/
   13_portfolio_performance.py` moved into this new category unchanged — it already
   answers "did the combined book make money" (Sharpe, equity curve,
   optimizer-vs-naive-blend comparison), a different question from the new page
   below, not a duplicate.
2. **New page, `dashboard/pages/14_portfolio_optimizer_health.py`** ("Optimizer
   Health" in the nav) — "is the machinery itself behaving sensibly," not a
   performance report. Reuses the exact same 6-Book construction already built in
   `research/portfolio.py`/page 13 (`_active_columns`, `_build_book`,
   `build_cov_dict`, same `GAMMA`/`KAPPA`/`LAMBD`/`MAX_WEIGHT`/`COV_WINDOW`/
   `COV_FREQ` constants) — no new backtest logic. Three sections, as agreed:
   - **Structure**: per-Book table — active assets (and % of the shared liquid
     universe), `gamma`/`kappa`/`lambd`/`max_weight`/`target_vol`, valid rebalance
     dates vs. skipped (warmup/NaN-gated).
   - **Covariance characteristics**: condition number of Σ over time per Book (log
     scale), average pairwise correlation trend, and a table making visible how much
     of the liquid universe survives the 90%-valid-return-history NaN-gating
     restriction per Book — previously buried in `_active_columns`'s own docstring,
     now a rendered table.
   - **Optimizer health**: turnover per Book over time, the vol-target scale factor
     trend (with `scale_min`/`scale_max` reference lines) plus a `% Pinned at
     scale_min`/`scale_max` column flagging Books where the target vol isn't
     genuinely achievable, and cap-bind frequency/rate (`n_cap_bind` — how often
     `max_weight` actually binds, a concentration-risk signal) per Book.

   **One real extension needed to build this, not hidden**: `src/portfolio/book.py`'s
   `Book.run()` previously only returned aggregate scalars (`turnover`, `avg_scale`,
   `n_cap_bind`) even though it already computed the full per-date series
   internally (`turnover_s`, `scale_history`, `cap_bind_hist`) before collapsing them
   to a mean/count for the return dict. Extended `run()`'s return dict with three new
   keys — `turnover_series`, `scale_series`, `cap_bind_series` — plus
   `n_rebalance_dates_total`/`n_rebalance_dates_valid` (also added to the early-exit
   `< 20 common dates` branch). Purely additive (existing keys unchanged, existing
   `tests/test_book.py` assertions use `key in result`, not exact-set equality) —
   surfacing already-computed diagnostics, not new backtest logic.

   Both pages verified exception-free via `streamlit.testing.v1.AppTest` (3
   dataframes, 4 plotly charts rendered on page 14, no exceptions). Full
   `pytest tests/` suite still passes clean (125, unchanged — dashboard-only work,
   no new unit tests needed; `tests/test_book.py`'s existing assertions on
   `Book.run()`'s return dict already tolerate the new additive keys).

   **Manually spot-checked live 2026-07-22** (Streamlit server launched locally,
   driven with Playwright to confirm both nav pages render with real numbers, not
   just AppTest's exception-free check) — caught and fixed one real cosmetic bug in
   the process: page 14's condition-number and avg-pairwise-correlation charts had
   their Plotly in-figure `title` overlapping the horizontal legend at the default
   `legend=dict(y=1.15)`. Fixed by moving all four chart titles on the page out to
   `st.markdown` headers above each `st.plotly_chart` call (matching the convention
   page 13's own charts already used) instead of Plotly's internal `layout.title`,
   and nudging legend `y` to 1.18 for headroom. **Marked for continued evaluation**:
   the user is reviewing the live dashboard directly (not further automated
   screenshot review) — treat page 14's content/framing as still open to revision
   pending that feedback, not a closed deliverable.

**Portfolio-construction gap analysis vs. institutional practice — 2026-07-22.**
Evaluated `src/portfolio/` (`optimizer.py`, `book.py`, `allocator.py`,
`covariance.py`), `research/portfolio.py`'s 6-Book construction, `signals/
transforms.py`, `data/universe.py`, and `backtest/costs.py` directly (code read, not
inferred from documentation) against `references/Portfolio Construction for CTA and
Managed Futures Strategies.pdf` — a literature/practitioner synthesis of
institutional CTA portfolio construction (AQR, Man AHL, Winton, Aspect, Graham,
Transtrend, Quantica public disclosures plus the academic canon: Markowitz,
Ledoit-Wolf, Garleanu-Pedersen, Moskowitz-Ooi-Pedersen).

Already close to institutional shape: rolling Ledoit-Wolf shrinkage covariance (the
report's #3-ranked practice, its "widely used" tier — arguably ahead of where a
solo researcher's build usually is at this stage), the pre-performance ADV liquidity
floor (`CLAUDE.md` Rule 1), a turnover-penalized closed-form optimizer structurally
matching the report's own Garleanu-Pedersen-style objective, per-Book EWMA vol
targeting, and genuine 6-family style diversification combined via a flat
equal-weight sum in the `Allocator` — which is itself the report's own recommended
STARTING point ("equal-weight families before optimizing families"), not a
shortcut.

**Real gaps found, logged as next steps, not yet built:**

1. **Evaluate signal-proportional sizing against the current sign-based approach.**
   `signals/transforms.py`'s `vol_targeted_sign_signal` (used for momentum/
   breakout/crossover in the 6-Book pass) sizes by direction only (`sign(feature)`)
   times vol — its own docstring says so explicitly ("size comes from vol alone,
   not from the feature's own magnitude") — a faithful reproduction of
   Moskowitz-Ooi-Pedersen's own construction, not an oversight, but not the
   report's ranked #1 institutional default (signal-proportional sizing,
   `x ∝ f̃/σ`, proportional to a normalized, capped forecast). A magnitude-aware
   `continuous_signal(feature, vol) = feature/vol` already exists in the same
   module (currently unused outside the legacy `feature_engineering.ipynb`) — the
   mechanism exists, it just isn't wired into the current Book construction.
   **Per direct instruction: this must be evaluated empirically, not swapped in on
   the strength of the report's citation alone** — same discipline as every other
   signal-spec decision in this project (`CLAUDE.md` Rule 1/2: no spec change
   without honest train/validation/test evidence). Candidate test: rebuild 1-2 of
   the 6 Books (e.g. `momentum_12mo`, `breakout_system1`) with `continuous_signal`
   in place of `vol_targeted_sign_signal`, compare train/validation/test Sharpe AND
   turnover honestly against the existing sign-based result, report whichever wins
   — or report a genuinely mixed result, as every other signal-family finding in
   this project has been. Not started yet.
2. **No leverage/gross-exposure cap at Book or Allocator level.** `optimizer.py`
   only clips per-asset `±max_weight`; nothing bounds `Σ|x_i|` (gross) or `Σx_i`
   (net). The report explicitly separates "risk leverage" (vol targeting, already
   present here) from "notional leverage" (absent here).
3. **Cost-awareness is built but disconnected.** `optimizer.py`'s `lambd` L1
   turnover-cost term exists but runs at `LAMBD=0.0` in the actual 6-Book pass
   (visible directly in page 14's Structure table), and doesn't reuse the
   already-built `backtest/costs.py` liquidity-tiered bps-by-asset model used
   elsewhere in this project. Wiring these together is the report's explicitly
   named "line between gross and net alpha."
4. **No turnover-aware sleeve weighting.** This project's own prior results
   already show a ~100x spread in measured annualized turnover between short-term
   reversal (109-362x) and carry timing (0.7-3.3x), yet both get identical
   `GAMMA`/`KAPPA` in `research/portfolio.py`. The report explicitly recommends
   smaller risk budgets for faster/costlier sleeves — this project has the
   evidence to act on that recommendation already sitting in its own dashboard
   pages.
5. **No portfolio-level (Allocator-level) vol target.** Each Book targets its own
   vol; `Allocator.run()`'s combined PnL is a flat, unrescaled sum with no
   top-level target-vol layer, unlike AQR's own two-layer (asset + portfolio)
   construction cited in the report.

See Phase 8's own entry, same date, for the constraint/overlay half of this same
review (sector caps, stress testing) — those gaps land there, not here.

**Bug found and fixed, 2026-07-22: `Book.run()` silently held stale, unmanaged
positions through 16-month covariance blackouts.** Found while the user was
visually reviewing the live Portfolio Construction page (page 13) — the Combined
Equity Curve showed what looked like an instantaneous crash near January 2014
(spike to ~1.22, cliff to ~0.55). Traced with a direct diagnostic script (not
guessed): the chart point was NOT a single-month event. `portfolio.covariance.
build_cov_dict`'s 70% `min_frac` joint-clean-row gate silently skipped **15
consecutive month-end rebalance dates** (2014-05-30 through 2015-07-31) for
`carry_timing_mean` and `reversal_individual_5d` (and to a lesser extent other
Books sharing the same ~24-25-asset active universe) — no single asset was badly
broken (Platinum's the worst at 56/252 NaN days, ~22%), but the UNION of many
assets' small scattered gaps pushed the joint-clean-row count below the 176-row
gate for over a year straight, exactly the failure mode `_active_columns`'s own
docstring predicted in the abstract but had never been confirmed to actually bite.
`Book.run()`'s rebalance grid consequently jumped straight from `2014-04-30` to
`2015-08-31`, and the position set on April 30 — several assets already pinned
near `max_weight=0.3` — rode completely unmanaged (no re-optimization, no
re-vol-targeting, no risk check of any kind) through two real, large market moves
that happened to fall inside that gap: the 2014-2015 oil price collapse (WTI
summed return -79%, Brent -68%, Natural Gas -71% over the gap) and the January 15,
2015 SNB franc de-pegging shock (+19.4% single-day move). The resulting -45.7%/
-33.8% single-"period" losses were real numbers from real price moves, but
attributing 16 months of unmonitored, maximally-levered exposure to a single chart
point is not something any real risk process would allow — an institutional book
would have either found a usable covariance estimate on a shorter fallback window
or flattened when it couldn't.

**Fix**: `Book` gained a `max_gap_days` parameter (default 60 — tolerates one
skipped monthly rebalance from ordinary data noise, decisively catches multi-month
blackouts). `_period_return_map` now checks the REAL calendar-day gap between
consecutive valid rebalance dates; a gap longer than `max_gap_days` contributes a
**zero** return for that stretch (flatten) instead of pricing the book against
whatever the market did while risk couldn't be measured — not a claim the position
was actually flat, a refusal to keep pricing risk that was never being verified.
`run()`'s returned dict gained `n_stale_gaps` (count of flattened gaps) alongside
the existing diagnostics. One new regression test (`tests/test_book.py::
test_period_return_map_flattens_long_gaps_instead_of_holding_through_them`)
reproduces a normal gap (priced as before) and a stale gap (flattened) side by
side and asserts both behaviors; the existing `_period_return_map` test updated
for the new `(period_ret_map, n_stale_gaps)` return signature. Full `pytest
tests/` clean (126 passed — 125 prior + 1 new). Both dashboard pages (13, 14)
re-verified exception-free via `streamlit.testing.v1.AppTest` — no call-site
changes needed since `max_gap_days` defaults to 60.

**Numbers changed for real** (`research/portfolio.py` rerun against the fix,
`Data/research/portfolio/` regenerated): per-book Sharpe — momentum_12mo 0.61→
0.31, breakout_system1 0.59→0.30, crossover_50_200 0.35→0.34, reversal_
individual_5d 0.33→0.39, carry_timing_mean -0.36→-0.17, xs_momentum -0.25→-0.04.
Combined Optimizer/Allocator Sharpe: train 0.46, validation -1.21 (essentially
unchanged — the 2020-2021 validation window doesn't touch the 2014-2015 gap),
test 0.09 (down from 0.23 — the fixed version no longer benefits from whatever
share of the old test-period comparison was inflated by the stale-gap artifact
bleeding into later periods via the EWMA vol tracker). **The headline finding is
unchanged in spirit** (genuinely mixed, optimizer beats naive on test but not
validation) but every number backing it moved, and the old numbers should be
treated as superseded, not historical record, since they were computed on a
demonstrably broken mechanism, not a legitimate alternative construction choice.

**Prospective value factor — data acquisition done 2026-07-22/23, signal itself
not yet built.** Per direct instruction, researched Asness-Moskowitz-Pedersen
2013 "Value and Momentum Everywhere" (`references/Value and Momentum
Everywhere.pdf`, read directly, not from memory) for how it defines value outside
equities, since this project is all-futures with no book-value data. Outside
individual stocks the paper uses **negative 5-year past return** as the value
proxy everywhere (validated in the paper itself: correlates 0.83-0.86 with
BE/ME-based value where both exist), with two asset-class-specific refinements:
**bonds** use the 5-year change in the 10-year yield (not price return directly),
and **currencies** use a PPP-adjusted real FX return (nominal 5-year return minus
the relative CPI inflation differential vs. the US).

- **Groupings vs. this project's own**: the paper splits by asset class only (5
  groups — stocks, country equity indices, currencies, bonds, commodities — all
  27 commodities pooled into one factor). `data/sectors.py` is finer-grained (9
  groups, commodities split into Energy/PreciousMetals/IndustrialMetals/Grains/
  Softs/Livestock). Decided to rank within this project's existing sectors, not
  the paper's coarser split, for the same reason `sectors.py`'s own docstring
  already gives for reversal, and to match carry/reversal/XSMOM's established
  convention.
- **Bonds data**: already sitting in `Data/Yield_Curve_6M_to_30Y.csv`
  (`DATA_SCHEMA.md` section 3, collected 2026-07-14, previously unused) — full
  2Y-30Y curve, monthly since 1981. No new data needed; richer than the paper's
  own single-yield-per-country simplification, since each Rates-sector bond
  future (US_2Y...UltraBond) can get its own maturity-matched yield-change signal.
- **FX PPP data (CPI) — new, `Data/cpi_level_index.csv`.** Confirmed nothing
  existed locally. Sourced from FRED for 7 of 8 currencies (`jobs/
  update_macro_data.py`'s `update_cpi()`, same keyless pattern as the other
  macro pulls). **Real lesson found live**: a first attempt using FRED's OECD-MEI
  "growth rate previous period" series family resolved fine (valid URLs, real
  data) but turned out stale-to-discontinued for most non-US countries when the
  actual last-observation date was checked, not just whether the request
  succeeded — resolving is not the same as being current. Switched to each
  country's CPI level index instead (simpler for the PPP calc too — a direct
  log-ratio of levels, not compounding a growth-rate series with an easily
  mixed-up MoM/YoY convention), which fixed 6 of 7 remaining countries.
- **JPY CPI — fixed 2026-07-23 via e-Stat, not FRED.** Every FRED-hosted Japan
  CPI series (OECD-MEI family, level and growth-rate variants alike, and a
  reverse-engineered direct OECD SDMX API attempt that didn't pan out) was stuck
  at 2021-06, ~5 years stale. Confirmed this was a MIRROR-CHANNEL problem, not a
  Japan-side data gap, by pulling Japan's own Ministry of Internal Affairs and
  Communications (総務省) Statistics Bureau data directly via e-Stat
  (`api.e-stat.go.jp`) — table `0003427113`, filtered to nationwide/all-items/
  index-level (codes found live via `getMetaInfo`, not guessed) — reaches
  2026-05, a completely normal ~2-month publication lag. **This is the first
  credential-gated data source in this project** — e-Stat requires a free `appId`
  (registration confirmed required by testing unauthenticated first: got an
  explicit "check your application ID" error, not a silent failure), issued via
  e-Stat's own Mypage → API → "Issue Application ID" flow. Stored as the
  `ESTAT_APP_ID` environment variable — same OS-level env-var convention already
  established for `DATABENTO_API_KEY`, a deliberate choice over introducing a
  project `.env` file (this project has no `.env` file or dotenv usage anywhere;
  adding one now for 3 total credentials would introduce a second, strictly less
  safe convention alongside the existing one — a `.env` file is one `git add .`
  away from being committed, an OS env var never is). `update_cpi()`'s e-Stat
  sub-pull is independently fault-tolerant (if it fails, the other 7 FRED-sourced
  countries still get written), matching this file's own "one failing shouldn't
  block the others" principle stated in its module docstring.
- **Value signal itself: built 2026-07-23** (`signals/value.py`, `research/
  value.py`) — reuses `signals.transforms.cross_sectional_rank`, as planned
  above. Full recipe and result logged in `CLAUDE.md`'s Value row, not
  duplicated here. GBP/CAD/CHF/MXN's own 1-2yr FRED-mirror lag was still not
  individually fixed (JPY was by far the worst offender, which is why it got the
  dedicated e-Stat treatment first) — genuine staleness, left as-is per this
  project's "don't fabricate, let it gate out" convention. AUD turned out to be
  a different problem entirely — see below.

**AUD CPI bug — found and fixed 2026-07-23, while building the mix-vs-integrate
comparison below, not while evaluating value standalone.** Australia's ABS
publishes CPI **quarterly**, not monthly (confirmed directly: median gap between
real AUD readings in `Data/cpi_level_index.csv` is 92 days vs. ~31 for every
other country here) — a real reporting-cadence fact, not a pipeline gap. `data/
macro.py`'s `load_cpi()` previously left those two in-between months as NaN
every quarter (deliberately, by the same "don't forward-fill, let it gate out"
principle that correctly applies to the OTHER five lagged countries — but AUD's
situation isn't staleness, it's cadence, so blanket non-forward-filling was the
wrong call specifically for AUD). Combined with `signals/value.py`'s
`fx_ppp_value_feature` capping its own daily-reindex ffill at 35 days (sized for
a monthly-cadence source), AUDUSD's value score came back NaN 64% of the time.
That alone barely dented the STANDALONE value signal (`cross_sectional_rank`'s
per-date gating only drops AUDUSD itself on an affected day), but it broke
`portfolio.book.Book.run()` badly: `Book.run()`'s `alpha_df.dropna()` is a
JOINT, row-wise requirement across the whole active universe — one asset's NaN
drops that date for every asset in the Book, not just the affected one. Result:
value's Book only realized 60 of 125 possible monthly rebalance dates (vs.
XSMOM's 124 on the identical universe), and this was the actual reason an early
"mix beats integrate" result looked so clean before the bug was caught — see the
mix-vs-integrate writeup below.

**Fix**: `data/macro.py`'s `load_cpi()` now has a `QUARTERLY_CPI_COUNTRIES =
{"AUD": 2}` map — a bounded `ffill(limit=2)` in monthly-index space, applied
ONLY to AUD (every other country here is genuinely monthly). This isn't
fabricating data: for a quarterly-reporting country, the correct "latest known
CPI as of this date" for the two in-between months genuinely IS the last real
quarterly print — that's what the raw source itself represents, just not
literally re-stated every month. The `limit=2` bound still lets AUD go
genuinely NaN if it stops publishing for more than one quarter, same discipline
as `fx_ppp_value_feature`'s own capped ffill. Measured effect: AUDUSD's
value-score NaN rate dropped from 64% to 19%; value's Book went from 60 to 102
of 125 realized periods.

### Phase 7 exercised for real: mix vs. integrate, Value + XSMOM — built 2026-07-23

Per direct instruction, once value.py existed alongside XSMOM (both cross-
sectional, rank-weighted, same source paper — Asness-Moskowitz-Pedersen 2013),
tested this project's first real "combine two signal families" question using
`src/signals/combine.py` (built ahead of schedule per `cleanup.md` section 3,
never actually exercised until now) — the exact fork that module's own
docstring exists to leave open.

**Source read directly before building anything**: `references/AQR - Portfolio
Construction Matters.pdf` (Fitzgibbons, Hecht, McQuinn, Serban 2017) — AQR's own
"mix" (separately build the top-value and top-momentum portfolios, then combine
the two already-built portfolios) vs. "integrate" (average each asset's value
and momentum score into ONE composite first, then build a single portfolio from
that) terminology, confirmed against the PDF's own Exhibit 1/2, not assumed.
Also confirmed AQR's "momentum" there is cross-sectional/rank-based (matches
this project's XSMOM, not TSMOM, which has no rank-vs-peers to average) before
proceeding — an open question the handoff explicitly flagged, resolved by
reading, not by assumption.

**Found live: `src/signals/combine.py`'s own docstring had "mix" and
"integrate" backwards relative to AQR's actual terminology** (it called
blend-into-one-alpha "mixing" and combine-at-the-PnL-level "integrating") —
corrected the same day.

**Mapping, confirmed by direct instruction** (full Book/Allocator combination
chosen for "mix" over a lighter position-averaging alternative, when asked):
integrate = `signals.combine.combine_alphas([value_signal, xsmom_signal],
method="equal")` on the two already-rank-centered signals (algebraically the
same operation as AQR's own rank-averaging), through the standard lightweight
`backtest_signal` path; mix = one `portfolio.book.Book` per signal (calibration
reused verbatim from `research/portfolio.py`'s 6-Book pass — GAMMA/KAPPA/LAMBD/
MAX_WEIGHT/BOOK_TARGET_VOL/EWMA_HALFLIFE/COV_WINDOW/COV_FREQ, per-Book alpha
rescaled by its own train-period std — not re-derived, so this comparison isn't
accidentally also a "different calibration" comparison), combined via
`portfolio.allocator.Allocator`'s post-solve PnL sum.

**Result, before the AUD fix**: Mixed cleanly beat Integrated in all three
periods (test Sharpe +0.21 vs. -0.68) — this is what triggered the
investigation above; the clean sweep didn't match the "genuinely mixed, no easy
winner" pattern every other multi-spec comparison in this project has shown, and
turned out to be a data-coverage artifact (see AUD CPI bug above), not a real
finding.

**Result, after the fix — reported honestly, not tuned back to the pre-fix
story**: Sharpe (train/validation/test, gross) — Value standalone -0.01/+0.13/
-0.69; XSMOM standalone -0.34/-1.39/+0.04 (unchanged, no CPI dependency);
Integrated -0.24/-1.19/-0.72; Mixed -0.34/-0.80/+0.04. Mixed no longer cleanly
dominates: Integrated is now *better* in train (reversed from the pre-fix
read), Mixed still wins validation clearly, and Mixed's test edge shrank from
+0.21 to +0.04 (now barely distinguishable from XSMOM standalone's own +0.04).

**Two flagged asymmetries in this specific comparison, not smoothed over**:
(1) Mixed runs through the full covariance-aware optimizer (vol targeting,
position inertia) while Integrated is a raw rank-averaged signal at unit gross
exposure — AQR's own equities study doesn't have this confound (their mix and
integrate portfolios are both plain stock selections, no optimizer either way),
so "mixed vs. integrated" here is partly also "optimizer-combined vs.
rank-combined." (2) Mixed is gross-only (`LAMBD=0.0` in this Book calibration,
no `cost_bps` deduction inside Allocator/Book PnL) — not net-comparable to the
other columns' net rows.

Code: `research/value_momentum_combine.py`. Outputs in `Data/research/
value_momentum_combine/`. No dashboard page yet — a natural follow-on, not
built in this pass. Full result also logged in `CLAUDE.md`'s "Mix vs. integrate
(Value + XSMOM)" row.

### Signal-level dynamic correlation — first pass built 2026-07-23

Per direct instruction, following straight on from the mix-vs-integrate result
above (AQR's "integrate" edge depends on value/XSMOM being negatively
correlated — worth knowing whether that holds up over time, not just as a
full-sample average). Built `src/portfolio/correlation.py` (`rolling_correlation`,
`ewma_correlation`, `correlation_summary` — plain pairwise Pearson, EWMA alpha
derived the same `1 - exp(-ln(2)/halflife)` way `Book._apply_vol_target`
already does, deliberately NOT DCC-GARCH — see below for where that belongs)
and `research/signal_correlation.py`, run at two frequencies: daily standalone
strategy returns (~4,650 obs, no optimizer) and Book/Allocator monthly PnL
(~80-100 obs, full optimizer path).

**Result (daily leg — unaffected by the weekly-cadence Book switch below):
negative correlation is real and persistent, not stable.** Full-sample daily:
-0.31. Daily rolling (252d) and EWMA (hl=63d) agree closely (means -0.34,
negative ~86% of the time) — the AQR mechanism holds up as a typical
description here, not a full-sample coincidence. But both estimators swing to
genuinely POSITIVE at times (up to +0.46) — real stretches where value and
XSMOM move together, which a static full-sample number would hide.

**Result (Book/Allocator leg — re-run 2026-07-23 after the weekly-cadence
switch below, superseding the original monthly-cadence read): correlation
looks EVEN MORE consistently negative at weekly cadence, but read this with
the same autocorrelation caveat the tuning section below raises.** Full-sample
weekly: -0.40 (vs. -0.34 monthly). Weekly rolling (104wk) and EWMA (hl=52wk)
are negative 100% of the time (vs. 86-90% monthly), range -0.57 to -0.99
(rolling) / -0.17 to -0.99 (EWMA), materially tighter std (0.08-0.14 vs.
0.21-0.23 monthly). This is plausible, not obviously wrong — the daily leg
already shows -0.31 full-sample, and weekly Book PnL leans on the same
underlying monthly-cadence alpha for ~4-5 consecutive weeks at a time, which
would naturally smooth OUT the same short-lived positive-correlation episodes
the monthly and daily reads pick up, not necessarily reveal a truer signal. A
simple estimator was informative enough to answer the original question either
way; escalating to DCC-GARCH wasn't needed here.

**DCC-GARCH (`src/dcc_garch`, a separate local project at `Projects/DCC Garch
Rompolis`, evaluated 2026-07-23) — scoped as a NEXT step, after hyperparameter
tuning (see below, current top priority), not built yet.** Clean two-stage
implementation: per-asset GJR-GARCH(1,1,1) (`arch` package, Stage 1) feeding a
pure numpy/scipy DCC/ADCC estimator (Stage 2, reverse-engineered from R's
`rmgarch` with documented deliberate corrections, already validated on 10 real
equity ETFs — ADCC beats DCC by +197 log-likelihood). Has its own
`pyproject.toml` — the integration path is an editable local install
(`pip install -e`), not a rewrite, per direct instruction.

**Scope, per direct instruction: three levels of dynamic correlation
monitoring, not just the asset-level Σ question Phase 7's intro paragraph
already flagged**:
1. **Between sectors** — a sector-level aggregate correlation structure (is
   Energy diversifying Grains right now, or has that broken down).
2. **Between assets within a sector** — intra-sector redundancy (e.g. is
   WTI/Brent/HeatingOil still 3 genuinely different bets, or has Energy
   effectively become 1 degree of freedom).
3. **Between signals/Books** — the DCC-GARCH escalation of the simple
   rolling/EWMA work above, if a future comparison shows correlation dynamics
   the simple estimator can't capture.

(1) and (2) are both really the same asset-level Σ question already flagged in
this Phase's intro (an alternative/supplement to `portfolio/covariance.py`'s
rolling Ledoit-Wolf), just read at different aggregation granularity — not a
new architectural question, a refinement of an existing one. Real work still
ahead before any of this is production: fitting per-asset GJR-GARCH across
this project's 38-asset, multi-decade, genuinely-gappy panel (this project's
own documented sparse-calendar problems will stress `arch`'s optimizer harder
than the 10-ETF validation set did) and validating the resulting Σ against
Ledoit-Wolf before it goes anywhere near `Book` — "does DCC-based Σ actually
beat rolling Ledoit-Wolf here" stays an empirical question, not an assumption,
per `cleanup.md` section 3's own framing.

**IPCA (Instrumented Principal Component Analysis) — flagged 2026-07-23 as a
second covariance/risk-model candidate alongside DCC-GARCH above, not built,
paper read directly (`references/Instrumented PCA.pdf`, Kelly, Korsaye, Pruitt
& Su — the formal econometric-theory paper, not KPS19/Kelly-Pruitt-Su 2019
"Characteristics Are Covariances," which is the original *applied* paper this
theory paper underpins).** Model: `x_{i,t} = β_{i,t} f_t + μ_{i,t}`, with
loadings themselves instrumented by observable characteristics, `β_{i,t} =
z_{i,t} Γ + η_{i,t}` — a structural link between time-varying per-asset
attributes (`z`) and dynamic factor loadings, estimated jointly with the
latent factors `f_t` via alternating least squares (both sub-problems are
plain OLS, no numerical optimizer needed). Two properties make it relevant
here specifically: (1) loadings are allowed to vary over time without the
parameter count exploding (`Γ` is `L×K`, fixed size, not `N×K` growing with
the panel — the paper's own stated motivation is exactly this project's
situation, a moderate cross-section with loadings that plausibly drift), and
(2) it handles unbalanced panels natively (pooled-OLS-like), relevant given
this project's own well-documented sparse-calendar/gappy-history problem
(Yang-Zhang, crossover, breakout all hit variants of this). Estimation
convergence: `Γ` at rate `√(NT)` (faster than plain PCA's `β` at `√T`, because
IPCA also pools cross-sectional information), `f_t` at `√N`, same as PCA.

**Legitimate fit for this project, if pursued later**: replacing or
supplementing `portfolio/covariance.py`'s rolling Ledoit-Wolf Σ with
characteristic-instrumented, time-varying loadings — candidate instruments per
asset are already sitting in this codebase's own signal outputs (sector
membership from `data/sectors.py`, carry level from `signals/carry.py`,
momentum/trend state, realized vol from Yang-Zhang) rather than needing new
data collection, unlike DCC-GARCH's `arch`-based per-asset GARCH fitting. Not
built, same "next-step candidate, not started" status as DCC-GARCH — not
preferred over it a priori; whichever (if either) actually beats Ledoit-Wolf
on this panel is an empirical question per `cleanup.md` section 3's framing,
same discipline applied to DCC-GARCH above.

**Explicitly NOT a fit for the multiple-testing/effective-trials problem
below — considered and rejected as a mismatch, logged so this doesn't get
re-proposed later without re-deriving why.** IPCA needs an `L`-dimensional
characteristics panel per individual and outputs a structural `Γ` loading map
— it answers "what drives asset i's exposure to factor k," not "how many
genuinely independent trials are hiding inside 50 correlated grid points for
one Book." The effective-number-of-independent-trials question below is
answered by plain eigenvalue decomposition of the trial-return correlation
matrix (Cheverud 2001 / Nyholt 2004), which IPCA doesn't produce as an output
at all — IPCA is a strict generalization of plain PCA built to answer a
different question (dynamic factor loadings conditioned on observables), not
a tool for counting effective degrees of freedom in a correlated hypothesis
set.

### Per-Book hyperparameter tuning — attempted 2026-07-23, real negative result: don't adopt

Currently every Book shares one flat, explicitly-labeled-not-tuned calibration
(`GAMMA`/`KAPPA`/`LAMBD`/`MAX_WEIGHT`/`BOOK_TARGET_VOL`/`EWMA_HALFLIFE`/
`SCALE_MIN`/`SCALE_MAX`, `research/portfolio.py`'s own dimensional-sanity
constants, reused verbatim across every later research script). Per direct
instruction, tuning these per-Book (using train to fit, validation to select,
test untouched — the same discipline `backtest/splits.py`'s own docstring
already establishes for signal-level parameters like breakout's window length
or crossover's MA pairs, extended one layer up to the optimizer) was made top
priority, ahead of the DCC-GARCH work above.

**Blocker identified and fixed first: monthly Book rebalancing gave
validation far too few observations to tune against safely** — only ~22-24
monthly PnL points in the 2020-2021 validation window per Book. Fix, per
direct instruction: **moved Book rebalancing to weekly**, in `research/
value_momentum_combine.py`, `research/signal_correlation.py`, and the new
`research/tune_book_hyperparameters.py` (NOT in `research/portfolio.py`'s
original 6-Book pass — that result stays as previously logged, out of scope
here). Checked directly before committing to this (not assumed):

- Mechanically viable — `portfolio.covariance.build_cov_dict`'s `freq` param
  already generically supports `"W-FRI"` (`real_period_end_dates` is
  freq-agnostic), and compute cost is trivial (0.3-0.4s to build a full
  38-asset-panel weekly `cov_dict`, vs. 0.1s monthly).
- Real effect on usable Book dates, measured on the actual value/XSMOM active
  universes (24 assets each): value goes from 51 train / 22 validation
  (monthly) to 196 train / 89 validation (weekly); XSMOM from 58/22 to
  256/98. Roughly the expected ~4.3x (52/12), not diluted — validation alone
  goes from 22 to ~90-98 points, enough to actually tune against.
- **One silent-bug risk found while checking this, not yet fixed**:
  `EWMA_HALFLIFE` (currently 20, used in `Book._apply_vol_target`) is defined
  in REBALANCE-PERIOD units, not calendar time — 20 monthly periods ≈ 1.7
  years, but 20 WEEKLY periods ≈ 4.6 months. Switching `COV_FREQ`/
  `PERIODS_PER_YEAR` to weekly while leaving `EWMA_HALFLIFE=20` unchanged
  would silently make the realized-vol tracker ~4.3x more reactive in
  calendar-time terms than currently intended — needs rescaling (~87 periods
  to preserve the same ~1.7yr calendar halflife) or a deliberate fresh choice,
  not a silent carry-over.
- **A real, not-yet-resolved statistical caveat**: value's and XSMOM's own
  alpha only updates monthly (both deliberately monthly-cadence per their
  source paper — Asness-Moskowitz-Pedersen). Weekly Book rebalancing adds
  real information for the parameters actually being tuned (`gamma`/`kappa`/
  `target_vol`/`ewma_halflife`/scale bounds are all optimizer-REACTIVITY
  parameters — how aggressively to size against Σ_t and realized vol, both of
  which genuinely do move at weekly frequency even under a static alpha) —
  but the resulting weekly PnL series will be autocorrelated within each
  month (same alpha, same position, ~4-5 consecutive weeks), so Sharpe-based
  hyperparameter selection on it should account for that (e.g. a Newey-West-
  style correction, the same discipline already used for short-term
  reversal's VIX regression), not treat each week as an independent draw.
- Minor measured side-effect: the joint-completeness (`_active_columns`/
  `alpha_df.dropna()`) usable-date fraction is slightly worse at weekly
  cadence (value: 74% of weekly cov dates usable vs. 82% monthly) — still a
  large net gain in absolute observation count, not a blocker, just logged so
  it isn't rediscovered as a surprise later.

**Effect on the mix-vs-integrate result (`research/value_momentum_combine.py`)
from the weekly switch alone, before any tuning — superseding the monthly-
cadence numbers logged above and in `CLAUDE.md`'s "Mix vs. integrate (Value +
XSMOM)" row**: Sharpe (train/validation/test, gross) — Mixed goes from
-0.34/-0.80/**+0.04** (monthly) to -0.34/-0.75/**-0.17** (weekly). Train
essentially unchanged, validation slightly better, but **test flips from
marginally positive to negative** — a real, not cosmetic, change in the
headline read, found simply by switching rebalance cadence with the SAME flat
hyperparameters, before any tuning was attempted.

**Tuning procedure, `research/tune_book_hyperparameters.py`**: pre-committed
5x5 grid over `target_vol` (0.05-0.15) x `max_weight` (0.15-0.5) — chosen
before looking at any result, same discipline as every other spec search in
this project — evaluated per Book (value, XSMOM), selected by best VALIDATION
Sharpe (the same primary criterion used everywhere else here), full grid
reported not just the winner, TEST touched once for the winning combo only.
Alongside the naive Sharpe, reports a Newey-West HAC t-stat (`maxlags=8`,
~2 months of weekly data, scaled down from Nagel's own `maxlags=20` for daily
data via `research/short_term_reversal.py`'s already-established
`statsmodels` HAC convention) for whether the validation-period mean weekly
PnL is actually distinguishable from zero — built specifically to check the
autocorrelation caveat flagged above, not decoration.

**Result: real, and negative — tuning overfits to the validation window, and
the HAC check catches it.** Value's grid selected `target_vol=0.125,
max_weight=0.15`: validation Sharpe jumps from -0.06 (default 0.10/0.30) to
+0.79 — but that combo's TEST Sharpe is -0.42, WORSE than default's -0.20,
and the HAC t-stat on that validation Sharpe is only 0.97 (p=0.33) — not
statistically distinguishable from zero once autocorrelation is priced in.
The entire grid shows `max_weight=0.15` winning at every `target_vol` level —
tighter position caps happened to help specifically through the 2020-2021
COVID window (economically plausible on its own — caps limiting concentrated
losses in a crash — but exactly the kind of validation-window-specific
pattern that doesn't have to generalize, and here it measurably didn't).
XSMOM's grid is uniformly deeply negative in validation (-0.75 to -0.91,
HAC t between -1.2 and -1.6, still p>0.10 everywhere) — consistent with
XSMOM's own already-logged momentum-crash validation result; there's no good
sizing combo to find there, the signal itself struggled in that window.
Combined via Allocator, the same pattern repeats at the portfolio level:
Mixed-tuned's validation Sharpe (+0.17) beats Mixed-default (-0.75) by a wide
margin, but test flips from -0.17 (default) to **-0.34 (tuned)** — worse, not
better. Full grids in `Data/research/tune_book_hyperparameters/{value,
xs_momentum}_grid.csv`; summary in `default_vs_tuned_summary.csv`.

**Recommendation: do not adopt these tuned hyperparameters — keep the flat
default calibration.** This isn't a hedge; it's the actual outcome. The
overfitting risk flagged before running this (only ~90-100 validation
observations even at weekly cadence, a 2-parameter grid still has enough
freedom to fit validation-period noise) is exactly what happened, and the
HAC diagnostic built to check for it confirms the apparent improvement isn't
statistically real.

### Daily-marking the same weight path — added 2026-07-23, confirms rather than overturns the result above

Per direct instruction, answering a real question raised after the result
above: can this project's DAILY price data be used to get an adequate
tuning sample while still rebalancing (re-solving the optimizer) at a slower
cadence? Yes, via a genuinely different mechanism than the earlier monthly-
to-weekly cadence switch — added `portfolio.book.daily_mark_pnl(weights,
returns_df)`: forward-fills an already-solved Book weight path (whatever
cadence it was solved at) onto the DAILY return index, `.shift(1)`s it
(CLAUDE.md Rule 3, same convention `backtest.engine.normalized_positions`
already uses — without the shift, the weight solved ON a rebalance date,
using that date's own return, would be multiplied against that same day's
return, a real look-ahead), and marks it against real daily returns. This
decouples two things the earlier cadence switch conflated: how often the
OPTIMIZER re-solves (which changes the strategy's real turnover/reactivity —
a genuine behavior change) vs. how finely the resulting PnL is MEASURED
(which changes nothing about the strategy, only the precision of evaluating
it). Turns ~90-100 weekly validation points into ~500-620 daily ones, without
re-solving any more often.

**This does NOT manufacture new independent alpha decisions** — value's and
XSMOM's own alpha is still monthly-cadence, so daily marks of a month-long
mostly-unchanged position are themselves highly autocorrelated. That's
exactly why the HAC correction matters more here, not less: `DAILY_HAC_
MAXLAGS=25` (~1 trading month, the real redundancy length) vs. the original
`HAC_MAXLAGS=8` weeks for the period-level series.

**Result, re-run with daily-marked selection: the earlier finding holds up,
more rigorously, not less.** Value's grid still selects `target_vol=0.125,
max_weight=0.15` (daily-marked validation Sharpe -0.10→+0.55, HAC t=0.86,
p=0.39 — STILL not significant, now on 623 observations instead of ~90).
Test Sharpe for the selected combo is -0.49 vs. default's -0.66 — less bad
this time, not clearly better, both deeply negative. XSMOM's tuning barely
moves anything (daily-marked validation -0.486 default vs. -0.474 tuned,
HAC t=-0.52 vs. -0.49 — a wash, consistent with its own uniformly-bad grid).
Combined, Mixed-tuned's validation improves from -0.51 to +0.04 (near flat,
not clearly positive) and test from -0.75 to -0.64 (less bad, not good).
**Daily marking made the measurement far more precise and the conclusion
more trustworthy — not more favorable.** The apparent gains from tuning
still aren't statistically distinguishable from noise; a much better-
measured version of the same experiment confirms the original "don't adopt"
call rather than reversing it. Full results in `Data/research/
tune_book_hyperparameters/default_vs_tuned_summary.csv` (both weekly-marked
and daily-marked rows, side by side, not overwritten).

**On extending this to all ~20 Books across every signal family (per direct
instruction, raised as a question, not yet done)**: recommend NOT scaling
this up yet. Daily marking fixed the MEASUREMENT-precision half of the
problem, but the deeper one is untouched — 2020-2021 is a single, narrow
validation window, and no amount of marking granularity increases the number
of independent MARKET REGIMES it contains. Tuning ~20 Books x a 25-point grid
each (~500 "is this better than default" comparisons) without a multiple-
testing correction would very likely produce several apparently-significant
"improvements" by chance alone, and — per the actual result just found — that
risk is real, not hypothetical. Before scaling: either (a) a regularized
selection rule (require the HAC t-stat past a real threshold, and ideally a
multiple-testing correction such as Bonferroni/FDR across however many Books
are tested at once, before accepting ANY change from default), or (b)
genuinely more independent validation information — multiple non-overlapping
or expanding-window validation blocks spanning different market regimes, not
one fixed 2020-2021 window — before trusting a tuned result anywhere in this
project, not just for value/XSMOM.

**On the tuning METHOD itself (per direct instruction: "since we use an
optimizer we should use a good one")**: brute-force exhaustive grid search
(5x5=25 points per Book, all evaluated) — not Bayesian optimization, random
search, or a gradient-based method. For a 2-parameter, cheap-to-evaluate
space (each `Book.run()` call is ~1s), this is a reasonable, arguably
preferred choice — exhaustive (no risk of missing a region), fully
transparent/reproducible, and doesn't introduce its own extra hyperparameters
the way an acquisition-function-driven Bayesian search would. Important:
**the search algorithm was never the actual bottleneck here** — a smarter
search would have found the same statistically-insignificant "optimum"
faster, not a more real one; it does nothing to fix the underlying data-
scarcity/multiple-testing problem. A sample-efficient method (e.g. Bayesian
optimization via `optuna`/`scikit-optimize`) would become genuinely worth
adopting if/when the search space grows much larger (more parameters per
Book, or many Books tuned jointly) purely for COMPUTE reasons — grid search
doesn't scale past a handful of dimensions — but that's a separate concern
from statistical validity, and shouldn't be conflated with it.

Code: `research/tune_book_hyperparameters.py`, `portfolio/book.py`'s
`daily_mark_pnl`. Not yet done (at the time): the Allocator correlation-
weighting idea from the signal-correlation section above, and the multi-
window validation approach recommended above, before any tuning was scaled
beyond value/XSMOM — both addressed below except the Allocator idea, which
remains open.

### All 20 Books, properly corrected — built and run 2026-07-23, same conclusion, now decisively confirmed

Per direct instruction: scaled the tuning above to the full 20-Book roster
(`research/tune_all_books.py`, `build_all_signals()` — every alpha reused
verbatim from its own family's already-established `src/signals/` module, no
new signal-construction logic), with a properly designed two-stage multiple-
testing correction (requested explicitly, not assumed): **within-Book
Bonferroni** (best candidate's raw p-value x `GRID_SIZE`, controlling the
false-positive risk from picking the best of many candidates within one
Book's own search) **then across-Book Benjamini-Hochberg FDR**
(`statsmodels.stats.multitest.multipletests(..., method="fdr_bh")` on the 19
evaluable Books' Bonferroni-corrected p-values, controlling the expected
false-discovery proportion across the whole family of Books tested at once).
A Book's tuned hyperparameters are adopted ONLY if they survive BOTH stages.

**First pass (2D grid: target_vol x max_weight, no real costs, GRID_SIZE=25)
— 0 of 19 evaluable Books adopted.** Value came closest (raw p=0.0013,
Bonferroni p=0.032 — would have passed a single-Book FWER check — but FDR-
corrected across the 19-Book family, p=0.615, fails decisively). Several
others looked individually promising on raw p-values alone before correction
(breakout_system2 p=0.014, carry_timing_mean p=0.024, reversal_sector_10d
p=0.046) — exactly the false-discovery pattern the correction exists to
catch; none survived. (`carry_cross_sectional_1_12` excluded — only 18 dates
where alpha/covariance/returns jointly overlap, a real data-coverage limit
of that specific 12-month-smoothed spec, not a bug.)

**Same day, immediately following, per direct instruction: two more real
questions answered by building and testing, not reasoning alone.**

1. **Real transaction costs wired into `Book.run()` itself** — new `cost_bps`
   param on `Book` (`portfolio/book.py`), deducted via `backtest.costs.
   transaction_cost_drag` on the Book's own actual turnover, using the SAME
   liquidity-tiered ADV-based cost assumption every standalone signal's
   net-of-cost number in this project already uses — not a new/different cost
   model. Distinct from `lambd` (an ex-ante optimizer-smoothing penalty
   inside the objective) — `cost_bps` is an ex-post REALIZED cost deducted
   after weights are solved, the same real-cost concept as every other
   signal's gross/net split, now available at the Book/Allocator level for
   the first time. `portfolio.book.daily_mark_pnl` extended with the same
   `cost_bps` param (charges the real per-rebalance-date cost as a lump sum
   on that date's daily mark — needed for a FAIR cross-frequency comparison;
   without it, rebalancing more often could only ever look equal-or-better in
   a grid search, since more reactive would have no charged downside). Both
   changes covered by new regression tests (`tests/test_book.py`, 144 tests
   passing project-wide). **Explicitly verified, not just claimed**: `build_
   all_signals()` never references `cost_bps` anywhere in its bytecode
   (checked programmatically, `main()` asserts this at runtime) — every
   alpha fed to a `Book`/the `Allocator` stays gross; costs are deducted
   exactly once, inside `Book.run()`/`daily_mark_pnl()`, never pre-baked
   into a signal upstream.
2. **Rebalance frequency added as a third grid dimension** — `FREQUENCY_GRID`:
   monthly (`"ME"`, 12/yr, `EWMA_HALFLIFE=12`) and weekly (`"W-FRI"`, 52/yr,
   `EWMA_HALFLIFE=87`), each halflife rescaled to hold the same ~1.7yr
   calendar reactivity (the exact unit-scaling gotcha already found once for
   the original monthly->weekly switch, applied correctly here from the
   start). **Daily rebalancing deliberately excluded** — a real, stated
   compute-practicality tradeoff (a daily-cadence tier means ~2,600 rebalance
   dates per `Book.run()` call instead of ~125-540, several times slower, on
   top of being the most turnover-heavy option in any realistic scenario) —
   worth a dedicated, narrower follow-up, not bundled into this already-large
   run. `GRID_SIZE` = 5 x 5 x 2 = 50 (Bonferroni denominator doubled
   accordingly).

**Result, full 3D grid, cost-inclusive: 0 of 19 evaluable Books adopted —
the SAME conclusion, now more decisively confirmed, not reversed.** Value is
again the closest (diff_t=3.26, raw p=0.0011) but now fails even the
within-Book Bonferroni bar (p_bonf=0.056 vs. the 2D grid's 0.032 — the
larger GRID_SIZE, a direct, correct consequence of testing more candidates,
pushed it just past the 0.05 line) and its FDR-corrected p is 1.0. Every
other Book: p_bonf >= 0.66, most at the ceiling (1.0). `carry_cross_
sectional_1_12` failed at BOTH frequencies this time (monthly: 5 usable
dates; weekly: 18) — a real, if minor, finding that the 12-month-smoothed
spec's own long lookback interacts worse with monthly cadence's fewer,
farther-apart candidate dates, not a new bug.

**On which frequency actually won, purely descriptively (not prescriptively,
since nothing survived correction)**: weekly was each Book's own best single
candidate in 14 of 19 cases, monthly in 5 (`carry_cross_sectional_1m`,
`reversal_sector_10d`, `reversal_individual_5d`, `reversal_individual_10d`,
`carry_timing_zero`) — a mild lean toward this project's current weekly
default already being reasonable, but none of these individual wins clear
even a naive uncorrected significance bar (best raw p among the monthly
winners is 0.035), so this is a descriptive footnote, not a finding to act
on.

**Every Book keeps its flat default calibration** (target_vol=0.10,
max_weight=0.30, weekly rebalancing) — unchanged from before real costs and
frequency were added to the test. This is the strongest form of the result
so far: it survived a materially harder, more realistic test (real trading
costs, a genuine structural dimension added) without flipping.

Code: `research/tune_all_books.py` (now cost-inclusive, 3D grid — supersedes
its own first, 2D/no-cost run; both preserved above). Full grids in `Data/
research/tune_all_books/full_grid_all_books.csv`, per-Book candidate
selection and both correction stages in `candidate_selection.csv`. Not yet
done: the Allocator correlation-weighting idea (signal-correlation section
above), a dedicated daily-rebalancing check, and multi-window/expanding-
window validation (a genuinely different, not-yet-tried way to get more
INDEPENDENT information, as opposed to the measurement-precision and
correction-rigor improvements made so far).

### Beyond naive Bonferroni/FDR — next step, not yet built, discussed 2026-07-23

Real objection raised, and correct: naive Bonferroni/BH scales so harshly
with the number of trials (m) that no real multi-strategy shop could ever
adopt anything tested this way — a fund running hundreds of strategies isn't
passing each one through a per-strategy 5%-FWER gate. What's actually done in
practice splits into three DIFFERENT roles, not five competing alternatives
to "test and compare" against each other — several of the five ideas
discussed are overlapping fixes to the SAME thing, not independent options:

1. **The data problem (upstream of everything else)**: our real bottleneck is
   one fixed, narrow validation block (2020-2021) — no correction math turns
   one regime into many independent looks. Fix: **purged/embargoed walk-
   forward, ideally Combinatorial Purged Cross-Validation (CPCV)** — many
   train/test recombinations across the FULL ~15yr history, purging any
   training observation whose outcome window overlaps a test observation's,
   plus an embargo buffer after each test block for residual serial
   dependence. Source: **López de Prado, M. (2018), *Advances in Financial
   Machine Learning*, Wiley — Ch. 7 ("Cross-Validation in Finance") and
   Ch. 11-12**; **Bailey, D. H., Borwein, J., López de Prado, M., & Zhu,
   Q. J. (2017), "The Probability of Backtest Overfitting," *Journal of
   Computational Finance* 20(4), 39-69** (introduces Combinatorially
   Symmetric Cross-Validation and the Probability-of-Backtest-Overfitting
   metric).
2. **The estimation/decision problem, given whatever evidence exists**:
   replace the binary Bonferroni+FDR pass/fail gate (which degenerates to
   "0 of 19 adopted" and throws away partial evidence) with **empirical Bayes
   / James-Stein shrinkage** — pull each Book's own grid-selected
   hyperparameters toward the shared default, weighted by how noisy that
   Book's own estimate is, rather than requiring it to clear a threshold at
   all. A more natural fit for `target_vol`/`max_weight` specifically, since
   they feed a continuous optimizer, not an on/off switch. Sources: **James,
   W., & Stein, C. (1961), "Estimation with Quadratic Loss," Proc. 4th
   Berkeley Symposium on Mathematical Statistics and Probability**
   (foundational shrinkage result — the sample mean is inadmissible for 3+
   simultaneous estimates); **Efron, B., & Morris, C. (1975), "Data Analysis
   Using Stein's Estimator and its Generalizations," *JASA* 70(350),
   311-319** (empirical Bayes framing — estimate the prior from the data
   itself); **Jorion, P. (1986), "Bayes-Stein Estimation for Portfolio
   Analysis," *JFQA* 21(3), 279-292** (the direct finance application —
   shrinking noisy per-asset estimates toward a grand mean before
   optimization, same idea proposed here for target_vol/max_weight). The
   shrinkage INTENSITY should itself be calibrated using the effective (not
   raw) number of trials — **Cheverud, J. M. (2001), "A Simple Correction for
   Multiple Comparisons in Interval Mapping Genome Scans," *Heredity* 87(1)**
   and **Nyholt, D. R. (2004), "A Simple Correction for Multiple Testing for
   SNPs in Linkage Disequilibrium with Each Other," *American Journal of
   Human Genetics* 74(4)** (eigenvalue-decomposition-based effective-N,
   originally genetics, imported into finance multiple-testing arguments by
   **Harvey, C. R., Liu, Y., & Zhu, H. (2016), "…and the Cross-Section of
   Expected Returns," *Review of Financial Studies* 29(1)**) — our 50
   correlated grid points per Book are nowhere near 50 independent trials,
   and treating them as such (as the Bonferroni pass above did) is itself a
   real error, not just overly conservative.
3. **The reporting frame**: track the AGGREGATE Allocator-level Sharpe after
   shrinkage-based sizing across all 20 Books, not a per-Book pass/fail
   count — the diversified combination of many individually-weak, imperfectly
   correlated bets is the thing that's actually supposed to be robust, not
   any single Book in isolation. Source: **Grinold, R. C. (1989), "The
   Fundamental Law of Active Management," *Journal of Portfolio Management*
   15(3), 30-37**; **Grinold, R. C., & Kahn, R. N. (2000), *Active Portfolio
   Management* (2nd ed.), McGraw-Hill** — IR ≈ IC × √BR (breadth: the number
   of INDEPENDENT bets). Caveat sharpened in **Clarke, R., de Silva, H., &
   Thorley, S. (2002), "Portfolio Constraints and the Fundamental Law of
   Active Management," *Financial Analysts Journal* 58(5)**: breadth only
   scales as claimed if the bets are genuinely independent — the same
   correlation-among-trials thread running through all four items above.

**Optional secondary check, not a primary method**: the **Deflated Sharpe
Ratio** (**Bailey, D. H., & López de Prado, M. (2014), "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-
Normality," *Journal of Portfolio Management* 40(5), 94-107**, building on
**Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier,"
*Journal of Risk* 15(2)** for the Probabilistic Sharpe Ratio, and **Mertens,
E. (2002)** for the non-normal-returns Sharpe variance formula) tests a
best-of-N-trials Sharpe against the extreme-value-theory expected maximum
under a true null, self-adjusting for trial correlation via the empirical
variance of the observed trial Sharpes — largely REDUNDANT with the
effective-trials correction above (both correct for the same thing via
different mechanics), so worth running once as a second opinion on the final
result, not built as a competing parallel pipeline.

**Recommended sequencing, discussed 2026-07-23, not yet started**: (1) CPCV
fold generation + re-run the 20-Book tuning grid across many folds instead of
one validation block — the biggest real fix, and the one everything else
depends on; (2) empirical-Bayes shrinkage of target_vol/max_weight toward
default, calibrated with the effective-trials-corrected uncertainty from that
expanded evidence; (3) report via aggregate Allocator Sharpe, not a per-Book
adopt/reject count; (4) DSR as a cheap final sanity check. Per this project's
own discipline (read sources directly before implementing, not from memory —
CLAUDE.md), the relevant papers above should be read directly before any of
this is built, not implemented from summary. Alternative faster path floated
but not decided: build (2) first against the CURRENT single-validation-window
evidence as a nearer-term improvement over the discrete gate, before tackling
CPCV's larger implementation lift — open question, not yet resolved.

**Open question resolved 2026-07-23: CPCV first, not shrinkage first.**
Shrinkage intensity has to be calibrated by *something* measuring how
noisy/reliable each Book's tuned estimate is — and the only evidence
available before CPCV exists is still the same one fixed 2020-2021
validation block this whole section opens by diagnosing as the actual
bottleneck ("no correction math turns one regime into many independent
looks"). Building shrinkage against that evidence first wouldn't fix that
diagnosis, it would inherit it: the resulting shrinkage weights would still
be calibrated on a single-regime noise estimate, just expressed as a
continuous pull-toward-default instead of a binary pass/fail — a smoother
symptom patch, not a root-cause fix. That's the same category of mistake
this project has already caught itself making and corrected elsewhere (carry:
the real fix was rebalancing cadence, not a better cost model layered on top
of the wrong cadence; the 16-month covariance-gap bug: the real fix was
`max_gap_days`, not smoothing over the resulting equity-curve spike). CPCV is
explicitly named as "the one everything else depends on" in the sequencing
above — that dependency is real, not just ordering preference, so it goes
first.

**Sequencing this against the project's own established "validate small,
then scale" precedent** (used for the first Book/Allocator pass — 6
representative Books before the full 20 — and for per-Book tuning itself —
2D grid before the cost-inclusive 3D grid): build CPCV fold generation and
prove the mechanics are right (purging + embargo actually remove the
overlap they're supposed to, fold count/size are sane given this project's
~15yr history) against a SMALL subset of Books first — not all 20
immediately. Once that's validated, decide whether to (a) scale CPCV to all
20 Books before touching shrinkage, or (b) layer empirical-Bayes shrinkage
on top of the small-subset CPCV evidence as an intermediate step before the
full scale-up — a decision to make with real CPCV output in hand, not
pre-committed here.

### CSCV/CPCV built and run on 3 Books — 2026-07-23, genuinely new evidence, not a simple confirmation either way

**Built `src/backtest/cpcv.py`** (7 new unit tests, 151/151 project-wide
passing) implementing Bailey, Borwein, López de Prado & Zhu (2017)'s CSCV
Algorithm 2.3 exactly (paper fetched and read directly, not from memory —
`references/` has no local copy of this one, so it was pulled from the
author's own site, `davidhbailey.com/dhbpapers/backtest-prob.pdf`), plus a
purge/embargo refinement grounded in skfolio's `CombinatorialPurgedCV`
parameter docs (fetched directly) since López de Prado's *AFML* book itself
— the primary source for purging/embargo specifically — was not accessible
this session; that gap is stated explicitly in the module's own docstring
rather than papered over with an unsourced paraphrase. See `src/backtest/
cpcv.py`'s docstring for the full mechanics and, importantly, why purging is
being reinterpreted (not applied identically to the classical ML-labeling
motivation) for this project's specific `Book` architecture — nothing here
is "refit" per fold; a grid point's weight path is solved ONCE over the full
history, so purge/embargo exist to remove day-to-day autocorrelation
adjacency at IS/OOS boundaries (position inertia, EWMA vol state), not
label-overlap leakage into a fitted estimator.

**Small-scale first pass, per the sequencing decided above**:
`research/tune_books_cpcv.py`, 3 pre-committed Books (`momentum_12mo` —
the project's own flagship spec; `breakout_system1` — continuity with the
original 6-Book portfolio pilot; `value` — the Book that came CLOSEST to
surviving `tune_all_books.py`'s own Bonferroni/FDR gate, the most
informative case to re-examine), weekly-frequency-only 25-point
(target_vol x max_weight) grid, `n_groups=8` (C(8,4)=70 combinations),
`purge_periods=embargo_periods=5` trading days. Reuses `tune_all_books.py`'s
own data/signal/Book construction by direct import — the first time one
`research/*.py` script imports another, a deliberate exception to avoid
copy-pasting the already-canonical 20-Book construction.

**Result — real, and more informative than a simple "confirms 0/19" or
"reverses it" headline:**

| Book | PBO | median logit | default win-rate (IS) |
|---|---|---|---|
| momentum_12mo | 0.086 | 1.20 | 0.0% |
| value | 0.243 | 1.50 | 0.0% |
| breakout_system1 | 0.371 | 1.00 | 0.0% |

All three PBOs are well below 0.5 — across 70 genuinely different
combinatorial recombinations of the full ~2006-2026 history (not one 2020-
2021 block), the grid's own IS-optimal choice beats the OOS median far more
often than a coin flip, for every one of the three Books tested. That's a
real, different finding from "0 of 19 adopted": PBO measures whether the
SELECTION PROCESS itself (the whole target_vol x max_weight ranking) is
internally stable and generalizes across regimes, which is a different
question from "does the single best candidate beat the single flat default
by a margin that clears a paired significance test in one specific window" —
the question `tune_all_books.py`'s Bonferroni/FDR pipeline actually asked.
Low PBO here is genuine evidence the earlier "0/19" result wasn't simply
because target_vol/max_weight are meaningless dials — the ranking across
grid points is real and reasonably stable.

**Two honest caveats that keep this from being a green light to adopt
tuned parameters, flagged explicitly rather than smoothed over:**

1. **The flat default (target_vol=0.10, max_weight=0.30) never once won
   as the IS-optimal choice — 0 of 70 combinations, for all 3 Books (210
   total combo-Book evaluations).** Low PBO says the winning choice
   generalizes; it does NOT say the winning choice's edge over the CURRENT
   default is large or robust to noise — those are different claims, and
   `tune_all_books.py`'s own HAC-tested "does tuning beat default"
   comparison is still the right tool for that second question, not this
   one. A "default never wins" pattern this consistent across three
   unrelated signal families is also worth treating as a flag to
   investigate on its own terms (does cost-inclusive Sharpe on this daily-
   marked measure have a structural, not-purely-alpha-driven relationship
   with target_vol/max_weight — e.g. through how the vol-target scale cap
   or cost drag scale with leverage — rather than assuming it's "real edge
   the default is leaving on the table"), not proof tuning should be
   adopted wholesale.
2. **breakout_system1's PBO (0.37) is the least reassuring of the three,
   closest to the noise-like 0.5 threshold** — a genuinely different
   picture from `tune_all_books.py`'s single-window result, where `value`
   (not breakout) was the closest-to-surviving Book. CPCV surfaced this
   because it tests generalization across MANY regimes, something a single
   fixed validation window structurally cannot do — exactly the point of
   building this.

**Not yet done, explicitly**: scaling to the full 20-Book roster (the small
subset was a deliberate first step, matching this project's own "validate
small, then scale" precedent — the 3-Book PBO spread (0.09-0.37) is
reassuring enough that scaling to all 20 now looks reasonably well-motivated,
but that's a real compute/time commitment, not attempted in this same pass);
empirical-Bayes/James-Stein shrinkage (the sequencing decision above put
this after CPCV specifically so its shrinkage intensity could be calibrated
against genuine multi-regime evidence like this, rather than the single
2020-2021 window — that calibration work hasn't started); the
Cheverud/Nyholt effective-number-of-trials correction (a natural next
diagnostic on THIS CPCV output specifically — the 70 combinations' `logit`
values are themselves correlated with each other by construction, since
adjacent combinations share most of their group membership, so the
"effective" number of independent CPCV combinations is itself smaller than
70 and hasn't been estimated yet); DSR as a final sanity check. Full
combination-level output and logit histograms: `Data/research/
tune_books_cpcv/`.

### CPCV scaled to all 20 Books — 2026-07-23, the 3-Book pilot turned out to be unrepresentative

Per direct instruction, scaled the pilot above to the full roster
(`research/tune_all_books_cpcv.py`, a new script per this section's own
"validate small, then scale" precedent — the 3-Book pilot script is
preserved unchanged, not overwritten, same convention `tune_all_books.py`
already established relative to `tune_book_hyperparameters.py`). Identical
mechanics to the pilot (weekly-only, 25-point grid, `n_groups=8`,
`purge_periods=embargo_periods=5`) so results are directly comparable, not
re-parameterized.

**Result: the 3-Book pilot's reassuring PBO spread (0.09-0.37) does NOT
generalize to the full roster — a real, humbling finding, not a confirmation
of the pilot.** 19 of 20 Books were evaluable (`carry_cross_sectional_1_12`
skipped — 0 viable grid points, consistent with `tune_all_books.py`'s own
already-logged finding that this spec has too few usable dates at any
frequency, not a new bug). Full spread:

| Book | PBO | Book | PBO |
|---|---|---|---|
| reversal_sector_5d | 0.00 | momentum_24mo | 0.53 |
| momentum_12mo | 0.09 | crossover_ma_50_200 | 0.54 |
| crossover_ma_100_200 | 0.13 | reversal_individual_5d | 0.61 |
| carry_timing_zero | 0.17 | reversal_individual_10d | 0.64 |
| value | 0.24 | crossover_ma_50_100 | 0.69 |
| reversal_sector_10d | 0.26 | breakout_system2 | 0.69 |
| carry_timing_mean | 0.26 | reversal_individual_1d | 0.71 |
| reversal_sector_1d | 0.31 | carry_cross_sectional_1m | 0.71 |
| breakout_system1 | 0.37 | xs_momentum | 0.77 |
| | | momentum_3mo | 0.81 |

**Mean PBO 0.450, median 0.529 — essentially a coin flip on average, and
outright bad for roughly half the roster.** 10 of 19 Books have PBO > 0.5
(the IS-optimal grid point tends to underperform the OOS median — the
classic overfitting signature), 8 of those above 0.6, and `momentum_3mo`
(0.81) and `xs_momentum` (0.77) are close to the "actively perverse" end —
the IS-optimal choice on those two is WORSE than a coin flip at picking
something that generalizes, not merely uninformative.

**Why the pilot missed this, stated plainly rather than glossed over**: the
3 pilot Books were chosen for reasons independent of any CPCV result already
seen (flagship spec, portfolio-pilot continuity, closest-to-surviving under
the earlier Bonferroni/FDR gate) — a legitimate selection process, but one
that, in hindsight, happened to land on 3 of the more CPCV-favorable Books
out of 19. This is exactly the kind of thing a small pilot can miss and a
full scale-up is supposed to catch — not a failure of the pilot's design,
a reminder that "validate small, then scale" means the scale-up step is
load-bearing, not a formality.

**This actually STRENGTHENS `tune_all_books.py`'s original "0/19 adopted"
Bonferroni/FDR finding, via a completely different method reaching a
compatible conclusion** — independent evidence, not the same evidence
re-packaged: broad, genuine overfitting risk in the tuning-selection process
across most of this Book roster, not an artifact of one narrow 2020-2021
validation window being too harsh a gate. The original concern motivating
this whole CPCV detour (Bonferroni/FDR might be a false negative caused by
thin, single-regime evidence) does not hold up for MOST of this roster —
for most Books, the underlying selection process really does look
overfitting-prone even under many independent regimes.

**The "default never wins IS" pattern is now even harder to read as "real
edge the default is missing"**: 18 of 19 Books, including the ones with the
WORST PBO (momentum_3mo, xs_momentum, reversal_individual_1d/10d) — Books
where the IS-optimal choice demonstrably does NOT generalize — still never
once picked the flat default as IS-optimal. If picking a non-default config
were capturing real, generalizable skill, it should track PBO (better PBO
Books "deserve" their non-default pick more than worse-PBO ones); instead
the pattern is UNIVERSAL regardless of whether that pick actually
generalizes. This is now the stronger candidate explanation, not just a
flagged possibility: something in how cost-inclusive, vol-target-capped
daily-marked Sharpe is computed on this grid likely mechanically disfavors
`target_vol=0.10, max_weight=0.30` specifically, independent of real
market-timing skill. **Not yet diagnosed** — worth resolving before any
further downstream use of this grid's "winner" identity, though it doesn't
change the PBO-based overfitting conclusion above, which stands on its own.

**One standout, flagged rather than taken at face value**:
`reversal_sector_5d`'s PBO is exactly 0.000 — the IS-optimal choice beat the
OOS median in all 70 combinations. This could be a genuinely strong,
consistent effect (short-term reversal's sector tier already showed a real,
HAC-significant VIX-conditioning result earlier in this project — see
CLAUDE.md's Short-term reversal row), or it could reflect an unusually
narrow/degenerate PnL distribution for this specific Book making the
ranking artificially clean. Not investigated further this session — a
genuine PBO=0.000 across 70 independent-ish combinations is unusual enough
to be worth a dedicated look before treating it as a real finding, not
unusual enough on its own to distrust the rest of this run.

**Recommendation given this result: do NOT proceed to empirical-Bayes
shrinkage assuming most Books have solid, generalizable tuning evidence to
shrink FROM.** For roughly half the roster the honest input to shrinkage
would be "very little real signal in the tuned direction, shrink hard
toward default" — which is close to just keeping the existing flat-default
policy for those Books, not a meaningfully different outcome from today's
"0/19 adopted." The Books with genuinely low PBO (`reversal_sector_5d`,
`momentum_12mo`, `crossover_ma_100_200`, `carry_timing_zero`, `value`) are
the ones where shrinkage could plausibly do something other than confirm the
status quo — a narrower, more defensible next step than a blanket 20-Book
shrinkage pass would be. Next steps, in order: (1) diagnose the "default
never wins IS" pattern — real structural driver vs. measurement artifact —
since it currently muddies interpretation of every "most frequent IS-winner"
value reported here; (2) spot-check `reversal_sector_5d`'s PBO=0.000 result
directly; (3) only then consider empirical-Bayes shrinkage, scoped to the
handful of low-PBO Books rather than all 20. Full per-Book combination-level
output and logit histograms: `Data/research/tune_all_books_cpcv/`.

### Gerber statistic covariance — built 2026-07-31, honest negative diagnostic result

Raised directly by the user: `references/The Gerber Statistic.pdf` (Gerber, Markowitz,
Ernst, Miao, Javid, Sargen — final version, July 2021), read directly before scoping
this. Proposed as a third covariance-estimator candidate alongside the current
Ledoit-Wolf default (`portfolio.covariance.build_cov_dict`), following the exact same
"diagnostic comparison first, live-integration only with decisive evidence" pattern
already used for GARCH vol-targeting (decision #11) and DCC-GARCH sleeve covariance
(decision #12).

**What the paper does, verified by reading it directly, not from memory**: for each
asset k, a threshold `H_k = c * s_k` (`s_k` = that asset's own sample std dev; the paper
sweeps `c` in {0.5, 0.7, 0.9} as a robustness check, not a single fixed choice). Each
date/asset is classified Up (>= +H_k), Down (<= -H_k), or Neutral. A pair is
**concordant** if both pierce their threshold in the same direction, **discordant** if
opposite directions — anything touching Neutral is excluded from the count entirely
(the "noise-stripping" the whole statistic is built around). The paper's own actual
formula (Eq. 11 — a naive earlier version, Eq. 4, is NOT positive-semidefinite-safe and
is not used in any empirical result):

```
g_ij = (n_UU + n_DD - n_UD - n_DU) / (T - n_NN)
```

Denominator is `T` minus only the both-neutral count (not `n_concordant + n_discordant`)
— empirically always PSD in the paper's own tests (9 clean asset-class indices, not
proven as a theorem). `Σ_GS = diag(σ) @ G @ diag(σ)`, same construction as converting a
correlation matrix to a covariance matrix.

**Paper's own backtest** (context, not something being reproduced 1:1): 9 broad
asset-class indices, monthly returns, 1990-2020, long-only turnover-penalized
mean-variance optimization, 24-month rolling lookback, 5 risk targets x 3 thresholds (15
scenarios). Gerber beat historical/sample covariance in all 15 scenarios, beat Ledoit-Wolf
in 14 of 15 (lost only at c=0.5, the most conservative 3% risk target).

**Two disclosed differences from the paper, stated up front, not discovered mid-build**:
(1) their win was demonstrated on clean, monthly, fully-overlapping data across 9 broad
indices — this project's own universe (~31-42 sparse, ragged-history commodity/FX/rates
futures) is a much noisier setting their own evidence doesn't automatically transfer to;
(2) their own optimizer is a plain long-only MVO, not this project's turnover-penalized
Book optimizer running at WEEKLY cadence (`Book`'s own `COV_FREQ="W-FRI"`) — Gerber will
be computed at that same weekly cadence here, not literally monthly like the paper.

**Implementation outline**:

1. **`src/portfolio/gerber_covariance.py`** (new, sibling to `covariance.py`/
   `sleeve_covariance.py`): `gerber_correlation(returns, c=0.5)` (NaN-safe U/D indicator
   matrices via matrix multiply — `N_UU=U'U`, `N_DD=D'D`, `N_UD=U'D`, `N_DU=D'U`,
   `N_NN` from a separate neutral-indicator matrix; **per-pair `T_ij`, not one global T**
   — this project's real data has ragged joint histories, unlike the paper's clean
   overlapping dataset), `gerber_covariance(returns, c=0.5)` (`diag(sigma) @ G @
   diag(sigma)`), and `build_gerber_cov_dict(returns, window, freq, c=0.5)` — same
   date-indexed dict shape as `build_cov_dict`, a drop-in alternative, at the Book's own
   weekly cadence. **Explicit PSD check + eigenvalue-clipping fallback** — the paper only
   observed PSD empirically on 9 clean assets; do not assume it holds unchecked on ~40
   sparse futures. All three thresholds (c=0.5/0.7/0.9) built as parallel specs per the
   paper's own robustness sweep, fixed a priori — no cherry-picking after seeing results
   (CLAUDE.md Rule 1/2).
2. **`research/covariance_estimator_comparison.py`** (new): three estimators (rolling
   sample covariance, Ledoit-Wolf [current default], Gerber x3 thresholds) on Trend's own
   adopted, compressed universe (already the most relevant, already-decided universe — no
   need to invent a new one). Metric, matching page 06's own precedent rather than just
   eyeballing matrices: at each rebalance date, form a reference portfolio (e.g.
   minimum-variance) from each estimator's Sigma, compare forecast portfolio variance at
   formation vs. realized portfolio variance over the following period — the multivariate
   analogue of page 06's QLIKE-against-realized-variance test. Secondary diagnostic
   matching page 22's style: pairwise correlation time series for a few representative
   pairs across calm vs. stressed regimes. Cached to `Data/research/...parquet`, same
   precompute-once convention as pages 06/22.
3. **New Technical Appendix page (25)**: same shape as pages 06/22 — comparison
   table/chart, condition number over time, the forecast-accuracy metric, plain-language
   conclusion.
4. **`tests/test_gerber_covariance.py`** (new): `g_ii == 1` exactly (provable
   algebraically — U and D are mutually exclusive for an asset against itself); bounded in
   [-1, 1]; reduces to Kendall's Tau at `c=0` (a clean, checkable special case of Eq. 11's
   specific denominator); PSD-check/fallback exercised on a deliberately ragged synthetic
   panel; per-pair `T_ij` handling verified for assets with different valid-date coverage.

**Explicitly out of scope for this pass**: swapping any live Book's actual optimizer
covariance. That's a separate, later decision, only pursued if stage 2's diagnostic shows
a real forecast-accuracy edge — and would then need the same realized-Sharpe/turnover/
drawdown re-test every other estimator swap in this project got before touching
`single_strategy_portfolios.py`'s actual Book construction.

**Built, all 4 steps, same day.** `src/portfolio/gerber_covariance.py` (`gerber_correlation`,
`gerber_covariance`, `build_gerber_cov_dict`, `drop_until_complete`/`_nearest_psd_correlation`
— the latter renamed from a private helper to a public, reusable one once
`research/covariance_estimator_comparison.py` needed the identical NaN-cleanup logic for
Ledoit-Wolf/sample-covariance matrices too, CLAUDE.md Rule 6) + 14 new unit tests
(`tests/test_gerber_covariance.py` — `g_ii == 1` exactly, bounded in [-1,1], the `c=0`
Kendall's-Tau-a reduction, per-pair `T_ij` verified against a deliberately gapped asset,
the all-neutral NaN case, PSD-clip no-op/fix behavior, `drop_until_complete`'s worst-offender
removal). 324 tests passing project-wide (310 including these 14, plus pre-existing).

**A real universe gap found live, not assumed**: computing both Ledoit-Wolf and Gerber
directly on Trend's own 31-asset compressed universe (WORKFLOW.md decision #13) at the
plan's stated weekly/252-day settings, `build_cov_dict` produced **zero** usable
rebalance dates — its row-wise `dropna(how="any")` gate needs every one of 31 assets
non-NaN in the same window, and several of those assets are only ~55-60% covered in any
252-day slice of this panel, so the union of everyone's scattered gaps eats every single
window. Gerber's own per-pair `T_ij` tolerance handled the identical raw 31-asset panel
far better (800 usable dates vs. Ledoit-Wolf's 0, confirmed live) — a real, disclosed
structural advantage of the per-pair design, exactly the motivation stated in the
Implementation Outline above, not a surprise. But it made the raw universe unusable for a
fair side-by-side comparison. Fixed by adding one more filter on top of Trend's
universe — assets with >= 90% overall non-NaN return coverage, the same threshold
`single_strategy_portfolios.py`'s own `_active_columns` already uses for its own Book
construction (reused, not re-guessed) — narrowing to 21 of the 31 assets. This is a
comparison-fairness fix, not a finding about which estimator to prefer: Gerber's own
practical advantage on the raw, ragged panel is real and worth remembering separately
from the accuracy question below.

**Forecast-accuracy result (`research/covariance_estimator_comparison.py`, cached to
`Data/research/covariance_estimator_{qlike,summary,pairwise}.{parquet,csv}`, Technical
Appendix dashboard page 25): Gerber does NOT beat Ledoit-Wolf.** Five estimators (rolling
sample, Ledoit-Wolf, Gerber c=0.5/0.7/0.9) scored by the multivariate QLIKE analogue
described in the Implementation Outline (global minimum-variance portfolio, forecast vs.
realized variance over the following week), on 651 common formation dates:

| Estimator | Mean QLIKE (lower=better) | Per-date win rate | Mean condition number |
|---|---|---|---|
| Sample | 0.888 | 45.9% | ~39,074 |
| **Ledoit-Wolf** | **0.629** | 25.5% | **~150** |
| Gerber c=0.5 | 0.716 | 9.7% | ~9,816 |
| Gerber c=0.7 | 0.739 | 8.1% | ~9,234 |
| Gerber c=0.9 | 0.750 | 10.8% | ~8,800 |

Ledoit-Wolf wins pooled-average QLIKE by a wide margin, and Gerber gets monotonically
*worse* as the threshold `c` increases from 0.5 to 0.9 (the opposite of "pick the most
conservative threshold for safety" intuition) — reported as found, not tuned after the
fact (Rule 1/2), and not re-litigated by cherry-picking a different threshold post hoc.
Ledoit-Wolf's shrinkage also produces by far the best-conditioned matrices (~150 vs.
Gerber's ~8,800-9,800 even after eigenvalue-clipping, vs. plain sample covariance's
~39,000) — expected, since shrinkage is specifically designed to fix conditioning and
Gerber's PSD fallback only clips negative eigenvalues, it doesn't shrink toward a
well-conditioned target the way Ledoit-Wolf does. One genuine tension worth keeping,
same pattern page 06 already documents for its own pooled-average-vs-win-rate pair:
plain sample covariance has the WORST pooled mean QLIKE but the BEST per-date win rate
(45.9%, nearly double Ledoit-Wolf's 25.5%) — plausibly because its much worse
conditioning produces occasional wild misses that drag the mean up without changing how
often it's merely "less wrong than the others" on an ordinary date.

**Per the plan's own explicit gate ("only pursue live-Book integration if this diagnostic
shows a real forecast-accuracy edge"), this result does NOT clear that bar** — Gerber is
not adopted for any live Book's optimizer covariance. `portfolio.covariance.build_cov_dict`
(Ledoit-Wolf) remains the default everywhere. `portfolio.gerber_covariance` is kept as a
validated, tested estimator (not deleted) for future re-evaluation if the universe or
cadence changes enough to revisit this.

**Extended past the plan's own gate, per direct instruction, same day — a disclosed
exception, not a re-litigation of the QLIKE result above.** Raised directly: forecast
variance accuracy isn't the only channel through which a covariance estimator could
still raise a REALIZED Book's Sharpe — e.g. (1) weight/turnover stability (Gerber's
threshold excludes small noisy moves from the concordant/discordant count entirely, so
its correlation estimate may move less window-to-window than Ledoit-Wolf's, which could
mean a smoother position path and lower net-of-cost turnover drag even without a
variance-forecast edge); (2) QLIKE here only scores ONE portfolio's (the global min-var
portfolio's) variance forecast — `Book`'s actual objective is a mean-variance trade-off
(`alpha - gamma*risk - kappa*turnover`) where Sigma's full off-diagonal structure shapes
which alpha gets crowded down or levered up, a return-side effect the risk-only
diagnostic above can't see; (3) outlier/noise robustness in the correlation estimate
itself, independent of the pooled-average variance-forecast score; (4) regime-specific
behavior a full-sample pooled average can wash out (a worse-in-calm/better-in-stress
profile nets to "loses on average" here but could still matter for realized tail Sharpe).
None of these are asserted as true — they're the reasons a real Book-level backtest is a
genuinely different question from the diagnostic above, not a redundant re-check, and
worth running despite the negative QLIKE result.

**Book-level follow-up, built and run 2026-07-31 (`research/gerber_book_performance.py`,
`gerber_xsmom_value_seasonality.py`, `gerber_integrated_value_xsmom.py`,
`gerber_sector_breakdown.py`): still not adopted — no clean rule found, strategy-specific
effects in both directions.** `single_strategy_portfolios.build_book` gained two backward
-compatible optional params for this (`cov_dict_builder=None` defaults to the existing
Ledoit-Wolf `build_cov_dict`; `cost_bps=None` defaults to gross, matching every existing
caller's unchanged behavior) — no existing script's output changed as a result of this
wiring. **NET of the same liquidity-tiered transaction costs every other net-of-cost
comparison in this project uses** (`backtest.costs.liquidity_tiered_cost_bps`, reused not
re-derived) — the first pass through this follow-up was gross-only, a real gap raised
directly and fixed before drawing any conclusion from turnover differences.

Seven Books tested (Ledoit-Wolf baseline vs. Gerber c=0.5/0.7/0.9, all three thresholds as
parallel specs per the usual discipline, not just the best-looking one), train/validation
/test net Sharpe:

| Book | Ledoit-Wolf (train/val/test) | Best Gerber (train/val/test) | Turnover: LW vs. Gerber |
|---|---|---|---|
| Trend `tsmom_alone` | 0.018 / **0.679** / 1.451 | 0.235 / 0.166 / 1.476 | 0.623 vs. 0.55-0.57 (lower) |
| Trend `tsmom_seasonal` | -0.015 / **0.690** / 1.449 | 0.223 / 0.183 / 1.509 | 0.630 vs. 0.55-0.57 (lower) |
| Carry `carry_timing_zero` | -0.624 / **-1.264** / 0.112 | -0.178 / **-0.327** / -0.036 | 1.256 vs. 1.02-1.08 (lower) |
| XSMOM | -0.244 / -1.161 / -0.097 | -0.169 / -1.114 / **0.285** | 0.542 vs. 0.74-0.88 (HIGHER) |
| Value | -0.682 / **0.541** / -0.303 | -0.019 / -0.054 / -0.250 | 0.314 vs. 0.37-0.38 (higher) |
| Same-month (economic-driver) | -0.240 / -0.442 / 0.385 | -0.324 / -0.562 / **0.596** | 0.158 vs. 0.14 (lower) |
| Integrated Value+XSMOM | -0.824 / -1.188 / **0.207** | -0.585 / -1.663 / -0.233 | 0.684 vs. 0.62-0.68 (flat) |

**No consistent rule survives contact with all seven.** Validation gets worse under Gerber
in 6 of 7 Books — Carry is the lone, large exception (validation -1.26 -> -0.33, its own
worst historical stretch cut dramatically). Test is a genuine mixed bag: better for XSMOM
and same-month, worse for Value/Carry/Integrated, roughly tied for both Trend flavors.
Turnover direction even flips between strategies (down for Trend/Carry/same-month, UP for
XSMOM/Value, flat for Integrated) — the "Gerber smooths turnover" mechanism from the
original diagnostic write-up does not hold universally, contrary to what the earlier
(Trend/Carry-only) partial result suggested before XSMOM/Value/Integrated were tested.
Both Trend flavors move together nearly identically under Gerber (as expected — CLAUDE.md
already documents `tsmom_alone`/`tsmom_seasonal` as statistically indistinguishable), a
clean confirmation the effect here is driven by the shared universe/covariance mechanics,
not by which Trend flavor sits on top of it.

**Per-sector breakdown** (`gerber_sector_breakdown.py`, coarse 4-group roll-up of
`data.sectors.SECTORS` — Commodities/Equities/Rates/FX — applied to XSMOM, Value,
Integrated, and Carry; an EXACT decomposition of each Book's own net PnL via
`asset_contributions` minus a per-asset transaction-cost allocation, not modeled, since
every Book's `LAMBD=0.0` leaves no penalty term to allocate — verified by a sanity assert
that every sector's net PnL sums back to the whole Book's own net PnL exactly, on all 8
signal x estimator combinations run): Carry's headline validation improvement under Gerber
is NOT spread evenly — it's concentrated in Commodities (-0.790 -> -0.126) and especially
Rates (-0.495 -> **+0.208**, a sign flip), while FX and Equities barely move or get
slightly worse. Cached to `Data/research/gerber_sector_breakdown.csv`.

**Multi-strategy Allocator combination** (`run_multi_strategy_combinations`, naive
equal-Book-risk `Allocator`, the same baseline construction `research/multi_strategy_
seasonality.py` already established — demonstrates directly that a different covariance
estimator per Book is already architecturally supported, since `Allocator` only ever
touches each Book's own already-solved `pnl`, per its own docstring): combining the
ADOPTED Trend(`tsmom_alone`)+Carry(`carry_timing_zero`) mandate under three pairings —

| Combo | Train | Validation | Test |
|---|---|---|---|
| Both Ledoit-Wolf (current mandate) | -0.387 | -0.796 | **0.773** |
| Both Gerber c=0.5 | -0.097 | -0.385 | 0.713 |
| Per-Book best by validation (Trend->LW, Carry->Gerber c=0.9) | -0.238 | **+0.036** | 0.521 |

The per-Book-best mix rescues validation from deeply negative to roughly flat, but test
gets WORSE (0.521 vs. 0.773) — and since the mix was itself SELECTED by validation Sharpe,
its own validation win is partly mechanical, not free evidence (the same selection-bias
caution this project's own hyperparameter-tuning work in Phase 7 already established).
Test is the honest read here, and on test the current all-Ledoit-Wolf mandate still wins
outright.

**Conclusion: Gerber is not adopted anywhere in this project.** The original diagnostic
gate (real forecast-accuracy edge) was not cleared. The extended Book-level investigation,
pursued anyway per direct instruction specifically because forecast accuracy isn't the
only channel through which an estimator could move realized Sharpe, also does not clear a
"clearly better" bar for any single Book or combination — it helps Carry meaningfully, hurts
Value and Integrated Value+XSMOM meaningfully, and is a mixed bag everywhere else, with no
predictive rule found (by strategy speed, cadence, or cross-sectional-vs-time-series
construction) that explains the pattern across all seven Books tested. `portfolio.
covariance.build_cov_dict` (Ledoit-Wolf) remains every live Book's covariance input, with
no exception. `portfolio.gerber_covariance` is kept as a validated, tested estimator (not
deleted) for future re-evaluation if the universe, cadence, or strategy roster changes
enough to revisit this.

**RV/cointegration spread strategies — a revised prior, logged for Phase 2d, not yet
tested (that signal family isn't built yet).** Raised and corrected directly: an RV book
with 5-6 spreads is NOT a "single pairwise covariance" problem — each spread is itself a
synthetic instrument with its own P&L series, and the optimizer needs the full N x N
covariance ACROSS SPREADS to size them jointly (crowding, netting, diversification), the
same estimation problem as Trend or Carry, just one level up (spread-to-spread instead of
asset-to-asset) — the initial framing understated this and was corrected before drawing
any conclusion. Revised prior: **Ledoit-Wolf likely still wins, for a sharper reason than
originally stated** — the classic RV/stat-arb tail risk is a cross-spread correlation
spike during a market-wide deleveraging shock (multiple "unrelated" spreads suddenly
moving together — the August 2007 quant-quake pattern is the textbook case), exactly the
kind of fast, magnitude-driven event a threshold-based estimator is structurally slow to
register (the same mechanism that hurt Trend and Value here). Working the other way:
Gerber's per-pair `T_ij` tolerance is genuinely relevant again for a multi-spread book
(each spread has its own warm-up/history, unlike a single fixed pair), and most of a
mean-reverting spread's life is quiet chop where Gerber's noise-stripping should help
turnover for free, the same way it helped Carry. Not testable until Phase 2d's RV spread
Book exists.

---

## Phase 8 — Risk management 🟡

Position/sector limits, drawdown circuit breakers, and a regime overlay re-derived for
trend-following — not ported from DCC-GJR crowding logic, which assumes crisis hurts the
strategy (untrue for trend; see `CLAUDE.md` Rule 7). `src/regime/interface.py`'s lookup
shape is already built (see Phase 5); the regime classifier itself — content, not
interface — is a separate research task, likely macro-driven (growth/inflation,
GSCPI supply-chain stress, yield-curve shape — data already collected and point-in-time
-correct per `DATA_SCHEMA.md` section 3, just unused) rather than the retired project's
correlation-spike regimes.

**Constraint and overlay gaps found vs. institutional practice — 2026-07-22.**
Cross-referenced against `references/Portfolio Construction for CTA and Managed
Futures Strategies.pdf` (see Phase 7's own entry, same date, for the
sizing/cost-awareness half of this same review). Two items from that report's
ranked institutional checklist land squarely in this phase, not yet built:
- **Sector/concentration constraints in the optimizer itself.** Sectors exist
  (`data/sectors.py`) and are already used inside signal construction
  (cross-sectional demeaning for reversal/carry/XSMOM), but there is no
  sector-level exposure cap anywhere in `Book`/`optimizer.py`. The report's own
  language for this: "not optional."
- **Stress testing and top-down risk overlays** (drawdown throttle,
  correlation-shock gross-exposure throttle) — the report ranks this ABOVE
  regime-dependent allocation (#7 vs. #8 on its priority list), meaning it should
  arguably be built ahead of the regime classifier itself once this phase starts.
  `Allocator.regime_lookup`'s interface (built Phase 5) can carry either — a
  throttle is just a coarse regime signal — so the interface doesn't need to
  change, only its first real caller does.

**Historical VaR / Expected Shortfall — built 2026-07-22, first risk-calculation
pass, per direct instruction.** `src/portfolio/risk_metrics.py` (`historical_var`,
`expected_shortfall`, `expanding_var`, `expanding_expected_shortfall`) — 4 pure
functions, no optimizer dependency, same pattern as `covariance.py`/`costs.py`.
Scope, per direct instruction: the COMBINED Allocator portfolio only (not
per-Book — a natural next step, not attempted here), historical/empirical method
(not parametric-Gaussian — CTA return distributions aren't normal: trend tends
toward positive skew, carry/reversal toward negative skew and fat tails, so a
normality assumption here would be exactly the kind of unvalidated shortcut this
project has avoided everywhere else), 95% confidence (the user's own call, "pretty
straightforward").

**Real design decision surfaced and flagged before building, not discovered
after**: `Book.run()`'s pnl is monthly-periodicity (one value per rebalance date),
the same trap already documented multiple times in this project (the 4.6x
annualization bug, the `_period_stats` docstring) — only ~120-150 monthly
observations exist total for the combined Allocator portfolio. A fixed ROLLING
window long enough to be meaningful (36-60 months) would leave a 95% tail estimate
with just 2-3 effective observations — too thin to trust. Used an EXPANDING window
instead (`min_periods=24`, ~2 years) — the same "rolling OR expanding window"
allowance `CLAUDE.md` Rule 2 already makes for exactly this situation. Even at 24
observations the tail is genuinely thin (~1.2 expected observations at 95%
confidence) — stated plainly in the code and dashboard caption, not hidden or
implied to be more precise than it is.

Wired into `research/portfolio.py` (writes `Data/research/portfolio/var_es.csv`)
and `dashboard/pages/13_portfolio_performance.py` (new section: full-sample
VaR/ES as metric tiles, plus a point-in-time expanding VaR/ES chart overlaid on
realized monthly PnL). Real result on current data: full-sample 95% VaR -10.2%,
ES -15.6% (monthly return units) on the combined Allocator portfolio. 7 new unit
tests (`tests/test_risk_metrics.py`) — quantile correctness against a hand-checked
distribution, ES-more-extreme-than-VaR tail ordering, ES-equals-mean-of-worst-5%,
empty-series handling, `min_periods` gating, and a point-in-time correctness test
(a shock added AFTER a given date must not move that date's already-computed
expanding VaR — the same look-ahead discipline this project tests for everywhere
else, e.g. `portfolio.covariance`'s own point-in-time construction). Full
`pytest tests/` clean (133 passed, up from 126). Both dashboard pages
re-verified exception-free via `streamlit.testing.v1.AppTest`.

**Not attempted in this pass, deliberately**: per-Book VaR/ES (only the combined
Allocator portfolio); rolling (vs. expanding) window; parametric or Monte Carlo
VaR; Transtrend's own exVaR framework (an extreme-risk measure specifically
designed to be less mechanically distorted by recent price shocks than standard
VaR — a real, named alternative in the institutional-practice report, worth
revisiting once this first pass is validated). Sector/concentration caps and
stress-testing overlays (this phase's other two logged gaps, above) are still
unbuilt — VaR/ES is a risk MEASUREMENT, not a risk CONTROL; it doesn't throttle
anything on its own.

**DCC-GARCH, re-scoped (2026-07-15, see `cleanup.md` section 3):** the tool itself
(dynamic, time-varying covariance/correlation) is a legitimate CTA risk-management
input — a correlation-regime spike is a real diversification-breakdown signal that
institutional trend books use to throttle *gross leverage*, independent of whether the
signal itself still works. That's a different role than the retired project's use
(correlation spike → suppress a specific book's alpha), which assumed crisis hurts
performance. If used here, it belongs in the covariance/risk layer (Phase 7's
optimizer input, or a gross-exposure throttle in the Allocator) — and whether
DCC-based Σ actually beats rolling Ledoit-Wolf at 41 assets (a harder estimation
problem than the 13-name case it was built for) is an empirical question for Phase 7,
not an assumption.

**Macro Data Explorer — built 2026-07-23, new dashboard nav category.** Per
direct instruction: a page to visualize every macro/auxiliary series collected
in `Data/` (6 sources — Yield Curve, Fed Funds, GSCPI, Trade Policy Uncertainty,
VIX, CPI), extensible to future additions, with a family selector, a
within-family series multiselect, and a timeframe control (1Y/5Y/10Y/Max/
Custom). `dashboard/pages/15_macro_explorer.py`, new "Macro Data" nav category
in `dashboard/app.py` (separate from page 04's "Macro" QA page, Coverage
category — page 04 answers "is data flowing, point-in-time-correct," this page
answers "let me explore/compare the series themselves," same kind of split this
project already uses between Coverage/Strategy Performance/Portfolio
Construction). Registry-driven (`MACRO_FAMILIES` dict — loader/unit/default
series/caption per family), same pattern as page 04's own `HISTORY_LOADERS` and
`jobs/update_macro_data.py`'s `UPDATERS` — a 7th macro source later is one new
registry entry.

**Deliberately does not support cross-family comparison on one chart** (e.g. VIX
vs. CPI) — units and scales are incompatible (VIX ~15-80 vol points, CPI
~100-330 index levels on different base years per country, yields in %), so a
free-for-all multiselect across all six would mostly produce unreadable charts.
Comparison is scoped to WITHIN a family (multiple yield maturities, multiple CPI
countries, multiple Fed Funds rate types), which is where the real value is.

**GSCPI gets its first-ever parser in this codebase.** `gscpi_data.xls` had only
ever been fetched as raw bytes (`jobs/update_macro_data.py`); `DATA_SCHEMA.md`
noted `xlrd` wasn't confirmed installed. Checked live 2026-07-23: it is (v2.0.2),
and the file parses cleanly (`skiprows=4`, columns renamed to `Date, GSCPI`) —
matches `src/macro_point_in_time.py::get_gscpi_as_of`'s own raw-load step
exactly, not reinvented. That module's lag-gating (`as_of()`) is for backtest
look-ahead safety and isn't relevant to a display-only chart, same reasoning
page 04 already used for its own raw-file reads.

**One real bug found and fixed via AppTest, not shipped silently**:
`plot_df.agg(["last", "min", "max", "mean"])` for the summary table crashed —
`"last"` collides with pandas' `NDFrame.last()` date-offset method (needs a
positional `offset` arg) rather than being treated as a plain reduction in this
pandas version. Fixed with an explicit `{"Latest": plot_df.ffill().iloc[-1], ...}`
construction instead, which also correctly handles mixed-frequency families
(e.g. CPI's quarterly AUD alongside monthly columns) by forward-filling each
series to its own latest known value rather than reading a possibly-NaN last row
directly. Verified exception-free via `streamlit.testing.v1.AppTest` across all
6 families plus the Custom timeframe path. Full `pytest tests/` clean (133,
unchanged — dashboard-only work).

**Not attempted in this pass**: the BACI international trade database, discussed
in the same conversation as a candidate 7th macro source. Deliberately deferred
— live-checked its real size first (`BACI_HS92_V202601.zip`, longest available
history: 2.4GB compressed; even the smallest/most recent HS vintage, HS22, is
300MB), which is categorically different from every other file in `Data/` and
would need a real download + pre-aggregation pipeline (likely scoped to
countries/commodities connected to this project's own 42-asset universe, not
the full ~200-country x ~5,000-product matrix) rather than a page reading a raw
file directly like the other 6 families here. Logged as a real follow-up, not
silently dropped.

---

## Phase 9 — Performance attribution ⚪

PnL decomposition by signal, asset class, and regime; formalize the walk-forward/held-out
protocol from Phase 1 as the standard evaluation harness for every future signal.

---

## Phase 10 — Live data & paper trading ⚪

Not a new backtest — a continuous out-of-sample validation loop. Sequenced last on
purpose: only produces something meaningful once Phase 5–7 give it real target weights to
validate.

**Design principle: parity.** The function that decides "what should today's position be"
must be the literal same function called in the backtest loop and in tomorrow's live run.
Separate backtest-mode/live-mode code paths defeat the purpose. This is the concrete
payoff of the Phase 1 refactor.

- **10a — Callable pipeline.** Depends on Phase 1: signal construction, vol-scaling, and
  target-weight logic become plain functions, importable from both the backtest script
  and the live-run script.
- **10b — Daily data-append job.** 🟡 Five thin scheduled jobs exist ahead of schedule
  (`jobs/`), all running as the logged-in user (needs an active/logged-in session at
  trigger time, not full logged-out background execution) and none yet using shared
  functions (there are none until Phase 1):
  - `CTA_DailyDataUpdate` (6:00PM) — `jobs/update_data.py`, core OHLCV. Standalone
    re-pull, not incremental (re-downloads full `period="max"` and overwrites, same as
    `get_data.ipynb` — deliberately simple rather than build dedup/append logic before
    Phase 1 exists).
  - `CTA_VolatilityUpdate` (6:05PM) — `jobs/update_volatility.py`, Yang-Zhang
    volatility. Pure function of the OHLC data above; must run after it.
  - `CTA_TermStructureCapture` (6:15PM) — `jobs/capture_term_structure.py`, individual
    contract-month prices. See Phase 4 — this one is append/dedup, not overwrite, since
    it can't be caught up retroactively once a contract expires.
  - `CTA_MacroDataUpdate` (6:20PM) — `jobs/update_macro_data.py`, the 4 macro/
    auxiliary sources. See `DATA_SCHEMA.md` section 3.
  - `CTA_DashboardSummary` (6:25PM) — `jobs/update_dashboard_summary.py`, added
    2026-07-15. Pre-computes the QA dashboard's summary artifacts
    (`Data/dashboard_summary/`) after the other 4 jobs finish. See 10e below.

  Revisit once Phase 1 lands: each script should import refactored pull/clean functions
  instead of duplicating logic, for parity with the backtest pipeline.
- **10c — Broker integration.** Interactive Brokers via `ib_insync` — a free paper account
  gives both a live quote feed and a fill venue in one integration (and doubles as the
  Phase 4 carry data source). Nontrivial step: translating a *target weight* into a
  *target contract count* requires the contract multiplier (e.g. 1,000 bbl/contract for
  WTI) and current price, then diffing against the current paper position.
- **10d — Backtest-vs-live comparison log.** Every day, log three numbers side by side:
  what the backtest (re-run as of today, using only data available through today) says
  the position should be; what the live pipeline actually produced; what the paper
  account actually realized. Divergence between the first two flags a code-parity bug;
  divergence between the last two is real-world friction (slippage, timing, data
  revisions). This divergence log *is* the deliverable, and doubles as a live-fire
  version of the Phase 6 transaction-cost analysis.
- **10e — Minimal monitoring.** 🟢 done 2026-07-15, ahead of schedule. Originally
  scoped as "no dashboard needed at this scale — a daily CSV log read by a short
  weekly notebook is enough," but a data QA/monitoring dashboard was built anyway once
  the 5-job pipeline made silent failures a real, unwatched risk (`dashboard/`,
  `streamlit run dashboard/app.py`) — 5 pages (pipeline health, term structure, OHLCV
  coverage, volatility, macro), reading exclusively from the `CTA_DashboardSummary`
  job's pre-computed artifacts, never computing anything at render time. Explicitly a
  QA tool, not a signal-analysis tool (`CLAUDE.md`) — the daily CSV log / weekly
  notebook idea for orders/fills/PnL once paper trading exists (10c/10d) is still the
  right shape for *that* problem and isn't superseded by this.

---

## Phase 11 — Relative Value spreads & Seasonality 🟡 11b/11c built 2026-07-31, 11d not yet built

Two new candidate signal families, discussed and scoped in detail 2026-07-31, in parallel
with IBKR paper-trading setup. Neither has any code yet; this section is the design
record a fresh session should build from. IBKR futures trading permission is in a
30-day cooldown as of this date, which is the reason this became the priority while
paper trading is blocked.

### 11a. Three seasonality papers read — synthesis

Read directly (not from memory), in this order, all in `references/`:

1. **Blitz, van der Grient, Honarvar (2023), "Reversing the Trend of Short-Term
   Reversal"** (`Reversing the trend of short term reversal.pdf`) — not about calendar
   seasonality, but the session that led here started from this paper's short-term-
   reversal enhancements. Relevant carry-over: its Section 4.1 finding that a signal's
   own standalone alpha can be real but not cost-viable alone, only inside a composite —
   the same caution now applied to seasonality below.
2. **Li, Liu, Miao, Tse (2023), "Return Seasonality in Commodity Futures"**
   (`Returns Seasonality in Commodity Futures.pdf`) — 26 commodities, 1970-2022,
   replicates Keloharju et al. (2016) and Milonas (1991). **Half-month/month effects
   were real 1970-1989 (9 commodities: Corn, Kansas Wheat, Soy Oil, Soybeans, Soymeal,
   Wheat, Feeder Cattle, Lean Hogs, Silver) and have "almost completely disappeared"
   since — zero significant commodities in 2010-2022.** Their same-calendar-month
   cross-sectional strategy only worked in the 1990-1999 subperiod (t=2.99); every other
   decade, including the most recent, is insignificant. **Most directly relevant
   finding**: combining their seasonality signal with a 12-1 month momentum signal
   produced a LOWER return than momentum alone in every subperiod tested — the
   combination diluted, not enhanced, momentum. Their own explanation: seasonality and
   momentum draw on different information sources, and layering one onto the other adds
   noise rather than signal.
3. **Keloharju, Linnainmaa, Nyberg (2014), "Common Factors in Return Seasonalities"**
   (`Common Factors in Returns Seasonalities.pdf`) — the ORIGINAL paper Li et al. (2023)
   replicates. Through 2011, seasonality was huge and pervasive (13%/year in individual
   U.S. stocks), existed in commodities too (0.93%/month, t=1.93, marginally
   significant, 24 commodities 1970-2011), and was **not** explained by known risk
   factors (size, value, momentum, macro variables) even after controlling for them.
   **Two findings that matter for the plan below**: (a) within U.S. equities, momentum
   and seasonality sorts were *uncorrelated*, not conflicting (Table 1: momentum is the
   one sort where same-month and other-month strategies both earn high, similar
   returns) — a more optimistic picture than Li et al.'s combined-strategy result; (b)
   adding a monthly seasonality factor to a market+size+value+momentum opportunity set
   raised the max ex-post Sharpe from 1.04 to 1.67, a large diversification benefit, at
   the time. KLN's own subperiod table (their Table 8 Panel B) already shows the
   seasonality composite's t-value dropping to 1.75 in 2003-2011, the weakest of their
   five subperiods — an early hint of the decay Li et al. later confirm more fully.

**Net read**: seasonality in commodities was real and economically large through the
1990s, has decayed sharply since (consistent with the "financialization" explanation
both later papers point to — more capital, more information, more efficient markets),
and the one direct test of combining it with a momentum-style signal came back negative.
This sets a real, evidence-based expectation for 11c below: go in expecting a plausible
null or negative result, not a surprise if that's what's found. It does **not** mean
skip testing — Li et al.'s own combination methodology (a crude long/short meta-strategy
built by summing separate position sets, not scaling one signal's own alpha) is
different from what's proposed in 11c, and KLN's own correlation finding (momentum and
seasonality are empirically distinct, not conflicting, in equities) argues the
combination question is still open for a differently-constructed combination.

### 11b. Standalone seasonality signal (single strategy portfolio) — plan

- **Construction**: Milonas (1991)'s half-month effect — long the first half of the
  month, flat/exit the second half — could not be sourced directly (not in `references/`
  despite searching); use Li et al. (2023)'s own detailed replication of the
  construction instead (their Eq. 2-3, `RRT_it = R̄_it / σ_it`, risk-return-tradeoff
  normalized by within-half-month volatility, following their Section 2.2/4). Revisit
  if the original Milonas paper is later sourced.
- **Asset scope**: the 9 commodities Li et al. found genuine (if historically-confined)
  half-month effects in — Corn, Kansas Wheat, Soybeans, Wheat, Feeder Cattle, Lean Hogs,
  Silver are already in this project's universe; Soy Oil and Soymeal are not currently
  pulled (same gap already logged for Soybean Crush, decision #3) — either add them or
  scope this signal to the 7 already-available names first.
- **New module**: `signals/seasonality.py`, pure function, no optimizer dependency
  (Architecture section) — calendar-window logic only, zero look-ahead risk by
  construction (the current date is always known).
- **Evaluation**: own standalone Book (Single Strategy Portfolio pattern, same as Trend/
  Carry), same train/validation/test discipline (CLAUDE.md Rule 1/2). Report honestly
  even if null/negative — expected, per 11a, not a failure.

**BUILT 2026-07-31 — plan corrected after reading both source papers directly, not
just this section's own prior summary of them.** Before writing any code, Li et al.
(2023) and Keloharju-Linnainmaa-Nyberg (2014) were read directly (not re-derived from
the plan above, which turned out to itself be a misreading from the planning session
that produced this section). Real finding: **RRT (`RRT_it = R̄_it/σ_it`, Eq. 2) is
NOT a tradeable-strategy construction in either paper** — it is used exclusively as a
descriptive statistic to test the half-month effect's statistical significance (Li et
al.'s Table 5, a paired t-test), never fed into a backtested return series. Neither
paper reports a half-month strategy's Sharpe or t-stat anywhere. The strategy Li et
al. actually DO backtest with real numbers (Table 6: 0.66%-1.67% monthly, significant
only 1990-1999) is a completely different, unrelated effect from the same paper —
Keloharju et al.'s own **same-calendar-month** strategy (rank all commodities by
trailing 5-20yr historical average return in that SAME calendar month, long top-3/
short bottom-3, full universe, monthly rebalance) — not restricted to the 9-name
half-month-effect subset at all.

Per direct instruction after this was flagged and explained, **both** are built as
parallel specs, no headline pick — `src/signals/seasonality.py`:
- **half_month** — this project's OWN trading-rule interpretation of Milonas'
  documented (but never paper-backtested) finding: vol-targeted ±1 direction (`+1`
  first half of month / `-1` second half, the Lakonishok-Smidt boundary Li et al.
  restate — day 1-15 vs. 16-end), `signals.transforms.vol_targeted_sign_signal`
  (target_vol=0.40, same convention as momentum/crossover), daily rebalancing.
  Scoped to the 7-name subset (Corn, KC_Wheat, Wheat, FeederCattle, LeanHogs, Silver,
  Soybeans) as originally planned — explicitly labeled as NOT a reproduction of RRT
  or of Milonas (1991) itself (unsourceable, per the plan above), since no published
  trading-rule construction exists to reproduce.
- **same_month** — Keloharju et al.'s actual construction: trailing 5-20yr same-
  calendar-month average return (`same_month_average_return`, min 5/max 20 years,
  strictly excluding the current occurrence), rank-weighted cross-sectionally
  (`cross_sectional_rank`, sector-scoped — the same continuous-not-binary,
  within-sector departure from the paper's own discrete top-3/bottom-3 full-cross-
  section portfolio already made for carry/XSMOM/value), ADV-filtered liquid
  universe, monthly rebalancing.

**One real, non-cosmetic bug found and fixed before shipping, caught specifically
because the FIRST result looked too good relative to this signal family's own
evidence-based null/negative prior (Sharpe: same_month train/validation/test
+0.10/**+0.54**/+0.18 gross)** — exactly the situation CLAUDE.md's own discipline
says to re-verify rather than accept. Root cause: `same_month_average_return`'s
score for calendar month m (e.g. January) only needs data through the END of the
PRIOR month (December) but is naturally labeled at January's own month-end row.
Every other cross-sectional signal in this project (momentum, carry, XSMOM) wants
exactly that labeling, because `backtest.engine`'s universal month-lag convention
(form at month-end t, trade month t+1) is built for a signal whose formation and
target periods are naturally ADJACENT, different months. A same-calendar-month
effect's formation and target are the SAME month by construction, so left as
originally labeled, the standard lag would trade FEBRUARY on "how good has January
historically been" — a full calendar month of misalignment, not a subtle one. Fixed
with a `.shift(-1)` on the raw monthly score (moves the January-effect value onto
December's row) before the daily reindex/ffill, so the same universal t→t+1 lag
lands it on the month it actually describes. A dedicated regression test
(`test_same_month_signal_decembers_row_reflects_januarys_history_not_decembers`,
`tests/test_seasonality.py`) pins this exactly: two synthetic assets with opposite
December-vs-January historical performance, confirming December's row rank-orders
by January history, not December's own.

**Result after the fix, reported honestly, not tuned back toward the pre-fix
story**: same_month train/validation/test Sharpe **+0.10/-1.04/+0.18 gross,
-0.02/-1.15/+0.06 net** — turnover ~7.4x annualized (net ≈ gross). Sharply negative
in validation (spans the 2020 COVID shock), the same pattern XSMOM/carry/value all
show there; weak-positive-to-flat in train/test. Genuinely mixed/weak, consistent
with Li et al.'s own finding that the same-calendar-month effect decayed sharply
after 1990 — not a clean win, not a disaster either. half_month: train/validation/
test Sharpe **-0.19/-0.79/+0.55 gross, -0.91/-1.31/-0.08 net** — turnover ~166x
annualized (the highest of any family in this project, similar in kind to short-term
reversal's own daily-flip-driven turnover problem), deeply negative net-of-cost in
every period but one. Both null/negative-to-mixed results are the expected outcome
per 11a's synthesis, not a failure — reported as found (CLAUDE.md Rule 1/2). 12 new
tests (`tests/test_seasonality.py`, 283 passing project-wide). Dashboard page built
(`23_seasonality_performance.py`, asset/spec/gross-net selectors, half_month reads
N/A outside its 7-name scope same as XSMOM's page reads N/A for Copper), verified
exception-free via `streamlit.testing.v1.AppTest` across all 24 pages.

### 11c. Seasonality as a TSMOM modifier — plan

- **Architecture decision (already made, not to re-litigate)**: this is NOT a
  `regime_lookup` case — `src/regime/interface.py`/`Allocator._apply_regime` operate at
  the Book level (one multiplier for the whole Trend Book), but this needs per-asset
  granularity (boost Natural Gas's winter conviction without touching FX/equity-index
  TSMOM in the same Book). Build as a pure **alpha-construction-time modifier** applied
  to `signals.momentum.tsmom_signal`'s own output, upstream of `Book` entirely — same
  category as vol-scaling, not a new portfolio-level regime layer.
- **Mechanism (already decided)**: a **continuous** seasonal weight, not a binary gate.
  Direct precedent against a gate already exists in this project: `tsmom_deadband` (the
  Trend Book bake-off's hard on/off conviction filter) had the best train Sharpe of all
  7 flavors but the *worst* validation Sharpe, and higher turnover than continuously-
  resized alternatives, not lower.
- **Candidate assets and windows (fixed now, before any backtest, per Rule 1)** —
  confidence levels stated honestly, not overclaimed:

  | Asset | Driver | High-conviction window | Confidence |
  |---|---|---|---|
  | Natural Gas | Winter heating demand | Nov-Mar | High |
  | HeatingOil | Winter heating demand | Oct-Feb | High |
  | RBOB | Summer driving season + spring blend-switchover vol | Apr-Sep | High |
  | Corn | Growing-season weather risk (July pollination) | Jun-Aug | Medium-high |
  | Soybeans | Growing-season weather risk | Jun-Aug | Medium-high |
  | Wheat / KC_Wheat | Winter-wheat dormancy-through-harvest weather risk | Mar-Jun | Medium |
  | LiveCattle / FeederCattle / LeanHogs | Grilling-season demand, marketing cycles | — | Lower — needs a literature check before committing to exact dates |

  Explicitly **not** applied to FX, Rates, Equity indices, or precious/industrial
  metals — no physical seasonal demand driver to justify it there.
- **Expectation**: per 11a, a real chance this hurts rather than helps, matching Li et
  al.'s direct finding. Test once, report honestly either way — do not iterate on window
  boundaries after seeing a negative result (that would be exactly the overfitting
  Phase 7's CPCV work already found real risk in).

**BUILT 2026-07-31, per direct instruction, as an 8th Trend Book flavor** (not a
separate standalone signal) — implemented exactly as planned above, no deviation.
`signals.seasonality.seasonal_weight_multiplier` + `tsmom_seasonal_signal`: a
raised-cosine (Hann) taper, continuously differentiable, `1.0` (unchanged) at a
window's edges and everywhere outside it, rising smoothly to `1 + amplitude` at
the window's center — `amplitude = 0.5` fixed a priori (a moderate, round-number
+50% max conviction boost), not tuned from any result. Year-end-wrapping windows
(Natural Gas Nov-Mar, HeatingOil Oct-Feb) handled via circular day-of-year
distance, verified directly (Natural Gas's center lands on Jan 15-16, exactly the
window's midpoint). Sign-preserving by construction (a pure magnitude scale on
TSMOM's own already-signed position, never flips or zeroes it) — 8 new tests,
`tests/test_seasonality.py` (296 passing project-wide).

Added to `research/single_strategy_portfolios.py`'s `build_trend_flavors()` as
`tsmom_seasonal`, run through the exact same validation-selected, test-touched-
once bake-off as the other 7 flavors — on the COMPRESSED (redundancy-removed)
universe specifically, since that's the adopted Trend construction (decision
#13). **Result: tsmom_seasonal narrowly wins the bake-off on validation Sharpe
(0.863 vs. tsmom_alone's 0.859 — a statistical coin-flip, not a clear margin),
but the two are essentially indistinguishable under the final GARCH vol-targeted
Book treatment**, checked directly head-to-head:

| | Train | Validation | Test | Turnover | Max DD |
|---|---|---|---|---|---|
| tsmom_alone | 0.216 | 0.851 | 1.600 | 0.623 | -15.0% |
| tsmom_seasonal | 0.187 | 0.860 | **1.5995** | 0.630 | -15.8% |

Test Sharpe differs by 0.0005 — noise, not a real effect either direction.
Train is slightly worse for the seasonal variant, validation slightly better,
turnover/drawdown essentially unchanged. **Conclusion: this specific continuous
seasonal-conviction construction makes no material difference to Trend's own
performance, in either direction** — not the clear "hurts" result 11a's own
synthesis anticipated (Li et al.'s finding that seasonality+momentum combined
underperforms momentum alone), but also not a genuine improvement; the modifier
is small enough in aggregate effect (up to +50% for ~2-6 months/year on 7 of 31
Trend-universe assets, inside an already gross-exposure-normalized, pooled Book)
that it washes out to statistical noise at the Book level. **Not adopted** —
`TREND_FLAVOR` stays `"tsmom_alone"` in `single_strategy_portfolios.py`/
`dashboard/_single_strategy_pipeline.py` (re-verified live on page 18: test
Sharpe still exactly 1.600, unaffected) — a coin-flip bake-off margin isn't a
basis to change the live mandate. Reported as found (Rule 1/2), not a failure.

### 11d. Relative Value sleeve — plan

**Seven pairs** (decided 2026-07-31): WTI-Brent, Gold-Silver, Crack Spread (3-leg: crude
+ RBOB + HeatingOil), Corn-Wheat, Platinum-Palladium, RBOB-HeatingOil, Wheat-KC_Wheat.
`Time_Series_Models.ipynb`'s own existing rolling-window Engle-Granger foundation already
tested Corn/Wheat, Gold/Silver, Brent/WTI (Brent/WTI the strongest candidate, 45% of
windows cointegrated) — this list extends, not replaces, that prior work.

**Data**: confirmed 2026-07-31 — every leg (WTI, Brent, Gold, Silver, Corn, Wheat,
KC_Wheat, Platinum, Palladium, RBOB, HeatingOil) is already in `Data/continuous_futures.
parquet`, refreshed daily for free by the existing `yfinance` job. **No new Databento
pull needed for this list** — Databento's spread data is specifically the within-
commodity calendar-spread/butterfly data carry needs (e.g. WTI Z26 vs WTI H27), not
cross-commodity outright pairs like these.

**Construction method per pair — three groups, not one blanket choice**:
- **Ratio / fixed beta=1 in log-price space** (same commodity or market-convention
  ratio; no statistical estimation): WTI-Brent (same commodity/units, industry already
  quotes it as a dollar spread), Gold-Silver (the "gold-silver ratio" *is* the market
  convention), Platinum-Palladium (same units, both PGMs, a real quoted ratio in that
  market), Wheat-KC_Wheat (literally the same commodity, different region/protein).
- **Fixed known economic ratio, not statistical at all**: Crack spread — 3:2:1 (or a
  simplified 2:1 crude:gasoline), driven by real refinery conversion chemistry. Decide
  WTI vs. Brent as the crude leg (or build both) before implementation.
- **Estimated, time-varying beta (Kalman filter, bake off against static/rolling OLS —
  never adopt static full-sample OLS, that's the exact look-ahead bug CLAUDE.md Hard
  Rule 2 already documents)**: Corn-Wheat — different crops, no natural 1:1
  equivalence, feed-substitution economics don't imply equal bushel value.
  RBOB-HeatingOil is ambiguous (same units, but no obvious 1:1 fundamental
  equivalence) — bake off ratio vs. Kalman rather than assume either.
- Ratio trading, explained for the build: trade `log(A) - log(B)` directly as the
  mean-reverting object (implicitly a fixed beta of 1), instead of estimating a hedge
  ratio via regression — appropriate specifically when there's a structural reason the
  two legs should track 1:1 in log terms.

**Book architecture (already decided)**: **one pooled "Relative Value" Book**, not one
Book per spread. Matches how Carry and XSMOM are already built — cross-sectional signals
pooled into one Book, not one Book per asset. Reasoning: RV/stat-arb edge specifically
comes from diversifying across many small, low-correlation spread bets; giving each
spread its own top-level Allocator slot would make a single pair trade compete for
capital against a 41-asset Trend program and a 33-asset Carry program as if it were the
same scale of diversified bet, which it isn't. The "asset" unit in this Book's
`alpha_df` is the *spread*, not a tradable outright — a downstream step must translate
each spread's Book-level weight into actual leg-level (WTI/Brent/RBOB/etc.) contract
counts, and that leg exposure will double up with Trend/Carry's own existing positions
in the same names (WTI, Brent, Gold, Corn, Wheat all already carry Trend and Carry
positions) — total portfolio risk needs to net across all three sleeves at the asset
level, not just sum sleeve-level PnLs blindly, the same caveat the Multi-Strategy
page's attribution caption already flags for Trend/Carry today.

**Shared construction, per Rule 6**: one generic, parameterized signal function (legs +
hedge weights → z-scored spread signal), reused across all 7 pairs — not copy-pasted
per pair, since crack spread is a 3-leg spread and the rest are 2-leg pairs but the
z-scoring/signal-generation mechanics after the spread series is built are the same.

**Validation discipline**: bake off each pair standalone first (own train/validation/
test, same discipline as the Trend Book's own flavor bake-off) before deciding whether
pooling them into one RV Book actually adds value over any single pair alone — do not
assume pooling helps just because it's the architecturally-preferred end state.

**Open, not yet decided**: the actual z-score entry/exit rule (e.g. continuous vol-scaled
sizing, preferred per Rule 5's own "binary underperforms continuous" finding, vs. a
threshold entry/exit band) — needs deciding during implementation, not before.

---

## Integration point: port congestion signal

Not a phase with its own number — this is a cross-cutting note. If/when the Port
Congestion Market Signals project (separate repo) validates a signal for WTI, Brent, or
the 9-commodity Track B basket, the integration point is **Phase 7** (a new Book/alpha
module, combined at the Allocator level like any other signal family) — not before, and
not a special-cased merge into an existing signal. See `CLAUDE.md`'s "Integration with the
port congestion project" section for the current (deferred) status.

---

## Open decisions log

| # | Decision | Needed before | Current lean |
|---|---|---|---|
| 1 | Rebuild vs. relabel `Time_Series_Models.ipynb` | Phase 1 | **Done 2026-07-14** — rebuilt against the ADV-filtered universe with rolling-window cointegration; re-validating the legacy Corn/Wheat claim did not confirm it as a static fact (33% of windows, regime-dependent). |
| 2 | Inclusion-rule methodology for universe edits (liquidity/ADV threshold vs. formal train/held-out split) | Phase 1 | **Done 2026-07-14** — ADV floor (≥1,000 contracts/day) implemented in both `feature_engineering.ipynb` and `Time_Series_Models.ipynb` (`signal_lib.get_liquid_universe`), plus a train/held-out split (≤2019 / 2020+) in `feature_engineering.ipynb`. |
| 3 | Soybean Crush tickers (`ZM=F`, `ZL=F`) — add now or when Phase 2d actually reaches this spread | Phase 2d | Lean: add opportunistically to `get_data.ipynb` next time it's touched, cheap either way |
| 4 | Carry data source | Phase 4 | **Transform/join stage done 2026-07-20** — `databento/transform_databento.py` now run against all 42 assets (`term_structure.parquet` outrights, `_spreads`/`_butterflies`/`_condors`/`_averages`/`_packs`), not just LE. Historical data is fully in place; carry itself is still not built into a signal (deferred as a priority per `CLAUDE.md`'s signal scope — this closes the data-availability side of the decision, not the "build the signal" side). |
| 5 | Vol target level — per-Book vs. portfolio-level (Phase 5) | Phase 5 | **Per-Book leg decided 2026-07-29**: 10% annualized per Book — standard CTA-sleeve convention (normalizing each sleeve to a common vol level before combining, so risk budgets are directly comparable), matches the existing `TARGET_VOL_BOOK=0.10` default already in use, confirmed rather than left an unexamined placeholder. **Portfolio-level (combined Trend+Carry) target still open** — a separate, later decision once both Books are actually combined and their real cross-correlation is known, not answerable from the single-Book stage. `scale_min`/`scale_max` (0.1x-5.0x, the actual per-Book leverage cap) also still an inherited dimensional-sanity default, not yet a deliberate risk-limit decision — Phase 8 territory. |
| 6 | Continuous-curve construction: roll rule and adjustment method | Phase 0, blocking further Phase 2 signal backtesting | **Built 2026-07-21, adjustment method corrected same day.** `src/data/continuous_curve.py` + `databento/build_continuous_curve.py`, `Data/continuous_futures.parquet`, all 42 assets. Roll rule: volume crossover, fixed-days-before-expiry backstop, upgrade to OI crossover once OI is purchased. Adjustment: both raw/unadjusted and **ratio/proportional** back-adjusted series built — corrected from an initial additive choice once real front-continuous close was checked directly and found never non-positive for any asset (the earlier "ratio undefined across WTI's negative print" reasoning didn't hold for this series specifically; additive itself had a live bug pushing 5 assets' adjusted close through zero — see Phase 0's change log). Two real algorithmic bugs found and fixed via testing against real data (stuck-front chain gaps, bad day-1 initialization) — full detail in Phase 0. **Now consumed by a real signal**: `research/momentum.py`, Phase 2a — done, not just re-pointed. |
| 7 | Portfolio mandate (`next_steps.md` Phase 1) | `next_steps.md` Phase 1, gates all portfolio-construction code (CLAUDE.md instruction: no portfolio-construction code before this is agreed) | **Decided 2026-07-28 — Two-Book mandate.** A medium-horizon systematic macro/commodity futures portfolio combining directional trend and curve-based carry, targeting stable risk with controlled turnover. Relative-value (cointegration/spread trading) is explicitly **not** part of the mandate yet — only a rolling-window Engle-Granger foundation exists (`Time_Series_Models.ipynb`), no z-score entry/exit signal or sizing has been built (see `next_steps.md` Phase 3 pushback) — RV is deferred to a future mandate revision once its signal actually exists, not baked in now. Vol target level itself still unresolved, see decision #5 above. Reversal/XSMOM/value remain parked research candidates per `next_steps.md` Phase 3, consistent with their already-documented weak/negative results. |
| 8 | Trend-family taxonomy (`next_steps.md` Phase 3) — merge TSMOM/crossover/breakout into one ensemble, or keep separate? | `next_steps.md` Phase 3/4, before the Trend Book is built | **Decided 2026-07-28, evidence-based, not assumed from the roadmap's own suggestion.** `research/trend_correlation.py` (rolling/EWMA Pearson + DCC-GARCH, via the author's own `dcc_garch` package, `../DCC Garch Rompolis/src`, local sibling project not vendored here) on the same three representative specs `research/portfolio.py`'s pilot already used (momentum_12mo, breakout_system1, crossover_50_200). **Result: not a uniform "merge all three."** TSMOM vs. crossover(50/200) are highly correlated (full-sample 0.648, DCC-GARCH mean 0.588, rising to 0.79 during the 2022 rate-shock window — converging further under stress, not diversifying) — genuinely one trend bet, matching the roadmap's premise for this pair only. Breakout is ~uncorrelated with both (full-sample/DCC means all in the 0.02-0.12 range, in every regime tested) — structurally independent noise relative to the other two, not "a third way of measuring the same trend." Combined with breakout's already-documented weak/negative net-of-cost result and ~50-60x turnover, there's no correlation-based case for folding it into a Trend ensemble either. **Lean: Trend Book = TSMOM (primary) + crossover 50/200 (secondary, confirms but adds little new information) — breakout stays parked alongside reversal/XSMOM/value as a research-only candidate**, not merged in. |
| 9 | Volume-source discrepancy between `Data/volume.parquet` (yfinance) and `continuous_curve`'s own volume, found while re-running the ADV floor for decision #8 | Phase 2A of `next_steps.md` (operational eligibility screen) | **Diagnosed and fixed 2026-07-28 — real data-completeness defect, not a "which source is right" question.** Full mechanism and evidence trail logged in `DATA_SCHEMA.md`'s Known gaps section. Coffee/Cotton/OrangeJuice's Databento backfill is missing rows for their genuine 2023-2024 front months, leaving `continuous_curve.assign_front_contract()` stuck on a single far-dated near-zero-volume contract for 13-28 months (price still tracks the real commodity reasonably — checked directly against `Data/close.parquet`, no roll-jump artifacts — only volume/contract-selection is defective). Sugar/Cocoa unaffected. **Fix**: `src/data/universe.ICE_SOFTS_DATA_BLOCKED` (new) excludes these 3 from `get_liquid_universe()` independently of the ADV number itself, purchase-contingent (remove once the fuller ICE history is bought), one new test (`test_universe.py`, 234 total passing). Changes no already-published result — all 3 already failed the ADV floor under `continuous_curve`'s own corrupted volume in every trend-family script that's been run so far. |
| 10 | "Single Strategy Portfolios" bake-off (`next_steps.md` Phase 4/5 — the user's own name for this phase) — which Trend Book and Carry Book construction to actually run with | Phase 4/5, before Phase 6 standardization | **Built and run 2026-07-29** (`research/single_strategy_portfolios.py`, supersedes the earlier `two_book_mandate.py` first pass). Two validation-selected bake-offs, test touched once for each winner — deliberately NOT the same multiple-comparisons situation as `tune_all_books.py`'s Bonferroni/FDR-gated grid search (5-7 economically distinct constructions per Book here, not a numerical grid over 19 Books — the earlier finding doesn't transfer, no correction applied). Weekly Book rebalancing (not `research/portfolio.py`'s original monthly), reusing the already-established precedent from `value_momentum_combine.py`/`tune_book_hyperparameters.py` to get enough validation observations (~71-94 vs. ~16-21 monthly) — `TRAIN_END`/`VALIDATION_END` themselves untouched. Two new `signals/combine.py` functions (`risk_parity_combine`, `confirmation_filter_combine`) and one new `signals/transforms.py` function (`vol_targeted_sign_signal_with_deadband` — a genuine flatten-when-low-conviction construction, answering the direct question "can the Trend Book NOT always be in the market": yes, mechanically, via a per-asset trailing-quantile threshold on trend strength). 10 new tests (244 total passing). **Trend Book: `tsmom_alone` wins decisively** (validation Sharpe 0.594 vs. 0.115-0.510 for every blended flavor; test 1.324, touched once — not comparable to momentum.py's own documented 0.402 test Sharpe, different methodology: Book/optimizer machinery + weekly annualization, not the plain standalone path). None of equal-weight/fixed-tilt/IC-weighted/risk-parity/confirmation-filter beat running TSMOM alone on validation — consistent with the earlier correlation finding (decision #8) that crossover doesn't diversify TSMOM, it mostly just dilutes it. **`tsmom_deadband` (the "not constantly trading" construction) is a real, honest cautionary result**: highest TRAIN Sharpe of all 7 flavors (0.424) but by far the WORST validation Sharpe (-0.440, deeply negative) — a textbook in-sample-looks-great/out-of-sample-falls-apart pattern; a conviction filter that requires trend strength to clear its own trailing median before trading ends up gating out of the market during COVID's fast, ambiguous whipsaw, then re-entering late. Its turnover is also the HIGHEST of the 7 (0.84 vs. 0.59-0.70 for the always-in flavors) — a non-obvious but real finding: flipping a position on/off between zero and full size generates MORE turnover than continuously resizing an always-held position, undercutting the "trades less -> costs less" intuition. **Recommendation: reject `tsmom_deadband` for now** — mechanically possible, but the evidence argues against it, not for it; `tsmom_alone` (i.e. no crossover blend at all) is the selected Trend Book. **Carry Book: `carry_timing_zero` wins** (validation -0.938, best of 3 evaluable flavors — `carry1_12` excluded outright, only 18 valid weekly rebalance dates after its 12-month-smoothing warmup, below the 20-date floor; `carry1m` narrowly misses too, n=19). All evaluable carry flavors are negative on validation, consistent with carry's already-documented COVID-era underperformance. `carry_timing_zero` test Sharpe 0.374 (touched once). **Combined two-Book Allocator (equal Book-risk baseline)**: train -0.132, validation -0.499 (dragged down by Carry's weak validation despite Trend's strong one), test 0.930 (n=175). 95% VaR -3.3%, ES -5.0% (weekly). Selected: Trend = `tsmom_alone`, Carry = `carry_timing_zero`. |
| 11 | `Book`'s vol-targeting estimator (`_apply_vol_target`) — EWMA-of-realized-PnL was inherited from the retired stat-arb engine, never tested against alternatives | Before the hyperparameter-tuning grid (decision path following #10) | **Tested and decided 2026-07-29.** Direct follow-up question to the asset-level 3-way vol-estimator comparison (decision context: `research/vol_estimator_comparison.py`, where EWMA came in last) — structurally a DIFFERENT forecasting target (a Book's own aggregate realized-PnL vol, not an asset's price vol), so that result doesn't mechanically transfer, but GJR-GARCH doesn't care what 1D return series it's fed, so there's no structural reason not to test it here too. `research/book_vol_targeting_estimator.py` (new, cached to `Data/research/book_vol_targeting_{garch,series}.parquet` + `_comparison.csv`, same never-recomputed-live convention as the asset-level GARCH cache): EWMA vs. GJR-GARCH on the two selected Books' (`tsmom_alone`, `carry_timing_zero`) own DAILY-marked PnL (`portfolio.book.daily_mark_pnl`), QLIKE/MSE against forward-realized variance, TRAIN only. First pass returned all-NaN for GARCH — a real bug, not a null result: `daily_mark_pnl`'s NaN-row `.sum(skipna=True)` silently returns 0.0 (not NaN) for the ~900 days before each Book's first real weight, and GARCH's own zero-variance guard correctly rejected that degenerate warmup outright (EWMA would NOT have caught this — it would have just silently understated vol for years, a worse failure mode). Fixed by trimming each Book's PnL to its own first real rebalance date before comparing. **Result after the fix: GJR-GARCH wins decisively, not marginally** — QLIKE cut ~60-70% vs. EWMA at every horizon (21d/63d), both Books (Trend: 0.689→0.207, 0.679→0.147; Carry: 0.615→0.213, 0.404→0.149). **Decision: `Book.vol_estimator="garch"` adopted for the Single Strategy Portfolios.** Implemented as an opt-in constructor param on `Book` (`src/portfolio/book.py`), default `"ewma"` unchanged — every other already-published Book result in this project (the 6-Book pilot, all 19-20 tuned Books, value/XSMOM mix-vs-integrate, etc.) is untouched, not silently altered. GARCH path: refit every `garch_refit_freq` periods (default 20) using ONLY realized PnL through the previous period, `filter_gjr_garch` forward with fixed params between refits (`data.garch_volatility`'s own validated primitives reused directly, not reimplemented — O(T²) over the full walk since `filter_gjr_garch` re-runs its cumulative recursion each call, a deliberate correctness-over-speed tradeoff, consistent with this project's existing "GARCH is slow by construction, offline use only" convention). Falls back to the EWMA recursion during GARCH's own warmup (`garch_min_warmup=104` periods, ~2 years weekly) and on any per-period fit/filter failure. **One disclosed caveat, not smoothed over**: the validated comparison used DAILY-marked PnL; `Book`'s actual internal vol-targeting still operates at each Book's native (weekly) rebalance cadence — the estimator-class advantage is assumed, not separately re-proven, to carry over to that granularity (documented in both the `vol_estimator` docstring and the dashboard page). 2 new tests (`test_book.py`, 246 total passing). **Re-run with GARCH**: Trend train 0.295→0.321, test 1.324→1.356 (slightly better); Carry train -0.344→-0.368, test 0.374→0.293 (slightly worse); combined test 0.930→0.869. Validation Sharpe essentially unchanged for both Books (0.594, -0.938) — a modest, plausible-magnitude shift either direction, not a dramatic overnight flip, consistent with GARCH and EWMA both being reasonable vol-targeting mechanisms on average (the QLIKE improvement is about forecast RESPONSIVENESS to regime shifts, not a large average-leverage change). Dashboard: new "Book-Level Vol-Targeting" section on `06_volatility_estimators.py`, reads the cached comparison table + vol-forecast-over-time chart, verified exception-free via `streamlit.testing.v1.AppTest` (18/18 pages). Only GARCH's own parameter (`garch_refit_freq`) belongs in the later hyperparameter grid — `ewma_halflife` is moot for these two Books now. |
| 12 | How to combine Trend and Carry sleeves in the Multi-Strategy Portfolio — `Allocator.run()` just sums each Book's PnL (`.add(fill_value=0.0)`), no risk-weighting at all | Multi-Strategy Portfolio dashboard page, before wiring any weighting scheme into shared/live code | **Built, validated on real data, and wired into `Allocator`/the dashboard, all 2026-07-29-30.** Flagged after the naive Allocator sum visibly let Carry (weak/negative) drag down Trend (strong): equal-PnL-sum has no notion of risk contribution at all, unlike real CTA multi-strategy practice (combine sleeves by RISK, not return, specifically to avoid overfitting relative sleeve weights on ~140-250 weekly observations — far too little history for a return-based optimization to be trustworthy). Discussed and decided to build a general n-sleeve equal-risk-contribution (ERC) solver, not the 2-sleeve closed form (`w_1/w_2 = sigma_2/sigma_1`, correlation-independent) — more sleeves (Relative Value, etc.) are planned, and the n=2 closed form doesn't simplify the n>=3 machinery needed later. **`src/portfolio/risk_parity.py`** (new): `risk_contributions(weights, cov)` (exact Euler decomposition, sums to portfolio vol by construction) and `risk_parity_weights(cov, risk_budgets=None)` (Spinu 2013 / Bruder-Roncalli convex log-barrier reformulation — `minimize 0.5*w'Sigma*w - sum(budget_i*log(w_i))`, L-BFGS-B — not the naive non-convex "minimize squared RC-difference" objective, which can converge to different local optima depending on start point). One real fix during verification: L-BFGS-B's default `ftol`/`gtol` left risk contributions off by ~6e-5 relative for a 3-sleeve synthetic check — cheap to tighten (`ftol=1e-15, gtol=1e-12`) for a problem this small and convex, brought the mismatch down to ~1e-12. Verified numerically before any test was written: 2-sleeve solution matches the closed-form inverse-vol ratio across rho in [-0.5, 0.8]; n=3 gives exactly equal risk contributions off a non-diagonal Sigma; unequal budgets ([0.6, 0.3, 0.1]) reproduce those exact risk-contribution shares. 8 new tests, `tests/test_risk_parity.py`. **`src/portfolio/sleeve_covariance.py`** (new): takes a (T x N) sleeve-PnL DataFrame, returns ONE current covariance matrix (not a full rebalance-dated history like `portfolio.covariance.build_cov_dict` — the risk-parity solver only ever needs today's Sigma). Two estimators: `rolling_covariance`/`ewma_covariance` (plain pandas `.cov()`/`.ewm().cov()`, the cheap baseline) and `dcc_garch_covariance` (Engle 2002 DCC-GARCH, reusing `research/trend_correlation.py`'s exact two-stage fit call — `dcc_garch.garch.gjr_garch.fit_multivariate_gjr` then `dcc_garch.dcc.optimizer.fit`, the already-validated local sibling package). One unit gotcha caught and fixed before it could silently corrupt every DCC-derived weight: `dcc_garch`'s own GJR-GARCH scales returns x100 internally (documented in that package's own `gjr_garch.py`), so its `H` output (the conditional covariance path) comes back in (%)^2 units, not decimal-return units — divided by 100^2 before returning, confirmed by checking the DCC covariance landed within the same order of magnitude as the plain sample covariance on synthetic correlated data (a dedicated regression test for exactly this, `test_dcc_garch_covariance_same_order_of_magnitude_as_simple_sample_covariance`). The `dcc_garch` import is deferred to inside the function (not module load time), same discipline as `data.garch_volatility`'s own lazy import, since it's a private local sibling repo the public dashboard environment won't have — the two pandas estimators keep working regardless. 10 new tests, `tests/test_sleeve_covariance.py` (3 DCC-specific tests skip cleanly, not fail, if the sibling repo isn't present). **`research/sleeve_risk_parity.py`** (new): reuses `single_strategy_portfolios.py`'s exact winning constructions (both Books GARCH vol-targeted, decisions #10/#11) called directly outside Streamlit — same pattern `research/book_vol_targeting_estimator.py` already established. Risk-parity weights fit ONCE on TRAIN sleeve PnL only (CLAUDE.md Rule 1/2 — the weight choice itself must not see validation/test before being fixed), then applied as a FIXED static weight across all three periods; walk-forward refitting is flagged as a natural next step, not built here. **Result, real numbers, not fabricated**: all three covariance estimators agree closely on the split (rolling 58.7/41.3, EWMA-halflife-87 57.5/42.5, DCC-GARCH 59.3/40.7, Trend/Carry) — reassuring convergence given the small-sample DCC concern raised at the outset. Risk parity improves combined Sharpe vs. naive in every period, by every estimator: train -0.136 (naive) → -0.05 to -0.07; validation -0.512 → -0.32 to -0.36; test 0.869 → 0.98 to 1.00. Combined annualized vol also drops slightly in every period (e.g. test 0.167 → 0.157), consistent with a genuine diversification-weighting effect, not just a return-chasing tilt (the fitted weight, ~59% Trend / ~41% Carry, was decided from TRAIN alone, before validation/test's stronger Trend performance could have leaked into the choice). **Wired into `Allocator` and the Multi-Strategy dashboard page the same day, per direct instruction once the real-data validation above landed.** `Allocator.__init__` gained an optional `book_weights: dict[str, float]` param — each active book's own "pnl" (and "asset_contributions", if present) is scaled by its weight before the existing `.add(fill_value=0.0)` combination; a book not named in `book_weights` defaults to 1.0, so `book_weights=None` (the default) is byte-for-byte identical to every existing caller's prior behavior (`Allocator`'s own architecture doc now notes this is a STATIC weight decided by the caller — no covariance-estimation logic added to the Allocator itself, same boundary `regime_lookup` already established). 5 new tests, `tests/test_allocator.py` (weighted-combination, missing-name-defaults-to-1.0, `book_weights=None` parity, weighted `asset_contributions`, and the Book-doesn't-provide-attribution case) — 271 tests passing project-wide. `dashboard/_single_strategy_pipeline.py` gained `compute_risk_parity_weights(estimator)` (`st.cache_data`, not `st.cache_resource` — returns plain floats, so the estimator toggle doesn't re-run the expensive Book construction `load_and_run` caches), reusing the exact same TRAIN-only-fit, rescaled-by-n_sleeves=2 construction `research/sleeve_risk_parity.py` already validated. `dashboard/pages/20_multi_strategy_portfolio.py` gained a "Combination Method" radio (Naive / Risk Parity) plus a covariance-estimator sub-radio (Rolling / EWMA / DCC-GARCH) when Risk Parity is selected — the combined equity curve, per-Book attribution, VaR/ES, and key-takeaway captions all recompute against whichever method is selected, not just a headline number (verified live: selecting DCC-GARCH reproduces the exact 59.3%/40.7% split `sleeve_risk_parity.py` found). Verified exception-free via `streamlit.testing.v1.AppTest` across all 22 pages AND across all three combination-method/estimator branches on page 20 specifically (naive, risk-parity-rolling, risk-parity-DCC-GARCH). |
| 13 | `next_steps.md` Phase 2 universe compression (layers B/C — economic redundancy, economic coverage) — never actually applied to any Book, checked directly against `research/single_strategy_portfolios.py`'s own code before starting | Raised directly by the user after the Phase 11b handoff; applied to same_month first, then Trend/Carry, per direct instruction | **Built and run 2026-07-31.** `research/universe_compression.py` (new): train-period (≤`TRAIN_END`) pairwise return correlation within each `data.sectors.SECTORS` cluster, for the ADV-filtered 36-asset universe — a performance-blind, structural analysis (CLAUDE.md Rule 1's own concern, one level up: a portfolio-level universe edit must not be justified by having seen any Book's backtest result). **Rule applied** (judgment-informed, not a mechanical cutoff): ≥0.95 train correlation ("near-duplicate") → drop the less-liquid twin (by ADV, not Sharpe) for EVERY family; 0.85-0.95 ("redundant for a directional bet only") → drop for **trend** (a per-asset time-series signal — two nearly-identical directional bets multiply exposure to one factor) but KEEP for **rank** (same_month, carry — cross-sectional rank signals, where a highly-correlated member still contributes its own seasonal/carry estimate; 11c's own seasonal-window table already treats WTI/Brent/RBOB/HeatingOil as economically distinct despite high price correlation); <0.85 → no action. **Real numbers**: US_30Y/UltraBond 0.98, SP500/Dow 0.97, US_5Y/US_10Y 0.965 (all near-duplicate — the third of these was missed in the first manual pass and only surfaced by running the actual script and applying the stated rule consistently, not selectively); Wheat/KC_Wheat 0.93, WTI/Brent 0.92 (directional-only); Gold/Silver 0.80, LiveCattle/FeederCattle 0.81 (below the bar, no action). ADV tie-breaks: WTI ~11x Brent's, Wheat ~2.4x KC_Wheat's, US_10Y ~1.4x US_5Y's. **`data.universe.compress_for_family(included, family)`** (new, `CLUSTER_REDUNDANT_ALL = ["Dow", "UltraBond", "US_5Y"]`, `CLUSTER_REDUNDANT_TREND_ONLY = ["Brent", "KC_Wheat"]`) — same documented-constant pattern as `ICE_SOFTS_DATA_BLOCKED`. Coverage (C) check passes for both families (every sector retains ≥2 members: Energy 4, EquityIndex 3, Rates 3, Grains 3). **Separately discovered, unrelated finding, not fixed here**: Sugar and Cocoa have zero valid return data before `TRAIN_END` — Softs contributes nothing to any train-period redundancy read, a genuine data-coverage gap distinct from the redundancy question. 15 new tests (`tests/test_universe.py`, 288 passing project-wide). **Applied to same_month** (`research/seasonality.py`'s `load_and_prepare_data(family=None)` — `None` preserves prior behavior exactly, `family="rank"` applies the compression; half_month dropped from further work per the same session's decision — deeply negative net-of-cost, not a paper-validated construction to begin with, code/tests/dashboard page kept as historical record only) — new `research/seasonality_single_strategy.py` reuses `single_strategy_portfolios.py`'s own `build_book()` directly (weekly cadence, GARCH vol-targeting, no bake-off needed — same_month has one construction, like XSMOM/Value). **Result: train 0.123, validation -1.269, test -0.353, turnover 1.26, max DD -56.8%** — weaker than the standalone plain-Sharpe read (train +0.10/validation -1.04/test +0.18 gross), looking more like Value/XSMOM's already-parked profile than Trend/Carry's, consistent with the expectation set going in. **Applied to Trend/Carry** — `single_strategy_portfolios.py` gained a `run_pipeline()` extraction (Rule 6) so the existing UNCOMPRESSED path (byte-for-byte reproducing decision #10/#11/#12's published numbers) and a new COMPRESSED path run side by side in one script, no silent replacement; `load_and_prepare_data()` itself is untouched (zero risk to `dashboard/_single_strategy_pipeline.py`, confirmed via `streamlit.testing.v1.AppTest` across all 24 pages — pages 18-20 unaffected). **Trend: a clean, genuine improvement** — test Sharpe 1.356 → 1.600, turnover 0.78x → 0.62x (both better, with LOWER turnover — consistent with removing genuinely redundant, correlated directional bets, not a fluke). **Carry: a real methodological complication, not a clean result** — compression changed `carry1m`'s valid-rebalance-date count from 19 (below the bake-off's own 20-date floor, excluded in the uncompressed run) to 21 (barely included), flipping the bake-off's winner from `carry_timing_zero` to `carry1m` on a knife-edge validation Sharpe of +0.039 (n=21) — and `carry1m`'s test Sharpe is a wild **-2.035 on just 26 observations**. This is a selection-artifact red flag (a marginal data-sufficiency threshold flipping the winner, not genuine outperformance) — reported as found, NOT adopted, and NOT to be read as "compression hurt Carry." Combined Allocator test Sharpe improved (0.869 → 1.077) but is entangled with Carry's fragile pick and shouldn't be read at face value either. **Decided, same day, per direct instruction after discussion**: Trend's compressed universe ADOPTED into the live Two-Book mandate — `dashboard/_single_strategy_pipeline.py`'s `load_and_run()` now builds Trend flavors on `compress_for_family(included, "trend")`, verified live on page 18 (test Sharpe 1.600, matching the research script exactly). Carry REVERTED to the uncompressed, originally-published construction (`carry_timing_zero`) — explicitly NOT mechanically re-picking whichever flavor the compressed bake-off happened to select, since that flavor (`carry1m`) was a knife-edge, unstable pick (a marginal data-sufficiency threshold effect, not genuine outperformance), not a real improvement to adopt. `research/single_strategy_portfolios.py` gained a `run_pipeline()` return of the constructed Book objects (not just summary stats) specifically so `main()` could build this exact ADOPTED mixed combination (Trend from the compressed run, Carry from the uncompressed run) and report it explicitly, rather than only ever showing the two "pure" (all-compressed / all-uncompressed) variants. **Adopted combined Allocator result: train -0.112, validation -0.435, test 0.994** (n=175, weekly) — better than the original uncompressed combined (test 0.869) and more honest than the fully-compressed number (1.077, which was inflated by Carry's unstable pick). 95% VaR -0.029, ES -0.043 (weekly). Verified via `pytest tests/` (288 passing) and `streamlit.testing.v1.AppTest` across all 24 pages, plus a direct spot-check confirming page 18 renders the new adopted Trend numbers live.

**Same-day follow-up, per direct instruction: a hypothesis-driven (not performance-driven) universe restriction for same_month.** The user's own framing, addressed directly: this is NOT the CLAUDE.md Rule 1 look-ahead pattern (never edit the universe after observing backtest performance) — the restriction is fixed from a documented PHYSICAL/ECONOMIC seasonal-demand theory (Phase 11c's own conviction table, built and logged before any of this session's backtests) BEFORE looking at same_month's performance on these specific names, the same discipline already used for `SEASONALITY_HALF_MONTH_ASSETS` and 11c's own asset scope. `signals.seasonality.SEASONALITY_ECONOMIC_DRIVER_ASSETS` (new) = Natural Gas, HeatingOil, RBOB, Corn, Soybeans, Wheat, KC_Wheat — 11c's table restricted to Medium confidence or higher (LiveCattle/FeederCattle/LeanHogs excluded — 11c's own table already flags their windows as needing a literature check never completed). `research/seasonality_economic_universe.py` (new): plain-Sharpe comparison (matching how same_month's full-universe result was first read, before any Book treatment), same_month on the full rank-compressed universe (32 assets) vs. this 7-name economic-driver subset. **Result: genuinely mixed, not a clean win, but a real reshaping of the risk profile** — train got WORSE (0.050→-0.202 gross), but validation flipped from deeply negative to near-flat (-1.015→-0.088 gross — the 2020 COVID window that crushes nearly every cross-sectional family in this project), and test improved, especially net-of-cost (0.036→0.137 net — nearly 4x). Turnover nearly identical (7.38x vs 7.56x), so this isn't a turnover-cost artifact. **Escalated to the weekly/GARCH Single Strategy Portfolio treatment same day, per direct instruction.** `research/seasonality_single_strategy.py` extended to run both universes side by side (FULL rank-compressed vs. ECONOMIC-DRIVER, no headline pick), reusing `build_book()` unchanged. **Result: a much more pronounced version of the plain-Sharpe finding, not just a confirmation of the same small effect** — train **0.123→-0.162** (worse, consistent with the plain-Sharpe read), validation **-1.269→-0.389** (still negative but far less catastrophic), test **-0.353→+0.454** (a genuine sign flip), turnover **1.259→0.158** (~8x lower — only 7 correlated names in 2 sectors, weekly-marked on a slow monthly-cadence signal), max DD **-56.8%→-31.7%** (nearly halved). Every single metric improved except train. This is a genuinely promising profile, not clearly "park it" like Value/XSMOM — train and validation are still negative, so it isn't a clean all-periods win, but the magnitude and consistency of improvement across Sharpe/turnover/drawdown when restricting to the physically-motivated 7-name universe is a real signal that the hypothesis-driven restriction is doing real work, not noise. **Same-day follow-up, per direct instruction: three multi-strategy combinations, before building any new dashboard page.** `single_strategy_portfolios.py` gained `build_adopted_books()` (returns the already-decided Trend/Carry Book objects directly, without re-running the 7/4-flavor bake-off — fast, reusable; `dashboard/_single_strategy_pipeline.py`'s own `load_and_run()` refactored to call this instead of duplicating the construction inline, CLAUDE.md Rule 6 — re-verified byte-identical live on page 18, test Sharpe still 1.600, all 24 pages + 288 tests still clean). `seasonality_single_strategy.py` gained `build_economic_seasonality_book()` (same pattern, for the economic-driver same_month Book). New `research/multi_strategy_seasonality.py`: three naive equal-Book-risk Allocator combinations (no risk-parity weighting - decision #12's own follow-up, not re-applied here), no headline pick:

- **A: Trend + Carry + Seasonality** — train -0.154, validation -0.442, test 0.982, VaR95 -0.031
- **B: Trend + Seasonality** — train **+0.095**, validation **+0.210**, test **1.131**, VaR95 **-0.020**
- **C: Trend + Carry** (the current mandate, reproduced here for a clean side-by-side) — train -0.112, validation -0.435, test 0.994, VaR95 -0.029

**A striking, clean result: B (Trend + Seasonality, WITHOUT Carry) is the ONLY combination positive in all three periods** — and it beats both A and C on every single metric (train, validation, test Sharpe, and VaR95). Carry's own already-documented weak/negative standalone profile (train -0.368, validation -0.938) appears to be actively dragging down both combinations that include it (A and C), while the economic-driver Seasonality Book pairs constructively with Trend specifically in the two periods (train, validation) where Carry hurts most.

**Real, more important finding caught immediately by direct question, correcting an incomplete first read: none of A, B, or C actually beats standalone Trend alone (test Sharpe 1.600) in ANY period.** The initial framing above (comparing A/B/C only to each other) understated this — the naive equal-Book-risk Allocator forces equal risk regardless of each Book's own quality, so blending Trend (strong) with a materially weaker Book (Carry test 0.293, Seasonality test 0.454) at a forced 50/50 or 33/33/33 split dilutes Trend's own edge, exactly the same failure mode already documented for Trend+Carry alone in decision #12 (why risk-parity weighting was built in the first place). Applying that SAME already-validated tool (`research/multi_strategy_seasonality_risk_parity.py`, generalizing `sleeve_risk_parity.py`'s pattern to n=3) to all three combinations: risk-parity improves every combination over its own naive version (consistent with decision #12's own finding), but **still, in every weighting scheme, standalone Trend beats every combination in every period**:

| Combo | Weighting | Train | Validation | Test |
|---|---|---|---|---|
| Trend alone | — | **0.216** | **0.851** | **1.600** |
| A: T+C+S | naive / risk-parity (ewma) | -0.154 / -0.102 | -0.442 / -0.306 | 0.982 / 1.058 |
| B: T+S | naive / risk-parity (ewma) | 0.095 / 0.071 | 0.210 / 0.121 | 1.131 / 1.064 |
| C: T+C | naive / risk-parity (ewma) | -0.112 / -0.043 | -0.435 / -0.217 | 0.994 / 1.184 |

Interesting reversal under risk-parity specifically: C (Trend+Carry) becomes the best-performing COMBINATION in test (1.184-1.217 depending on estimator), overtaking B — but even C's best risk-parity test Sharpe (1.217) still falls well short of Trend alone (1.600). One numerical gotcha hit and fixed while building this: `risk_parity_weights`'s log-barrier solver failed to converge (`ABNORMAL` termination) on this 3-sleeve case — its tight `ftol`/`gtol` (tuned for Trend/Carry's own 2-sleeve covariance scale) couldn't resolve the gradient balance against these smaller weekly-PnL-variance values (~1e-4 to 1e-5). Fixed locally (not by loosening the shared, already-tested function) by scaling the covariance matrix by 1e4 before solving — the solution is provably scale-invariant after normalization (Sigma -> c*Sigma solves for w/sqrt(c), which normalizes identically), confirmed directly, not assumed.

**Conclusion, stated plainly: on this evidence, neither Carry nor Seasonality (individually or combined, naive or risk-parity weighted) has been shown to add value over simply running Trend alone.** This doesn't necessarily mean either sleeve is worthless in every respect (real CTA practice sometimes keeps weaker, low-correlated sleeves for regime/tail-diversification reasons a single train/validation/test Sharpe split doesn't fully capture), but on the metric actually tested here, Trend alone wins outright. **Not yet decided**: whether to pursue that regime-diversification angle further, run a CPCV/PBO-style robustness check (per the same discipline already applied to Trend/Carry's own hyperparameters in Phase 7) before concluding anything more strongly, or simply accept Trend-alone as the current best evidence.

**Dashboard integration, same day, per direct instruction — lean scope, chosen over a full dedicated page per finding.** New `dashboard/pages/24_seasonality_book_performance.py` (registered in the "Single Strategy Portfolios" nav group alongside Trend/Carry Book) — the economic-driver same_month Book, same tearsheet/equity-curve/attribution shape as pages 18/19, reported honestly as weaker than standalone Trend, same discipline as Value/XSMOM/Carry's own weaker pages. The 3-combination comparison was NOT given its own page — instead added as a new, additive "Does Adding Seasonality Help?" section at the bottom of the EXISTING page 20 (Multi-Strategy Portfolio), since it's the same "combine sleeves" question that page already answers for Trend+Carry: a sleeve multiselect (Carry/Seasonality, Trend always included) + Naive/Risk-Parity toggle, live-computed, headline finding stated up front in the caption. `dashboard/_single_strategy_pipeline.py` gained `load_and_run_seasonality()` (a SEPARATE `st.cache_resource` entry from Trend/Carry's own `load_and_run()` — kept in this same shared module, not page-local, for the exact reason that module's own docstring already documents: two pages independently caching the same expensive pipeline caused a real Streamlit Cloud OOM crash) and `compute_risk_parity_weights_n()` (a general n-sleeve version of the existing 2-sleeve `compute_risk_parity_weights`, applying the same covariance-rescale-by-1e4 numerical fix found while building `research/multi_strategy_seasonality_risk_parity.py`). tsmom_seasonal got no dashboard treatment at all — a statistical tie, not adopted, stays documented in WORKFLOW.md/CLAUDE.md only. Verified exception-free via `streamlit.testing.v1.AppTest` across all 25 pages, including explicit interaction tests on page 20's new widgets (Seasonality added, Risk Parity selected, zero-extra-sleeves edge case, and the pre-existing Trend+Carry toggle re-verified unaffected) — 296 tests passing project-wide. |

---

## Change log

- Initial version: distilled from `project-review.html` (2026-07-08 snapshot) into a
  living roadmap, source-verified against `get_data.ipynb`, `volatility.ipynb`,
  `feature_engineering.ipynb`, and `Time_Series_Models.ipynb`. Added explicit signal
  mechanics for breakout/crossover, the RV spread candidate table, the carry-unblock
  table, and the port congestion integration point.
- Update: added `jobs/update_data.py`, a thin standalone re-run of the `get_data.ipynb`
  pull, built ahead of schedule relative to Phase 10 on the grounds that it's cheap,
  decoupled from having live signals ready, and keeps the parquet files from falling
  behind while research continues. Wired into Windows Task Scheduler (`CTA_DailyDataUpdate`,
  daily 6PM) — flagged in Phase 10b as a 🟡 partial rather than folded silently into
  Phase 0.
- Update: Phase 4 (Carry) revised from 🔴 blocked to 🟡 unblocked. Live-tested `yfinance`
  individual futures contract-month tickers (undocumented behavior) across all 38 assets
  — 38/38 resolved to real data, and multiple simultaneous contract months for WTI showed
  a genuine term structure. Corrected the earlier assumption that Interactive Brokers'
  historical data is free (it isn't — funded account + $30/month commission minimum
  required). Added confirmed CME DataMine pricing ($105–$2,100/month) and flagged that
  EIA's free futures-price series was discontinued in April 2024. Also discovered and
  fixed a real bug during this testing: the environment's `yfinance` was pinned at 0.2.50
  and silently broken by a Yahoo API change (even `CL=F` failed); upgraded to 1.5.1,
  which fixed it — this affects `update_data.py` and the scheduled task directly.
- Update (2026-07-14): re-verified the Phase 4 carry claim live rather than trusting the
  prior write-up, and found the two previously-open caveats resolve unfavorably for
  backtesting — a contract's ticker stops resolving entirely once it expires (tested on
  3 recently-expired WTI contracts, all 404), and the multi-year pre-2024 history on a
  live contract is a synthetic/theoretical print (`Volume=0`, flat OHLC), not real data.
  Net effect: the free path only supports forward-capture, not historical backtesting.
  Built and live-verified `jobs/capture_term_structure.py` (38/38 assets, idempotent
  re-runs) and wired it into Task Scheduler as `CTA_TermStructureCapture` (daily 6:15PM).
  Searched for a free historical-backfill alternative: TurtleTrader's free dataset is
  correctly-shaped (real per-contract-month OHLC+OI) but frozen at 1999-12-20, ruling it
  out; identified Databento's $125 new-account credit as very likely sufficient to cover
  a one-time historical batch pull for the full universe given how small daily OHLCV bars
  are relative to their $/GB pricing — account created, backfill script is the next step.
  Also resolved Phase 1 open decisions #1 (rebuild `Time_Series_Models.ipynb`) and #2
  (universe inclusion rule = ADV floor + train/held-out split, not a liquidity story for
  the specific Energy drop already in `feature_engineering.ipynb` — checked 2020+ ADV and
  Natural Gas/Brent are mid-pack, not outliers, so a real liquidity rule doesn't back that
  edit).
- Update (2026-07-14): investigated whether Yahoo's continuous futures series
  (`Data/close.parquet` etc.) is back-adjusted or a raw front-month splice, prompted by a
  user question about what the pre-term-structure price history actually represented.
  Confirmed it's the raw front-month contract, auto-rolled by Yahoo. An initial claim that
  this causes confirmed, material roll-jump contamination in every return calculation
  turned out to be an overreach from one coincidental example (a big WTI move landing near
  a predicted contract expiry) — a broader periodicity/volume test across 6 assets found
  no strong evidence of a systematic artifact. Logged as an unresolved, low-confidence
  caveat in `DATA_SCHEMA.md` section 1 rather than a confirmed defect; real resolution
  needs the same Databento historical contract data queued for Phase 4. Also extended the
  scheduled-job infrastructure built earlier today: added `CTA_VolatilityUpdate` (Yang-
  Zhang, 6:05PM) and `CTA_MacroDataUpdate` (yield curve/Fed funds/GSCPI/trade policy
  uncertainty, 6:20PM) — see Phase 10b and `DATA_SCHEMA.md` section 3 for source detail.
  None of the 4 macro sources had a documented origin before this; identified via each
  file's schema (FRED for yield curve, NY Fed Markets API for reference rates, and fixed
  download URLs for GSCPI and the Caldara et al. Trade Policy Uncertainty index).
- Update (2026-07-14): read `CTA Feature Engineering Primer.pdf` per direct request, to
  assess what other datasets might be worth having now that data infrastructure is
  actively being built. Per `CLAUDE.md`, treated it as a menu to filter, not a to-do list
  — most of its content (VIF multicollinearity filtering, harmonic seasonality, AIC
  stepwise selection, automated lookahead-bias CI tests) is Phase 2+/7 methodology, not a
  data gap. The one concrete new-dataset item: **open interest**, explicit in the
  primer's own recommended schema and something we currently lack entirely — added to
  the Databento historical-pull scope above. Separately, investigated whether the 4 macro
  sources are safe for point-in-time backtest use (ragged edge / restatement risk) —
  found GSCPI has a real ~1-2 week publication lag (confirmed empirically: the June-2026
  value wasn't available until ~July 10) and Fed funds reference rates have a confirmed
  1-business-day lag; yield curve is clean, TPU is near-real-time. Built
  `src/macro_point_in_time.py` with a point-in-time accessor per source, live-verified
  against the actual data, so this is solved before any signal consumes it rather than
  discovered as a look-ahead bug later — see `DATA_SCHEMA.md` section 3 for full detail.
- Update (2026-07-15): Databento credit reinstated (see Phase 4). Executed the
  historical backfill: a first attempt (single-process `get_range()` script) appeared to
  hang and was killed, which turned out to have discarded $4.20 of real, already-
  downloaded data — the process was actually working, just silent due to buffered
  output and an all-at-the-end write pattern. Root cause isolated: `get_range()` hangs
  specifically on `schema="definition"` even for small requests; Databento's own
  `get_range()` docstring recommends batch download for requests this size, which we
  hadn't been using. Switched to the batch API (`submit_job`/`download`, fully
  server-side, no long-lived connection to hang): 76 jobs submitted (44 succeeded, 32
  need retry after rate-limiting), now processing asynchronously. Transform/join stage
  (merging downloaded files into `term_structure.parquet`) not yet built. Full detail in
  Phase 4.
- Update (2026-07-15): built a data QA dashboard (Phase 10e, done ahead of schedule —
  see that section) and, in the same session, reorganized `Scripts/` into
  `src/jobs/databento/notebooks/dashboard` (all paths in this file updated
  accordingly; full rationale in `cleanup.md`). Also built portfolio-construction
  scaffolding (`src/portfolio/`, `src/signals/combine.py`, `src/regime/interface.py`)
  ahead of Phase 5/7's normal trigger, adapted from the retired Cross Asset Stat Arb
  Engine and evaluated piece-by-piece — see Phase 5, Phase 7, Phase 8, and `cleanup.md`
  section 3 for what transferred, what was adapted, and what was discarded. Added
  `Data/volatility_manifest.csv` so `CTA_VolatilityUpdate` has a real logged-run record
  like the other 3 jobs (previously inferred from file modification time — a real gap
  the new dashboard surfaced). Initialized git for the first time in this project's
  history (no prior version control), as a safety checkpoint before the reorg and the
  first step toward the GitHub upload planned for this repo.
- Update (2026-07-15): explored the first completed Databento asset download (LE /
  Live Cattle, both schemas) ahead of building the transform/join stage. Found and
  fixed a real design gap before it could corrupt anything: the raw `symbol` field is
  decade-ambiguous (single-digit CME year code — `LEZ5` collided between Dec-2015 and
  Dec-2025 across the 16-year pull), fixed by deriving the true year from the
  `definition` file's `maturity_year` field instead. Also found that parent-symbol
  queries return far more than outrights (66% calendar spreads, 13% butterflies, 21%
  outrights, this asset) — originally planned to filter these out, reconsidered:
  calendar spreads are a directly usable, arguably better carry source than
  back-differencing two outright legs, so the transform-stage design now keeps all
  three instrument classes in their own tables instead of discarding 79% of the pull.
  Logged in Phase 4 (spreads → carry) and Phase 3 (butterflies → curve-curvature, a
  candidate not a task, same "menu not to-do-list" treatment as other out-of-scope
  ideas).
- Update (2026-07-16): wrote the precise transform-pipeline design (Phase 4) ahead of
  building it — LE stays the experimental first case, with the other 32 CME assets'
  pulls arriving next. Re-confirmed the exact target schema by re-reading
  `capture_term_structure.py` directly rather than from memory (`contract_symbol =
  f"{root}{code}{yy:02d}.{exchange}"`, e.g. `LEZ26.CME` — not the ad hoc 4-digit-year
  form used during exploration). Design reuses `UNIVERSE`/`MONTH_CODES` from that same
  file rather than redefining them, and is built in polars end-to-end (`pl.scan_csv`
  lazy multi-file reads, not the exploration script's per-file Python loop) since this
  is the actual production path, not a one-off. Two real decisions flagged rather than
  silently resolved: which source wins on the 2026-07-13/07-14 boundary if
  Databento/yfinance ever overlap on the same `(date, contract_symbol)`, and the
  proposed leg-decomposed schema for the new spread/butterfly tables.
- Update (2026-07-16): built `databento/transform_databento.py` and validated it
  against LE, per the sequencing above. Found two real gaps in the design just
  written (not assumed from the single-example prior exploration, caught by
  smoke-testing at full scale): the definition row's `maturity_year`/`month` only
  disambiguates one leg of a spread/butterfly, not all of them, and it isn't
  reliably the first-listed one (937 of 3,011 LE spreads have it describing the
  second leg) — fixed with an anchor-leg-plus-modular-offset algorithm, validated
  3,011/3,011 spreads and 141/141 butterflies resolved, 0 unresolved, every LE-side
  leg cross-checked against LE's own real outright universe with 0 mismatches. And
  ~35% of LE's unique "spread" symbols turned out to be inter-commodity (LE-HE)
  pairs, not calendar spreads on a single root — kept, per direct instruction, in a
  generalized near/far schema with explicit `near_root`/`far_root` columns, since
  they're headed for the dashboard and a future synthetically-sourced tracking
  effort, not just carry. Full detail in Phase 4's "Built & validated against LE"
  note. A background run that auto-discovered and processed 9 assets (more zips had
  arrived mid-session than expected) was reverted at the user's request to keep this
  pass scoped to LE only, per direct instruction; the multi-asset runner itself is
  unchanged and ready when needed.
- Update (2026-07-20): corrected "38-asset universe" language throughout this file,
  `CLAUDE.md`, and `DATA_SCHEMA.md` to "42-asset universe" — the term-structure/carry
  side already covered 42 as of the Databento rollout completing, and the docs hadn't
  caught up. Then closed the gap on the data side too: added SwissFranc, MexicanPeso,
  and Lumber to `get_data.ipynb`/`jobs/update_data.py` (all three live-tested for real
  Yahoo continuous-ticker history first — `6S=F`/`6M=F` have full ~25yr histories,
  `LBR=F` has the real ~4yr history of the current physically-settled contract),
  re-ran the pull (41/41, 0 failures), re-ran `jobs/update_volatility.py` (automatic,
  pure function of the OHLC files), and added the 3 assets to
  `jobs/update_dashboard_summary.py`'s `ASSET_CALENDAR`. **SOFR confirmed permanently
  unfit for this pipeline** — `SR3=F`/`SR1=F` resolve on Yahoo but return only 1 row,
  not real history — so the core signal-research universe tops out at 41, not 42;
  SOFR remains term-structure/carry-only.
- Update (2026-07-21): audited whether the dataset built this session is "flawless" —
  it isn't. Found a real, previously-undocumented OHLC-consistency problem in the core
  `yfinance` panel (3.84% of cells, up to ~13% of days for Soybeans/Wheat/Corn),
  confirmed it isn't the same mechanism as the existing roll-splice caveat (no
  periodicity clustering near roll cadence), and confirmed a concrete downstream
  consequence in `yang_zhang_features.parquet` (real mid-series NaN gaps, not just
  warm-up). Also found a narrower residual OHLC-violation rate in the 5 ICE softs
  within `term_structure.parquet` itself (2.9-9.25% of their rows, concentrated in
  far-dated thin contracts), and an unexplained gap between the transform's own
  manifest (0 violations logged for Coffee) and the current merged file (623) that
  still needs root-causing. Decided, after discussion, not to erase any of this data —
  built `Data/asset_trusted_since.csv` (per-asset real Databento-coverage start date)
  and split the dataset into a trusted era (that date forward, primary basis for
  signal calibration) and a legacy era (pre-2010 `yfinance`-only history, kept but
  demoted to an out-of-sample robustness check only). Full detail, numbers, and the
  rejected alternatives in `DATA_SCHEMA.md` section 1 and Phase 0 above. Flagged, not
  built: re-examining the train/held-out split on the shorter trusted window, and
  constructing an actual continuous front-month series from `term_structure.parquet`
  to eventually replace the Yahoo-spliced series signals trade off of.
- Update (2026-07-21): re-examined the train/held-out split (flagged above) and moved
  the project off notebooks for new research work, per direct instruction — this
  platform is about to grow into breakout/crossover/RV signal research, regime
  identification, portfolio construction, and backtesting, all of which need shared,
  tested `.py` logic rather than copy-paste between notebooks. Pulled `cleanup.md`
  section 2's `src/signals/`/`src/backtest/` split early (same kind of deliberate
  early-trigger already used for the portfolio scaffolding), and added a `src/data/`
  package not originally planned there, motivated directly by the trusted-era work
  above. New layout: `src/data/` (panels, universe, trusted-era masking),
  `src/signals/` (`transforms.py` generic across every signal family,
  `momentum.py` the first per-family module), `src/backtest/` (`engine.py`,
  `performance.py`, and new `splits.py`). `src/signal_lib.py` is now a thin
  backward-compat shim, re-exporting the moved functions unchanged — confirmed live
  (ran the shim's `load_price_data`/`get_liquid_universe` directly, same 35-asset
  universe and excluded-list output as calling the new `data/` module functions
  directly) kept only so `Time_Series_Models.ipynb` (cointegration, untouched this
  pass) keeps working unchanged until Phase 2d migrates it too. Decided the actual
  split boundary: train ≤2019-12-31 (unchanged from the original), **validation
  2020-01-01 → 2021-12-31 (COVID deliberately placed here, not in test — a fast,
  violent crash doesn't discriminate much between "seen during fitting" and not, so
  its value as a pristine test period is low; validation is where a real
  hyperparameter, once one exists, gets picked)**, **test 2022-01-01 → present
  (touched once, still contains its own genuine stress event — the 2022 inflation/
  rate-hike/commodity shock — so "test isn't just quiet markets" is preserved with a
  different anchor)**. Built `research/momentum.py`, the first driver script in the
  new structure, porting `feature_engineering.ipynb`'s momentum logic unchanged
  except for the two methodology updates above — see Phase 2a for the resulting
  (materially different, honestly reported) numbers. Added `tests/` (new, mirrors the
  Port Congestion project's `pytest` convention) with focused unit tests on the two
  new pure functions (`restrict_to_trusted_era`, `train_validation_test_split`), all
  8 passing. Explicit non-goals, so this doesn't read as more than it is: did not
  build the Databento cross-validation/correction step for the post-2010 panel, did
  not build the actual roll-splice continuous-series construction, did not touch
  cointegration. `feature_engineering.ipynb` and the other existing notebooks are
  left in place, untouched, as historical record — not deleted, just no longer where
  new work happens.
- Update (2026-07-21, same day): checked whether trusted-era masking actually fixed
  the core panel's data-quality problem, rather than assuming it — it didn't. The
  post-`trusted_since` portion of `close.parquet` etc. still has a 0.53% OHLC-
  violation rate (755 of 143,495 cells), and `research/momentum.py`'s backtest was
  running against that same flawed, unaudited Yahoo splice with the worst part
  masked off, not a real continuous curve. Documented the actual construction as the
  new standing #1 priority, blocking further Phase 2 signal backtesting, in Phase 0
  above — general-purpose infrastructure (every price-level-based signal needs it,
  not just momentum), not built yet. Roll rule decided: volume crossover now (real
  per-contract volume already in `term_structure.parquet`), fixed-days-before-expiry
  as a backstop for thin contracts, open-interest crossover flagged as the actual
  industry-standard target method once OI is purchased (`DATA_SCHEMA.md` §1, ~$50-100,
  not bought yet). Adjustment method decided: build both a raw/unadjusted series (for
  TCA, the Phase 10d backtest-vs-live log, dashboard display, and as ground truth to
  validate the roll logic against) and an additive/"Panama" back-adjusted series (for
  signal construction/backtesting) — deliberately not ratio/proportional adjustment,
  which is mathematically undefined across WTI's real 2020 negative print already in
  this dataset. Phase 2a's momentum result marked provisional pending this. Added
  open-decisions-log entry #6. Not yet built: the actual construction script, exact
  location/schema decided when that work starts.
- Update (2026-07-21, same day again): built it. `src/data/continuous_curve.py`
  (chain construction, front-contract assignment with confirmation-day and
  fixed-days-before-expiry-backstop rolling, raw series, additive back-adjustment,
  wide-format loaders matching `data.panels.load_core_panel()`'s shape) plus
  `databento/build_continuous_curve.py` (driver, all 42 assets). 7 unit tests against
  synthetic data (`tests/test_continuous_curve.py`) all passed on the first attempt —
  but running against the real 1,318,677-row dataset immediately surfaced two real
  algorithmic bugs neither synthetic case had covered: (1) MexicanPeso/Palladium/
  Platinum got permanently stuck on one contract for up to 96.9% of their entire
  history, because the chain-successor logic only ever checked the immediately-
  adjacent contract, and some chain entries (e.g. a thin serial-month FX contract
  with 2 total rows) never had an overlapping real observation to compare or roll
  into — fixed by scanning forward for the nearest chain entry with a real row on
  the current date, not assuming adjacency; (2) all 4 remaining ICE softs
  initialized to a contract that hadn't started trading yet (`fillna(0.0)` before
  `idxmax()` on day 1 let "not listed" tie with "a genuine zero print"), costing
  Coffee 32.6% of its entire curve from a single bad first pick — fixed with
  `dropna()` before `idxmax()`. Residual after both fixes: 1,085 of 170,666 rows
  (0.6%) still `NaN` — checked directly, scattered across dozens of contracts/dates
  including recognizable holidays (2010-07-04, 2012-07-04), consistent with genuine
  isolated missing prints, left as `NaN` rather than fabricated. Spot-checked (not
  just trusted): `raw_close` matches `term_structure.parquet`'s own quoted price
  exactly in every sample checked across WTI/Lumber/Coffee; WTI shows 198 rolls over
  16 years (~monthly, matches its real listing cycle); sampled roll-date adjusted
  price changes are small and sane, not the raw mechanical gap. Output:
  `Data/continuous_futures.parquet`. **Not done**: re-pointing `research/momentum.py`
  (or anything else) at this new curve — that's the deliberate next step, not part
  of this build; promoting the driver to a scheduled job.
- Update (2026-07-21, later same day): re-pointed `research/momentum.py` at the new
  curve and rebuilt momentum to match Moskowitz-Ooi-Pedersen (2012) exactly — see
  Phase 2a above for the full recipe, the final result (train/validation/test Sharpe
  0.239/0.475/0.402), and the complete bug-fix narrative. Along the way, found and
  fixed a real bug in the continuous-curve construction itself, not just in the
  signal built on top of it: additive back-adjustment pushed HeatingOil/RBOB/Brent/
  WTI Crude/Oats through zero in old segments, corrupting percentage returns
  directly (RBOB hit a -26,740% one-day `pct_change()`). The original justification
  for additive over ratio (WTI's real 2020 negative print makes ratio undefined)
  turned out not to apply to this specific series — checked directly, raw
  front-contract close is never non-positive for any of the 42 assets, including
  WTI, because the roll rule rolls out of each contract before the per-contract
  event Databento genuinely captured (`DATA_QUALITY_REPORT.md` Asset 17) could ever
  reach the front-continuous series. Corrected to ratio/proportional adjustment
  (see this phase's "Adjustment method" section above for the full corrected
  reasoning) — verified 0 rows with non-positive `adj_close` anywhere after
  rebuilding `Data/continuous_futures.parquet`. `tests/test_continuous_curve.py`'s
  back-adjustment test and `dashboard/pages/05_continuous_curve.py`'s copy (built
  earlier this session) both updated to match. Also added `src/data/volatility.py`
  (Yang-Zhang, ported from `volatility.ipynb`) and `src/data/ewma_volatility.py`
  (new) for momentum's vol-estimator comparison, `signals/transforms.py` gained
  `vol_targeted_sign_signal`, and `backtest/engine.py` gained
  `holding_period_positions` (Jegadeesh-Titman-style holding-period blending) and
  `backtest_signal_per_asset` (un-normalized single-asset view, for the paper's
  Figure-2-style per-instrument evaluation).
