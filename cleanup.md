# Codebase Cleanup / Architecture Plan

Working document for the `Scripts/` → modular-folder reorg (2026-07-15) and the
forward architecture it sets up for. `WORKFLOW.md` remains the source of truth for
signal-research phasing; this file is specifically about code organization and the
portfolio-construction layer that's starting to get built ahead of the normal
sequencing (see "Why portfolio code now" below).

---

## 1. What changed — `Scripts/` → 5 top-level folders

Everything used to live flat in one `Scripts/` folder (notebooks, scheduled jobs,
shared library code, the Databento backfill scripts, and the dashboard all mixed
together). Split by concern instead:

| Folder | Contents | Why separate |
|---|---|---|
| `src/` | `signal_lib.py`, `macro_point_in_time.py` | Shared library code, importable from notebooks, jobs, and eventually the portfolio layer. Not a full installable package (no `pyproject.toml`) — a `sys.path.insert` per consumer is enough at this size. |
| `jobs/` | `update_data.py`, `update_volatility.py`, `capture_term_structure.py`, `update_macro_data.py`, `update_dashboard_summary.py` | The 5 Windows Task Scheduler targets (`CTA_DailyDataUpdate`, `CTA_VolatilityUpdate`, `CTA_TermStructureCapture`, `CTA_MacroDataUpdate`, `CTA_DashboardSummary`). Grouping these makes "what runs unattended every day" a single, obvious folder rather than mixed in with research code. |
| `databento/` | `submit_databento_jobs.py`, `backfill_databento.py` | The historical-backfill sub-initiative — its own state machine (batch jobs, rate limits, a transform/join stage not yet built), documented in `WORKFLOW.md` Phase 4. Not a daily job, not shared library code — it's its own thing. |
| `notebooks/` | `get_data.ipynb`, `volatility.ipynb`, `feature_engineering.ipynb`, `Time_Series_Models.ipynb` | Research notebooks — the project's source of truth for what's actually been tested, per `CLAUDE.md`. |
| `dashboard/` | `app.py`, `lib.py`, `pages/*.py` | The QA dashboard (promoted out of `Scripts/dashboard/` to a top-level sibling). |

**Mechanical fixes required by the move** (everything else's relative-path depth was
unaffected since the new folders sit at the same depth `Scripts/` did):
- `jobs/update_dashboard_summary.py`'s `sys.path.insert` now points at `../src` instead
  of its own folder (it imports `macro_point_in_time`, which moved).
- `dashboard/lib.py`'s `DATA_DIR` went from 3 `.parent` calls to 2 (it's no longer
  nested inside `Scripts/`).
- `notebooks/feature_engineering.ipynb` and `notebooks/Time_Series_Models.ipynb` both
  did a bare `from signal_lib import ...`, which only worked because Jupyter's cwd was
  the same folder `signal_lib.py` lived in. Added one `sys.path.insert(str(Path.cwd().parent
  / "src"))` line to each notebook's import cell.
- All 4 existing Task Scheduler actions repointed at `jobs/`; new `CTA_DashboardSummary`
  task registered (6:25PM daily, same `Interactive`/logged-in-user pattern as the other
  4). Verified via an actual `Start-ScheduledTask` run, not just direct script execution.

**Not fixed, logged instead**: `get_data.ipynb`, `volatility.ipynb`, and
`jobs/update_data.py` all use a hardcoded absolute `C:\Users\pcarg\...` path rather than
`Path(__file__)`-relative resolution. Unaffected by this move (absolute paths don't
care where the file itself lives), but a portability wart if this repo is ever cloned
to a different machine or a different user account. Left alone here since editing
`get_data.ipynb`/`volatility.ipynb` beyond what the move required risks touching
research notebooks unnecessarily — a candidate for a future pass, not urgent.

**Git**: this repo had no version control before this reorg. Initialized (`git init`),
committed a pre-reorg snapshot as a checkpoint, then committed the reorg itself as a
second commit. `.gitignore` excludes `Data/` (regenerated outputs, some large/daily-
growing) and `deprecated/` (includes a 228MB file that exceeds GitHub's hard size limit
outright) — `DATA_SCHEMA.md` and `deprecated/README.md` remain the record of what those
contain.

---

## 2. Forward split points inside `src/` — pulled early 2026-07-21, done

`src/` stayed flat (`signal_lib.py`, `macro_point_in_time.py`) through 2026-07-20
because there was only one signal family (momentum) — splitting into subpackages
before that would have meant guessing at Phase 2's shape rather than responding to
real duplication pressure, exactly the mistake `CLAUDE.md` Rule 6 already documents
fixing once. **Pulled early on 2026-07-21, before Phase 2b/2c/2d actually landed** —
same kind of deliberate, logged exception already used for the portfolio scaffolding
in section 3 below, not a precedent for skipping the "wait for real pressure" rule
generally. Trigger for pulling early: the trusted-era/legacy-era data split (found the
same day, see `DATA_SCHEMA.md` section 1) needed a real home that wasn't a notebook,
and the user asked to stop building new research logic in notebooks altogether ahead
of signal research, regime identification, portfolio construction, and backtesting all
expanding at once — waiting for Phase 2b to literally land first would have meant
building that data-layer logic in a notebook anyway, then re-migrating it.

| Piece | Status | Shape |
|---|---|---|
| `src/data/` (`panels.py`, `universe.py`, `trusted_since.py`) | **Built 2026-07-21** — not originally planned in this table; added directly in response to the trusted-era masking need | Pure functions, no optimizer dependency |
| `src/signals/` (`transforms.py`, `momentum.py` built; `breakout.py`/`crossover.py`/`relative_value.py` still to come with Phase 2b/2c/2d) | **Built 2026-07-21** | Pure functions, no optimizer dependency, each returns `(T×N)` |
| `src/backtest/` (`engine.py`, `performance.py`, `splits.py`) | **Built 2026-07-21** — `splits.py` wasn't in the original plan either; added for the train/validation/test split decided the same day | `backtest_signal`/`performance_stats` peeled off `signal_lib.py`; `splits.py` is new, not a peel |
| `src/portfolio/` | Built 2026-07-15 — see section 3 | `Book`/`Allocator`, adapted from the retired stat-arb engine |
| `src/regime/` | Built 2026-07-15 (interface only) — see section 3 | Macro-driven, not DCC-GJR crowding-based |

`src/signal_lib.py` is now a thin backward-compat shim re-exporting the moved
functions, kept only because `Time_Series_Models.ipynb` (cointegration, untouched by
this migration) still imports from it — retire it once Phase 2d migrates that
notebook to the new structure too. New research work happens in `research/*.py`
driver scripts (first one: `research/momentum.py`) importing from `src/`, not in new
notebooks.

---

## 3. Portfolio construction — why now, and what transfers

`CLAUDE.md` Rule 6 says reuse Book/Allocator "once a second signal family exists — not
before." This deliberately jumps ahead of that: not premature abstraction from scratch,
but porting an already-built, already-evaluated engine
(`Projects/Cross Asset Stat Arb Engine/engine/`) whose shape was reviewed against this
project's actual needs before writing anything. Logged here so it doesn't read as an
inconsistency later.

Evaluated against the retired 13-name stat-arb engine's `portfolio/book.py`,
`allocator.py`, `optimizer.py`, and `regime/regime_mapping.py`:

**Keep, adapted:**
- `chernov_weights` optimizer (closed-form quadratic MV: `(γΣ + κI)⁻¹(α + κx_prev −
  λ·sign(x_prev))`) — pure function, portable. **Drop the dollar-neutrality
  constraint** (`x - x.mean()`) — that's a market-neutral stat-arb requirement, not a
  CTA one. **Alpha must not be assumed to be in Fama-MacBeth-aligned E[r] units** — this
  project's momentum/breakout signals are continuous, vol-scaled, per-asset time-series
  values, not cross-sectionally calibrated.
- Ledoit-Wolf rolling covariance — the shrinkage tool transfers (arguably matters more
  at 38 assets than 13). The specific input (`residual_ret`, beta-neutralized) does
  not — there's no cross-sectional beta-neutralization step for trend/breakout signals.
  Σ must be built from whatever return series the signals actually trade.
- `Book(alpha, covariance, optimizer params, vol-target config, is_active).run(returns)
  -> {weights, pnl, sharpe, ...}` — the right unit for adding signal "sleeves"
  continuously. New signal family = new `Book`, independently tunable, without
  touching existing books.
- `Allocator(books).run(returns) -> combined pnl` — already minimal (run each active
  book, sum PnL), no regime logic baked in. Transfers as-is.
- The regime→book-action **interface**: `get_actions_for_date(regime_df, date,
  action_fn, book_names) -> {book_name: {"active": bool, "alpha_multiplier": float}}`
  — built in `src/regime/interface.py` as a generic lookup (finds the latest regime
  label on or before `date`, no look-ahead by construction, then resolves it via a
  caller-supplied `action_fn`) rather than a fixed decision table, so the actual
  regime→action mapping is swappable without touching the lookup mechanism. Applied to
  alpha **before** the optimizer runs — never post-solve, since vol targeting silently
  cancels out any post-solve scaling.

**Adapt for cleaner sleeve-mixing** (per direct request — research will explore mixing
signals within a sleeve vs. combining separate sleeves at the portfolio level):
- `Book.run()` gets split into composable steps (`_solve_weights`,
  `_apply_vol_target`, `_apply_constraints`, `_compute_pnl`) rather than one monolithic
  method, so a future book type (e.g. an RV-spread book with z-score entry/exit instead
  of continuous optimizer sizing) can override one step instead of copy-pasting the
  whole thing.
- A new `src/signals/combine.py` — pure functions to blend multiple alpha DataFrames
  before they reach a `Book` (`combine_alphas(alpha_dfs, weights=None,
  method="equal"/"ic_weighted"/"rank")`), giving both mixing modes (blend into one
  `Book`, or keep separate `Book`s and let the `Allocator` combine them) without
  forcing the decision now.

**Discard:**
- `normalize_alpha` (cross-sectional z-score + MAD winsorize across the universe) —
  reasonable for 13 comparable mean-reversion names, questionable across 38 assets
  spanning incomparable vol regimes (energy vol vs. rates vol vs. FX vol). This
  project already solved the analogous problem differently (per-asset Yang-Zhang
  vol-scaling), consistent with Rule 5's own finding that binary/unscaled signals
  underperform.
- The specific `normal/clustered/crowded/crisis/broken` regime definitions and their
  DCC-GJR correlation-spike detection — assumes crisis hurts the strategy, backwards
  for trend's documented crisis-alpha property (`CLAUDE.md` Rule 7). Being replaced
  with a macro-driven regime classifier (growth/inflation, GSCPI supply-chain stress,
  yield-curve shape — data already collected, sitting unused per `DATA_SCHEMA.md`
  section 3, and `macro_point_in_time.py` already solves point-in-time correctness for
  it).

**DCC-GARCH — a different role, not discarded outright.** The *tool* (dynamic,
time-varying covariance/correlation) is legitimate for CTA risk management: a
correlation-regime spike is a real diversification-breakdown signal that institutional
trend books use to throttle **gross leverage**, independent of whether the signal
itself still works. That's different from the retired project's use (correlation spike
→ suppress a specific book's alpha), which assumed crisis hurts performance. If reused,
it belongs in the covariance/risk layer (an alternative or supplement to rolling
Ledoit-Wolf Σ, or a gross-exposure throttle in the `Allocator`), not as a per-signal
on/off switch — and DCC-GARCH estimation gets harder at 38 assets than the 13-name
case it was built for, so "does DCC-based Σ actually beat rolling Ledoit-Wolf here" is
an empirical question for Phase 7/8, not an assumption.

---

## 4. Status

| Component | Status |
|---|---|
| Reorg (`src/jobs/databento/notebooks/dashboard`) | Done 2026-07-15 |
| git init + initial commits | Done 2026-07-15 |
| Task Scheduler updated (5 tasks) | Done 2026-07-15, verified via real scheduled run |
| `src/portfolio/` (optimizer, book, allocator) | Building now — see `WORKFLOW.md` Phase 5/7 for when this actually gets exercised against real signals |
| `src/signals/combine.py` | Building now |
| `src/regime/` | Interface stub now; macro-driven classifier content is a separate research task |
| `src/data/`, `src/signals/`, `src/backtest/` subpackage split | Done 2026-07-21, pulled early — see section 2 |
| `research/` driver scripts (notebook replacement for new work) | Started 2026-07-21 — `research/momentum.py` |

---

## 5. Change log

- 2026-07-15: Initial version. Documents the `Scripts/` reorg and the portfolio
  construction evaluation against `Projects/Cross Asset Stat Arb Engine/engine/`.
- 2026-07-21: Section 2's `src/signals/`/`src/backtest/` split pulled early (see
  section 2 for the full rationale) alongside a new `src/data/` package and a
  `research/` layer of driver scripts replacing notebooks for new work. Full detail
  in `WORKFLOW.md`'s 2026-07-21 change log entry.
