# Data Schema

Inventory of everything in `Data/` — what's collected and used, what's collected but
unused, and what's missing. Verified against the actual files, not assumed. See
`WORKFLOW.md` for how each of these feeds the roadmap.

---

## 1. Core price/volume pipeline — done, in production use

**Note: this section documents 41 of the project's 42 assets.** SOFR is permanently
excluded from this pipeline — no usable Yahoo continuous ticker exists for it (`SR3=F`/
`SR1=F` both resolve but return only 1 row of history, checked live 2026-07-20). It still
has real coverage via the term-structure/carry pipeline (section 4b). SwissFranc,
MexicanPeso, and Lumber were added here 2026-07-20 (previously only in the
term-structure universe since 2026-07-17) — see the change log.

Source: `notebooks/get_data.ipynb` / `jobs/update_data.py` (kept in sync, same universe
dict), `yfinance`, `period="max"`, `auto_adjust=False`.

**Files (`Data/`):** `open.parquet`, `high.parquet`, `low.parquet`, `close.parquet`,
`adj_close.parquet`, `volume.parquet` — each `(6503+ rows) × 41 columns`, indexed by
`Date`. Plus `metadata.csv` (per-asset row count / start / end) and `audit.csv` (the same
audit run before the production download, kept as a coverage sanity check).

**Universe (41 = 24 commodities + 17 financial futures):**

| Sector | Assets (Yahoo ticker) |
|---|---|
| Energy | WTI Crude (`CL=F`), Brent (`BZ=F`), Natural Gas (`NG=F`), RBOB (`RB=F`), HeatingOil (`HO=F`) |
| Precious Metals | Gold (`GC=F`), Silver (`SI=F`), Platinum (`PL=F`), Palladium (`PA=F`) |
| Industrial Metals | Copper (`HG=F`) |
| Grains | Corn (`ZC=F`), Soybeans (`ZS=F`), Wheat (`ZW=F`), KC_Wheat (`KE=F`), Rice (`ZR=F`), Oats (`ZO=F`) |
| Softs | Coffee (`KC=F`), Sugar (`SB=F`), Cocoa (`CC=F`), Cotton (`CT=F`), OrangeJuice (`OJ=F`) |
| Livestock | LiveCattle (`LE=F`), LeanHogs (`HE=F`), FeederCattle (`GF=F`) |
| Equity Index | SP500 (`ES=F`), Nasdaq100 (`NQ=F`), Dow (`YM=F`), Russell2000 (`RTY=F`) |
| Rates | US_2Y (`ZT=F`), US_5Y (`ZF=F`), US_10Y (`ZN=F`), US_30Y (`ZB=F`), UltraBond (`UB=F`) |
| FX | EURUSD (`6E=F`), JPYUSD (`6J=F`), GBPUSD (`6B=F`), AUDUSD (`6A=F`), CADUSD (`6C=F`), SwissFranc (`6S=F`), MexicanPeso (`6M=F`) |
| Lumber (own category) | Lumber (`LBR=F`) |

**Coverage notes (from the 2026-07-20 production audit):** all 41 assets returned data,
zero failures. Longest history: Rice (6,728 rows, from 1999-09-14). Shortest: Lumber
(992 rows, from 2022-08-05 — the current physically-settled contract's real
inception date, not a gap; see WORKFLOW.md Phase 4 for why the older `LBS` contract
wasn't used instead). Highest missingness in `close.parquet`: Lumber (84.7%, purely a
function of the panel's index going back to 1999 while Lumber only starts 2022),
Russell2000 (65.1%), UltraBond (36.1%), Brent (27.4%) — all genuine contract-inception
gaps, not data errors. Most of the panel starts within days of 2000-08-23; SwissFranc
(2000-11-10) and MexicanPeso (2001-06-18) fit that same early cohort; Lumber (2022) and
Russell2000 (2017) are the real outliers.

**Known data-quality patch already applied downstream (not yet pushed upstream):**
`feature_engineering.ipynb` nulls out a bad Rice print on 2024-06-17 across
open/high/low/close before computing anything. This fix currently lives in that one
notebook — if/when Phase 1 (`WORKFLOW.md`) extracts a shared cleaning step, this belongs
there so every downstream notebook gets it automatically instead of by convention.

**Low-confidence theoretical caveat, not yet acted on (logged 2026-07-14):** Yahoo's
`=F` continuous futures tickers are, on general/documented grounds, a raw front-month
price splice rather than a back-adjusted continuous series — a genuine mechanical fact
about how the vendor constructs this ticker type. This would mean `close_prices.pct_change()`
(used for every return calculation in this project) picks up a spurious jump at each
contract rollover. **However, a direct empirical test found no strong evidence this is a
material problem in this specific data:** across WTI, Corn, Gold, US_10Y, EURUSD, and
Natural Gas, the top-1%-by-magnitude return days did not cluster at the expected
contract-roll cadence (gaps between them were widely spread, not tightly centered near
~21 or ~63 trading days), and volume on those days was mostly ~1x the prior day's (no
consistent regime-shift signature). An earlier draft of this note over-claimed a
"confirmed" defect based on one coincidental example (a big WTI move landing near a
predicted expiry date) that this broader test does not support. **Do not treat this as
validated either way** — the only way to know for certain is to diff Yahoo's continuous
series against real contract-to-contract prices on actual documented expiry dates, which
requires historical individual-contract data (blocked on the Databento backfill, see
`WORKFLOW.md` Phase 4). Revisit then; not tracked as an active roadmap item until there's
real evidence either way.

**Resolved, not just theorized (2026-07-21): the core panel has a real, material OHLC-
consistency problem, separate from the roll-splice caveat above.** Checked directly
against every non-null cell in `open/high/low/close.parquet`: **3.84% of all (date,
asset) cells (9,558 of 249,186) violate basic OHLC consistency** — `Close` or `Open`
falls outside that day's `High`-`Low` range, which is impossible for genuine trading
data. Concentrated, not uniform: Soybeans/Wheat/Corn each have 800+ violation days
(~12-13% of their full history), followed by Coffee, Cocoa, Oats, Cotton, and the
precious metals; only Russell2000 and Lumber (both short-history assets) are clean.
Magnitude ranges from tick-level noise (`$0.25`) to material (`$111` on Soybeans,
2008-09-12). **Explicitly checked and ruled out as the roll-splice mechanism above** —
violation dates don't cluster near roll cadence, they're scattered with no clean
periodicity, so this looks like separate raw print/tick noise in Yahoo's feed, not the
same issue as the caveat above. **Confirmed downstream consequence**: `yang_zhang_
features.parquet` has real mid-series NaN gaps beyond warm-up (172 unexplained NaN days
for Soybeans' 21-day vol, 316 for Corn) because a bad OHLC print can push the
Rogers-Satchell variance component negative for every rolling window it falls inside —
the "Done... correct" characterization of Yang-Zhang volatility in `CLAUDE.md`'s
current-state table needs this caveat attached.

A second, narrower version of the same class of problem was found in `term_structure.
parquet` itself: the ICE softs (Coffee 9.25%, Cocoa 5.9%, Cotton 4.9%, Sugar 4.65%,
OrangeJuice 2.9% of their own rows) have a real residual OHLC-violation rate,
concentrated in far-dated, thin contracts (e.g. `KCN27.NYB`, a July-2027 Coffee
contract quoted in 2023-2024, 3+ years before expiry) — the same "synthetic/
theoretical settlement print on an illiquid far contract" mechanism already documented
for `CLQ26` in this same section, just not connected to this table before. Notably,
`databento/transform_databento.py`'s own validation reported **0 OHLC violations for
Coffee at transform time** — the discrepancy between that and the 623 found now in the
merged table is itself unexplained and worth root-causing, since it means a clean
manifest entry isn't necessarily authoritative for the current state of the merged
file. The overall rest of `term_structure.parquet` is close to clean (0.14% violation
rate outside the ICE softs, 0 duplicate `(date, contract_symbol)` rows) — this is a
narrower, ICE-specific residual, not a systemic problem with the Databento side.

**Decision (2026-07-21): don't erase either dataset — split into a trusted era and a
legacy/robustness-only era, per asset.** Blanket deletion was considered and rejected:
a deleted date is indistinguishable from a holiday/non-trading day unless tracked
separately, which is worse than a known-noisy value for rolling-window signals (vol,
momentum, and the still-unbuilt Donchian breakout, which depends directly on `High`/
`Low` integrity). Instead:

- **`Data/asset_trusted_since.csv`** (new, 2026-07-21) — one row per asset, giving the
  real date from which `term_structure.parquet` has genuine Databento-sourced coverage
  for that asset (verified directly against the file, not assumed uniform). 33 of 42
  assets start 2010-06-06/07 (the standard `GLBX.MDP3` history window); the rest are
  real exceptions with a documented reason each: Russell2000 (2017), SOFR (2018), and
  Lumber (2022) are genuine product-inception dates, not gaps; the 5 ICE softs
  (2023-08 → 2024-07, staggered by asset) reflect the budget-trimmed backfill scope
  from Phase 4, not a technical limit; **KC_Wheat (2013-12-16) is a real CME asset that
  starts later than the standard 2010-06-06 window for a reason not yet root-caused** —
  flagged, not explained.
- **Everything from an asset's `trusted_since` date forward is the primary basis for
  signal calibration** — real, exchange-quoted, largely-clean data, and it keeps
  extending forward at the same richness via the daily `capture_term_structure.py`
  forward-capture (which pulls every currently-listed contract month, not just the
  front month, so the "rich era" doesn't degrade back into a single series going
  forward).
- **Everything before `trusted_since` (the pre-2010 `yfinance`-only continuous-series
  history, still sitting in `close.parquet` etc.) is retained, not deleted, but
  demoted to an out-of-sample robustness check only** — "does an already-calibrated
  signal still look directionally sane on the older, noisier, structurally-different
  tape" — never used for primary calibration or the train/held-out split. Rationale:
  it's both structurally poorer (no per-contract identity, no curve, an unaudited
  Yahoo roll/splice convention) and now confirmed noisier (the 3.84% finding above),
  and post-2008 electronic/HFT-dominant market microstructure is different enough from
  the pre-2010 tape that it's not obviously more representative of what a signal will
  actually trade against going forward anyway.
- **Real trade-off, not free**: `CLAUDE.md` Rule 1's train/held-out split (train ≤2019,
  held-out 2020+) was built against the full pre-2010 history — restricting to
  `trusted_since` shrinks the train window from 19 years to as little as 9 (for the
  33 assets starting 2010) and much less for KC_Wheat/Russell2000/SOFR/Lumber/the ICE
  softs. Whether that split boundary still makes sense on the shorter window is an
  open question for whenever Phase 2 signal work resumes, not decided here.
- **Built (2026-07-21) — a proper continuous curve constructed from
  `term_structure.parquet`'s real per-contract data.** Prompted by finding, directly
  checked, that the trusted-era mask above never solved this on its own — the
  post-`trusted_since` portion of the core panel still had a 0.53% OHLC-violation
  rate, still Yahoo's unaudited splice. `src/data/continuous_curve.py` +
  `databento/build_continuous_curve.py` now build both a raw/unadjusted and a
  back-adjusted series for all 42 assets, written to `Data/continuous_futures.parquet`
  (170,666 rows). General-purpose, not momentum-specific: every signal that trades a
  single per-asset price level (breakout, crossover, momentum) can use this;
  RV/cointegration/carry don't need it, since they already consume
  `term_structure.parquet`'s real multi-contract data directly. Two real algorithmic
  bugs (a stuck-front chain gap affecting up to 96.9% of some assets' history, and a
  bad day-1 initialization costing Coffee 32.6% of its curve) were found only by
  testing against the real data, fixed, and left a small, honest 0.6% residual of
  isolated `NaN` gaps rather than fabricated values. Full construction spec and the
  complete bug-fix narrative — in `WORKFLOW.md` Phase 0.
- **Back-adjustment method corrected from additive to ratio/proportional (2026-07-21,
  same day, while building `research/momentum.py`).** Originally additive ("Panama"),
  chosen on the assumption that a ratio adjustment would be undefined across WTI's
  real 2020 negative print. That assumption was checked directly against this dataset
  and found false: raw front-contract close is never non-positive for any of the 42
  assets, including WTI (min 12.26) — the roll rule rolls out of each contract before
  expiry, so the front-contract series never actually captures that event. Meanwhile
  additive had a real, live bug: it pushed HeatingOil, RBOB, Brent, WTI Crude, and
  Oats through zero in old segments (250-3663 rows each with `adj_close <= 0`), which
  is worse than "an old segment can go negative" — it's a percentage-return
  singularity right at the crossing (`pct_change()` on RBOB hit -26,740% on one day).
  This corrupted every downstream percentage-return calculation for those assets, not
  just display. Ratio adjustment is a product of positive numbers, so it's always
  positive by construction (verified: 0 rows with `adj_close <= 0` anywhere in the
  regenerated file), and has a second, independent benefit: it preserves percentage
  returns consistently across the whole history, unlike additive in old, heavily-
  shifted segments — a better fit for a project whose signals are built on percentage
  returns throughout. `tests/test_continuous_curve.py`'s back-adjustment test rewritten
  to ratio semantics; the dashboard's continuous-curve page (below) updated to match.
- **Consumed by a real signal as of 2026-07-21** — `research/momentum.py`'s rebuild
  (`CLAUDE.md` current-state table) uses `load_continuous_backadjusted()` for signal
  construction/returns and `load_continuous_raw()` (with `is_roll_date` masking) for
  Yang-Zhang vol. No longer "not yet done."

---

## 2. Volatility — done, in production use

Source: `notebooks/volatility.ipynb` (research/reference version) and `jobs/update_volatility.py`
(production, scheduled daily as `CTA_VolatilityUpdate` at 6:05PM, 5 minutes after the
price update — this is a pure function of `Data/open,high,low,close.parquet`, not an
independent pull, so it must run after `CTA_DailyDataUpdate` or it computes against stale
prices). Yang-Zhang OHLC volatility estimator (Yang & Zhang, 2000), combining overnight
variance, open-to-close variance, and a Rogers-Satchell range component, weighted by
`k = 0.34 / (1.34 + (window+1)/(window-1))`.

**File:** `Data/yang_zhang_features.parquet` — `MultiIndex` columns
`(yz_vol_21 / yz_vol_63 / yz_vol_126 / yz_vol_252, <asset>)`, daily, non-annualized
(annualized versions computed on the fly as `× √252` where needed, not persisted
separately). Strictly backward-looking (rolling window, no forward information).

**File:** `Data/volatility_manifest.csv` — one row per `CTA_VolatilityUpdate` run
(`run_date, status, detail`), added 2026-07-15 so this job's pipeline health can be
read from a real logged record instead of inferred from the parquet's file
modification time (see the QA dashboard, section 8 below).

---

## 3. Macro/auxiliary data — collected, now scheduled (2026-07-14), still not wired into any notebook

Source: `jobs/update_macro_data.py`, scheduled daily as `CTA_MacroDataUpdate` at
6:20PM. Each of the 5 sources updates independently (one failing doesn't block the
others), logged to `Data/macro_data_manifest.csv`. None of these had a documented source
before 2026-07-14 — identified by inspecting each existing file's schema and confirming
the live endpoint:

| File | Source | Method |
|---|---|---|
| `Yield_Curve_6M_to_30Y.csv` | FRED (`fred.stlouisfed.org`), series `DGS6MO/DGS1/DGS2/DGS3/DGS5/DGS7/DGS10/DGS20/DGS30` | No API key needed. Full re-pull of each series, resampled to monthly (mean, month-start), **requiring all 9 maturities present** (`dropna(how="any")`) to match the original file's convention — an earlier draft used `how="all"` and silently expanded the file back to 1962 with many partial-NaN rows before this was caught. |
| `overnight_fed_fund_rates_US.xlsx` | NY Fed Markets Data API (`markets.newyorkfed.org/api/rates/all/search.json`) | No API key needed. Incremental: pulls the last 45 days, appends and deduplicates on `(Effective Date, Rate Type)` against the existing 15k+ row archive. |
| `gscpi_data.xls` | NY Fed fixed URL: `newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx` (found in the page's JS bundle, not linked in static HTML) | Full re-download and overwrite each run; updates monthly on the 4th business day per NY Fed's own schedule. Actual file format is legacy OLE2 `.xls` despite the `.xlsx` URL path — needs `xlrd` to parse (not installed in this environment; the ingestion script only fetches/saves raw bytes, doesn't parse it). |
| `trade_policy_uncertainty_US.csv` | `policyuncertainty.com/media/All_Daily_TPU_Data.csv` (Caldara, Iacoviello, Molligo, Prestipino, Raffo) | Full re-download and overwrite each run; schema matches exactly (`day, month, year, daily_tpu_index`). |
| `vix_data.csv` | `yfinance`, ticker `^VIX` (CBOE S&P 500 implied-vol index) | Added 2026-07-21 for the short-term reversal signal (`references/short_term_reversal_implementation_recipe.md`) — Nagel (2011) finds reversal-strategy expected returns are strongly predictable with VIX. Full re-pull each run (`period="max"`), same pipeline convention as the core 42-asset panel (`jobs/update_data.py`), pulled here instead since VIX isn't a tradeable instrument in this project's universe. Live-verified 2026-07-21: 9,205 rows, 1990-01-02 → present, no NaN. |
| `cpi_level_index.csv` | 7 countries from FRED (`fred.stlouisfed.org`) — `CPIAUCSL` US, `CP0000EZ19M086NEST` EUR, `GBRCPIALLMINMEI` GBP, `AUSCPIALLQINMEI` AUD, `CANCPIALLMINMEI` CAD, `CHECPIALLMINMEI` CHF, `MEXCPIALLMINMEI` MXN; JPY from **e-Stat** (`api.e-stat.go.jp`), Japan's own government statistics API, directly | Added 2026-07-22 for a prospective value factor (Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere" — FX value there is a 5-year PPP-adjusted real exchange-rate return, needing relative CPI for each of `data/sectors.py`'s FX-group currencies plus US as the PPP base). FRED series need no API key. **A first attempt using FRED's OECD MEI "growth rate previous period" series family (`...M657N` codes) was abandoned after live-checking recency, not just resolution** — those series resolve fine but are stale-to-discontinued for most non-US countries. Switched to each country's CPI LEVEL index instead, which fixed 6 of 7 remaining. **JPY fixed 2026-07-23**: every FRED-hosted Japan CPI series (OECD-MEI family, level and growth-rate variants alike) was stuck at 2021-06, ~5 years stale — confirmed to be a mirror-channel problem, not a Japan-side data gap, by pulling Japan's own Statistics Bureau data directly via e-Stat (table `0003427113`, nationwide/all-items/index-level, codes found via `getMetaInfo` not guessed) — reaches 2026-05, a completely normal ~2-month lag. e-Stat requires a free `appId` (registration confirmed required by testing unauthenticated first) — stored as the `ESTAT_APP_ID` environment variable, same OS-env-var convention as `DATABENTO_API_KEY` (no `.env` file in this project). The `update_cpi()` job treats the e-Stat pull as an independently-fault-tolerant sub-source — if it fails, the other 7 FRED-sourced countries still get written, matching this file's own "one failing shouldn't block the others" principle. |

**Point-in-time integrity (checked 2026-07-14, prompted by a direct question about ragged-edge/restatement risk):** none of these 4 files distinguish "date the value describes" from "date the value was actually known" — they mirror each source's own convention, which is correct for fidelity but unsafe to join directly against a backtest date. Findings, most to least material:

- **GSCPI has a real ~1-2 week ragged edge.** Confirmed empirically: as of 2026-07-14, the file's most recent row is dated 2026-06-30 (June), consistent with NY Fed's own release schedule (4th business day of the *following* month, i.e. published ~July 4-10). A naive join on the June-30 date would be an 8-10 day look-ahead. Revision policy not confirmed from a primary source this session (the NY Fed page is JS-rendered, static scraping was blocked) — treat as unconfirmed rather than assumed-clean.
- **Fed funds reference rates have a confirmed 1-business-day publication lag** (NY Fed's own "Details on Publication and Revisions" documentation, and empirically: querying on 07-14 returns 07-13 as the latest available `Effective Date`). Revisions are possible but same-day-of-publication only — effectively final once that day passes.
- **Yield curve (Treasury par yields) is clean** — same-day market observation, not survey-based, no revision mechanism.
- **Trade Policy Uncertainty is near real-time** (a pull on 07-14 reached through 07-13) and likely low revision risk (mechanical newspaper-keyword count, not a survey statistic), but zero revisions isn't confirmed by an authoritative source.

**Fix:** `src/macro_point_in_time.py` — point-in-time accessor functions (`get_yield_curve_as_of`, `get_fed_funds_as_of`, `get_gscpi_as_of`, `get_trade_policy_uncertainty_as_of`) that apply each source's confirmed (or conservatively assumed) publication lag, so `as_of(date)` can never return a value that wasn't actually knowable on that date. Built now, ahead of any consumer, specifically so this doesn't become a look-ahead bug discovered after the fact — same discipline as `CLAUDE.md` Rules 2-3. Live-verified: `get_gscpi_as_of` correctly withholds the June value until its computed publish date (2026-07-10) and returns it from that date forward.

All four sit in `Data/` today with no notebook referencing them. Verified schemas:

| File | Schema | Frequency / range | Notes |
|---|---|---|---|
| `Yield_Curve_6M_to_30Y.csv` | `Date, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y` (par yields, %) | Monthly, 1981-09 → present (455 rows) | US Treasury par yield curve — natural fit for a rates-relative-value signal, or a macro regime feature. |
| `overnight_fed_fund_rates_US.xlsx` | Sheet `Results`: `Effective Date, Rate Type (EFFR/OBFR/TGCR/...), ...` (19 cols) | Daily, 15,218 rows | NY Fed reference rates publication (Effective Fed Funds Rate, Overnight Bank Funding Rate, Tri-Party General Collateral Rate). Needs a filter to the specific rate type wanted before use — currently long-format with multiple rate types stacked. |
| `gscpi_data.xls` | Not yet parsed (needs `xlrd`, not installed in the environment used to audit this) | Monthly | **This is the NY Fed's Global Supply Chain Pressure Index** — the same GSCPI covered in depth in the port congestion project's `LITERATURE_REVIEW.md` (paper 7: Benigno, di Giovanni, Groen & Noble 2022). Genuine cross-project overlap: if this project ever wants a supply-chain-pressure macro control, it's already sitting here, and the port congestion project's literature review is the reference for how to interpret it (and its construction methodology, if a fresher pull is ever needed — see that repo). |
| `trade_policy_uncertainty_US.csv` | `day, month, year, daily_tpu_index` | Daily, 1985 → present (15,131 rows) | Needs `day/month/year` combined into a proper date column before use. |
| `vix_data.csv` | `Date, Open, High, Low, Close, Volume` | Daily, 1990-01-02 → present (9,205 rows) | CBOE VIX index level (Close is the number quoted everywhere; Volume is always 0 - VIX is an index, not a traded instrument). Same-day observation, no revision mechanism, no publication-lag concern the way GSCPI/fed-funds have - safe to join without a point-in-time accessor (unlike the other 4 sources here). |
| `cpi_level_index.csv` | `Date, US, EUR, JPY, GBP, AUD, CAD, CHF, MXN` (index levels, different base years per country/source — not cross-country comparable in LEVEL, only in RATE OF CHANGE) | Monthly (AUD genuinely published quarterly by Australia's ABS, not just laggy — confirmed 2026-07-23, median gap between real AUD readings is 92 days vs. ~31 for every other country here; the raw CSV's other two months per quarter are real absence-of-a-print, not a pipeline gap), ranges vary per column (US 1947→2026-06 current; EUR 1996-12→2026-06 current; **JPY 1970→2026-05 current (fixed 2026-07-23, was 2021-06 via FRED - see source table above)**; CAD 1914→2025-03; GBP 1955→2025-03; CHF 1955→2025-04; AUD 1948-07→2025-01; MXN 1969→2024-07) | Recency genuinely varies by country — checked directly, not assumed uniform. US/EUR/JPY are current to within ~2 months; GBP/CAD/CHF/AUD/MXN lag 1-2 years via their FRED mirror (no e-Stat-equivalent direct-source fix attempted for these yet — JPY was the worst offender by far, which is why it got the dedicated fix first). **`data/macro.py`'s `load_cpi()` now forward-fills AUD specifically (`QUARTERLY_CPI_COUNTRIES = {"AUD": 2}`, `ffill(limit=2)` in monthly-index space)** — found live 2026-07-23 that leaving AUD's in-between-quarter months as NaN was silently vetoing most rebalance dates for any `portfolio.book.Book` that included AUDUSD (its joint row-wise `dropna()` drops a date for every asset if even one is missing), via `signals/value.py`'s FX PPP feature. The other 5 lagged-but-monthly countries are deliberately NOT forward-filled here — their gaps are genuine pipeline staleness, not reporting cadence, and fabricating past that would violate this project's "don't fake it, let it gate out" convention (`CLAUDE.md` Rule 4's spirit). See `CLAUDE.md`'s "Mix vs. integrate (Value + XSMOM)" row and `WORKFLOW.md` Phase 7 for the full investigation. |

Four of these five are not wired into `feature_engineering.ipynb`, `volatility.ipynb`,
or any other current notebook - candidate macro-control/regime features for Phase 5+ in
`WORKFLOW.md`, not blocking anything in the near-term signal scope. **VIX is the
exception**: it has a concrete, near-term consumer already (the short-term reversal
signal's Nagel-conditioning check, `references/short_term_reversal_implementation_
recipe.md`), not just a speculative future regime feature like the other four.

---

## 4. Legacy / obsolete — moved to `deprecated/` 2026-07-14

All items below were confirmed unused and moved to `deprecated/Data/` or
`deprecated/Scripts/` (mirroring their original path at the time, before the
2026-07-15 `src/jobs/databento/notebooks/dashboard` reorg — see `cleanup.md`), not
deleted — see `deprecated/README.md`. `Data/` contains only production files.

| File (former location) | Size | Status |
|---|---|---|
| `Data/CRSP_data.csv` | 228MB | Unrelated to the current 38-market futures direction. Confirmed unused. |
| `Data/commodity_prices.csv` | 1.5MB, `(6474, 25)`, columns `Date, WTI, Brent, NatGas, RBOB, HeatingOil, Gold, Silver, Platinum, Palladium, Copper, Corn, Soybeans, Wheat, KC_Wheat, ...` | An earlier, commodity-only (no financial futures) version of what `Data/close.parquet` now contains for the same 24 names. Superseded. |
| `Data/commodity_futures.csv` | 132KB, `(2821, 6)`, columns `Date, CLc1, Cc1, HGc1, LCOc1, NGc1, Wc1` (old-style continuous-contract codes: WTI, Corn, Copper, Brent, Nat Gas, Wheat) | Stale 6-asset dataset, 2014-12-31 → 2025-11-27, used only by the legacy `Time_Series_Models.ipynb`. |
| `Scripts/*.parquet`, `Scripts/metadata.csv` | Duplicates of `Data/` outputs | `Data/` is now unambiguously the single output location. |

---

## 4b. Individual futures contract-month data (term structure) — forward-capture in production

`yfinance` serves individual contract-month prices at `{ROOT}{MONTH_CODE}{YY}.{EXCHANGE}`
(e.g. `CLQ26.NYM` = WTI, August 2026, NYMEX), separately from the continuous `{ROOT}=F`
tickers used everywhere else in this project. This is undocumented/unofficial — not part
of yfinance's public API surface. As of 2026-07-14 this is wired into production, not
just a proof of concept:

**Files (`Data/`):** `term_structure.parquet` — long format, columns `date, asset,
contract_symbol, root, exchange, expiry_code, expiry_year, open, high, low, close,
volume`. `term_structure_manifest.csv` — one row per (run, asset) logging how many
contracts were discovered vs. successfully pulled, for auditing gaps. Produced by
`jobs/capture_term_structure.py`, scheduled daily via Task Scheduler
(`CTA_TermStructureCapture`, 6:15PM, 15 min after `CTA_DailyDataUpdate`).

**Confirmed working (live-tested 2026-07-14):**
- All **38/38 universe assets** resolve, via empirical discovery (probe candidate
  tickers, keep the ones that return data) rather than a hardcoded listing-cycle
  assumption — this naturally handles products with non-uniform listing cycles (e.g.
  Cotton, Live Cattle, quarterly-only financial futures) without needing per-product
  exceptions.
- Multiple simultaneous contract months return **distinct, sensibly-ordered prices**
  (a genuine term structure), confirmed for WTI both in the original research and again
  in the production run (133,553 rows captured across all 38 assets).
- Idempotency confirmed: re-running the capture script the same day produces zero
  duplicate `(date, contract_symbol)` rows.

**Confirmed NOT working — this is the load-bearing limitation, read before using this
data for anything historical:**
- **A contract's ticker stops resolving entirely once it expires.** Tested on 3 WTI
  contracts that expired within the prior ~6 weeks — all returned HTTP 404
  ("possibly delisted"). **There is no way to retroactively backfill an expired
  contract's history.** This data source can only support a forward-capture archive
  (whatever has been captured since the job started running), not historical carry
  backtesting.
- **Multi-year history returned for a still-live contract is not reliable.** The
  pre-2024 portion of `CLQ26.NYM`'s history has `Volume = 0` and flat
  `Open = High = Low = Close` for extended stretches — a synthetic/theoretical settlement
  print, not real trading activity. Filter to `Volume > 0` before treating any row as a
  real price.

**Environment note, still applicable:** the `yfinance` version installed at test time
(0.2.50) was silently broken by a Yahoo API change — even the plain `CL=F` continuous
ticker failed. Upgraded to 1.5.1, which fixed it immediately. **Pin/verify
`yfinance >= 1.5.1`** — this affects `jobs/update_data.py`,
`jobs/capture_term_structure.py`, and both scheduled tasks.

**Historical backfill status (2026-07-20): done — all 42 assets transformed.**
Databento's batch API (`databento/submit_databento_jobs.py` + `retry_databento_jobs.py`)
pulled `ohlcv-1d` + `definition` for the 38-asset universe plus 4 assets added
2026-07-17 (SOFR, Lumber, SwissFranc, MexicanPeso — see the change log below), 76 jobs
originally, all eventually submitted after a rate-limit retry pass. `databento/
transform_databento.py` (`transform_asset(asset, root, exchange)`, one asset at a time,
never bare auto-discovery) has now been run against all 42 assets, staged asset-by-asset
over 2026-07-19/20 with each asset's raw data explored and documented before transforming
— full per-asset evidence trail in `databento/DATA_QUALITY_REPORT.md` (41 write-ups plus
cross-cutting fix sections), summary log in `WORKFLOW.md` Phase 4.

**Output files (`Data/`), all six now populated:**
- `term_structure.parquet` (outrights) — 1,318,677 rows, all 42 assets represented.
  Merge rule: Databento wins on the rare `(date, contract_symbol)` overlap with the
  yfinance forward-capture archive (only possible right at the 2026-07-13/14 boundary);
  forward-capture remains authoritative for every date after that boundary, where
  Databento has no coverage. Exactly 2 non-positive prices in the whole table, both the
  real, historically-documented WTI April-2020 negative settlement (`CLK20.NYM`) — not a
  data artifact, confirmed via web search.
- `term_structure_spreads.parquet` — exchange-quoted 2-leg calendar spreads (same-root)
  and inter-commodity spreads (different roots, e.g. LE-HE, WTI-Brent, Corn-Wheat),
  leg-decomposed (`near_contract_symbol`/`far_contract_symbol`/`near_root`/`far_root`
  etc.). Covers 37 of 42 assets.
- `term_structure_butterflies.parquet` — 3-leg spreads, same leg-decomposed pattern.
  Covers 15 of 42 assets.
- `term_structure_condors.parquet` — 4-leg spreads (found on KC_Wheat, generalized from
  there). Covers 6 of 42 assets.
- `term_structure_averages.parquet` / `term_structure_packs.parquet` — new tables added
  during SOFR's exploration (CME SOFR Average futures and Pack/Bundle spreads, markers
  `:AB`/`:SB`) — generic, not SOFR-specific, so any future asset using the same CME
  markers picks them up automatically. Currently populated only by SOFR (1 of 42 assets
  each).

**Generic bugs found and fixed this rollout, now protecting every asset's transform
automatically** (full evidence in `databento/DATA_QUALITY_REPORT.md`):
- **Exact-zero vs. non-positive filter.** A row missing an OHLC field (left at literal
  `0`) is dropped; a genuine negative print (rare, real — see WTI above) is kept. The
  cruder "any non-positive" version would have silently deleted the real WTI event. Also
  back-ported into `jobs/capture_term_structure.py`'s live yfinance pull, which had the
  same bug (caught via a spurious Cocoa zero-print).
- **Nearest-prior-definition fallback join.** Some instruments stop having their
  `definition` metadata republished shortly before their last real trade (a CME feed
  quirk near expiry) — the join now falls back to the most recent prior definition entry
  for the same `instrument_id` rather than dropping the row.
- **`_write_parquet_atomic()`** — write-to-temp-then-`os.replace()`, working around a
  Windows-only bug where reading a large parquet file then overwriting the same path in
  the same process fails once polars memory-maps it. All 6 output-table writes use this.

**Known gaps, logged not blocking:**
- ICE softs (Coffee, Sugar, Cocoa, Cotton, OJ) only have 2024-07-14→present (2 years),
  not the full 2018-12-23→present available on `IFUS.IMPACT` — a budget trim, not a
  technical limit. Full backfill would cost ~$124 on its own.
- Open interest is still not pulled at all (Databento's `statistics` schema, event-level
  tick data needing real parsing work, estimated ~$50-100 across the universe).
- ICE's spread/butterfly schema (`leg_instrument_id`-based, structurally different from
  CME's) is entirely unsupported — ICE outrights transform fine, combos are
  logged-and-skipped, never attempted.
- **Update 2026-07-20: closed for 3 of 4.** SwissFranc, MexicanPeso, and Lumber are now
  in both `term_structure.parquet` and the core `get_data.ipynb`/`jobs/update_data.py`
  OHLCV pull, plus `jobs/update_dashboard_summary.py`'s `ASSET_CALENDAR` (see section 1).
  **SOFR stays term-structure-only, permanently** — no usable Yahoo continuous ticker
  exists for it, confirmed live, not just unaddressed.
- Local raw Databento zips (~2.6GB, already backed up to Google Drive via
  `databento/archive_to_drive.py`) haven't been deleted from disk — the cleanup step
  exists (`cleanup_local_after_verified_upload()`) but defaults to dry-run and has never
  been invoked.

Full detail, per-asset evidence, and the paid-alternative cost comparison (CME DataMine,
Databento, Norgate, corrected Interactive Brokers pricing, discontinued EIA futures
series) are in `WORKFLOW.md` Phase 4.

---

## 5. Missing data (needed, not yet sourced)

| Data | Needed for | Status |
|---|---|---|
| Soybean Meal (`ZM=F`), Soybean Oil (`ZL=F`) | Soybean Crush RV spread (`WORKFLOW.md` Phase 2d) | Missing — cheap addition to `get_data.ipynb` when that spread is reached |
| Live/streaming price feed | Phase 10 (live data & paper trading) | Not started — planned via Interactive Brokers `ib_insync`, though note IB is not actually free (`WORKFLOW.md` Phase 4/10) |

---

## 6. Overlap with the port congestion project

The Port Congestion Market Signals project (separate repo,
`Projects/Port Congestion Market Signals/`) independently sources WTI, Brent, and the
same 9 Track B commodities (Corn, Soybeans, Wheat, KC Wheat, Copper, Coffee, Sugar,
Cocoa, Cotton) via its own copy of this project's `get_data.ipynb` output — no new price
data is needed on either side for those 11 instruments. That project also live-verified
free fundamentals data (EIA, USDA QuickStats, USDA FAS PSD Online, FRED, CFTC COT, CME
COMEX warehouse stocks) relevant to several of these same commodities — worth checking
that repo's `DATA_SCHEMA.md` before re-sourcing any fundamentals data here from scratch
for any future fundamentals-based signal in this project. Note: EIA's own futures-price
series (once a candidate free source for Phase 4 carry) was discontinued in April 2024 —
see section 4b above for the source that actually unblocked carry instead.

---

## 7. QA dashboard summary artifacts — added 2026-07-15

Source: `jobs/update_dashboard_summary.py`, scheduled daily as `CTA_DashboardSummary`
at 6:25PM, after the other 4 jobs. Does all the actual computation (pipeline-health
roll-up, real-exchange-calendar gap detection via `pandas_market_calendars`,
term-structure curve snapshotting, volatility/macro snapshots) and writes small,
pre-computed artifacts that `dashboard/` only reads and renders — no computation at
dashboard render time. This is a data QA/monitoring tool, not a signal-analysis tool
(see `CLAUDE.md`).

**Files (`Data/dashboard_summary/`):** `pipeline_health.csv` (one row per scheduled
job: last run, status, detail), `ohlcv_coverage.csv` (per-asset date range, missing
sessions vs. that asset's real exchange calendar), `term_structure_curve.parquet` +
`term_structure_summary.csv` (latest forward-curve snapshot per asset, contango/
backwardation), `volatility_snapshot.csv` (current annualized Yang-Zhang vol per asset/
horizon), `macro_latest.csv` (latest point-in-time-correct value per macro source, via
`src/macro_point_in_time.py`'s accessors). `Data/dashboard_summary_manifest.csv` logs
each run (one row per section, same `run_date, section, status, detail` pattern as the
other manifests).

---

## 8. Change log

- Initial version: full inventory of `Data/` verified against actual files (schemas,
  row counts, date ranges) rather than assumed. Flagged the GSCPI cross-project overlap
  with the port congestion project's literature review, and the price-data overlap for
  the 11 shared instruments (WTI, Brent, 9-commodity Track B basket).
- Update: added section 4b documenting the live-verified `yfinance` individual
  contract-month ticker discovery (38/38 universe assets confirmed, free), moved the
  term-structure row out of "missing data" since it's no longer accurately described that
  way, and logged the `yfinance` 0.2.50 → 1.5.1 upgrade needed to keep the data pipeline
  working at all.
- Update (2026-07-14): section 4b rewritten after re-verifying live rather than trusting
  the prior write-up — found expired contracts become permanently unreachable (no
  backfill possible) and multi-year "history" on a live contract is a synthetic print,
  not real data. Documented the new production files (`term_structure.parquet`,
  `term_structure_manifest.csv`) from `Scripts/capture_term_structure.py`, scheduled
  daily via `CTA_TermStructureCapture`. Logged that TurtleTrader's free historical
  dataset was checked and ruled out (frozen at 1999) and Databento identified as the
  likely path for a near-free one-time historical backfill.
- Update (2026-07-14): added a section 1 caveat about Yahoo's continuous futures series
  possibly being a raw, non-back-adjusted front-month splice. First draft of this note
  over-claimed it as a "confirmed" defect based on one coincidental example; a broader
  empirical test (return-jump periodicity + volume-regime check across 6 assets) did not
  support that framing, so it's logged as an unresolved, low-confidence caveat instead —
  a reminder to quantify across multiple data points before declaring a finding
  "confirmed" on the strength of a single example.
- Update (2026-07-15): section 4b updated — Databento historical backfill is in
  progress (credit reinstated, 76 batch jobs submitted for the agreed 33-CME +
  5-ICE-2yr scope). See `WORKFLOW.md` Phase 4 for the full execution log and current
  resumable state.
- Update (2026-07-15): `Scripts/` reorganized into `src/jobs/databento/notebooks/
  dashboard` — all path references in this file updated accordingly (see
  `cleanup.md` for the full rationale). Added section 7 documenting the new QA
  dashboard's summary artifacts (`Data/dashboard_summary/`) and the new
  `Data/volatility_manifest.csv` (section 2) that gives `CTA_VolatilityUpdate`
  parity with the other 3 jobs' logged-run manifests.
- Update (2026-07-20): section 4b rewritten — the Databento historical backfill and
  transform/join stage are done, not in progress. All 42 assets (the original 38 plus
  SOFR/Lumber/SwissFranc/MexicanPeso, added to `jobs/capture_term_structure.py`'s
  `UNIVERSE` 2026-07-17) are now merged into `Data/term_structure.parquet`
  (1,318,677 rows) plus 5 new/expanded combo-instrument tables
  (`_spreads`/`_butterflies`/`_condors`/`_averages`/`_packs`). Verified directly
  against the actual parquet files (row counts, asset counts, non-positive-price count)
  rather than trusted from `WORKFLOW.md`'s narrative — all matched exactly. Logged the
  three generic bug fixes from this rollout (exact-zero filter, nearest-prior-definition
  join fallback, Windows atomic-write helper) and the known remaining gaps (ICE's
  pre-2024 history, open interest, ICE combo schema, the 4 new assets not yet in the
  core `get_data.ipynb` pull or the dashboard's `ASSET_CALENDAR`, local zip cleanup
  deferred). Full per-asset evidence lives in `databento/DATA_QUALITY_REPORT.md`, not
  duplicated here.
- Update (2026-07-20): corrected every place in this file, `CLAUDE.md`, and
  `WORKFLOW.md` that still said "38-asset universe" — the project is a 42-asset
  universe as of the Databento rollout above; 38 is only accurate when specifically
  describing what the core `get_data.ipynb` pipeline covered *before* today. Then
  closed most of that gap for real: added SwissFranc (`6S=F`), MexicanPeso (`6M=F`),
  and Lumber (`LBR=F`) to `get_data.ipynb` and `jobs/update_data.py`, live-tested each
  ticker for real history first (all three confirmed, not assumed), re-ran the pull
  (41/41 assets, 0 failures) and `jobs/update_volatility.py` (picks up new assets
  automatically, pure function of the OHLC parquets), and updated
  `jobs/update_dashboard_summary.py`'s `ASSET_CALENDAR` so the QA dashboard doesn't
  show them as `UNMAPPED`. **SOFR was tested and found permanently unfit for this
  pipeline**: `SR3=F`/`SR1=F` both resolve on Yahoo but return exactly 1 row of
  history — not a real series. It keeps its term-structure/carry coverage (section 4b)
  but will never be in `close.parquet` etc. via this path. Backed up all touched
  `Data/` files to `Data/backups/` before running anything, per this project's
  standing habit.
- Update (2026-07-21): section 1's continuous-curve item rewritten from "not yet
  built" to built — `Data/continuous_futures.parquet`, 170,666 rows across all 42
  assets, raw + back-adjusted. Verified directly (dtype checks, per-asset gap-rate
  scans, spot-checks against `term_structure.parquet`'s own quoted prices), not
  trusted from the algorithm alone — this caught two real bugs (chain-successor
  logic stuck on dead contracts, bad day-1 initialization) before they could reach
  any signal's backtest. Full narrative in `WORKFLOW.md` Phase 0.
- Update (2026-07-21, later same day): while building `research/momentum.py` against
  this curve, found the back-adjustment method itself had a live bug — additive
  ("Panama") pushed 5 assets' adjusted close through zero, corrupting percentage
  returns. Corrected to ratio/proportional (section 1), verified 0 negative
  `adj_close` rows anywhere after rebuilding. Also added `src/data/volatility.py`
  (Yang-Zhang, ported from `volatility.ipynb`) and `src/data/ewma_volatility.py`
  (new) for momentum's vol-estimator comparison — both needed two further,
  real fixes beyond the port itself (a rolling-window `min_periods` too strict for
  the 42-asset panel's routine scattered exchange-calendar gaps; the overnight term
  specifically needing its own, more lenient tolerance than the other two Yang-Zhang
  components) before producing sane coverage. Full bug-by-bug narrative, including
  one wrong hypothesis along the way, in `WORKFLOW.md` Phase 1.
