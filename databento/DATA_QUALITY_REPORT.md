# Databento Transform — Per-Asset Data Quality Findings

Running log, one section per asset, in the order each was explored (`WORKFLOW.md`
Phase 4's "asset-by-asset, not a canary batch" rollout, decided 2026-07-19). Purpose:
trace back exactly what each asset's raw Databento data actually looks like and why
the transform handles it the way it does, rather than relying on memory once all ~42
are done. Shared reference material (OHLCV/definition schema, field by field) lives in
the Appendices at the end rather than being repeated per asset.

---

# Asset 1: LE (Live Cattle) — 2026-07-15

First completed asset from the Databento historical backfill (`WORKFLOW.md` Phase 4).
Explored before writing the transform/join stage, specifically to catch data-shape
surprises before they got baked into pipeline code. Two real findings changed the
transform-stage design (logged in `WORKFLOW.md` Phase 3/4); this report is the
detailed evidence behind those changes, plus everything else checked along the way.

**Source files:** `Data/databento_raw/CME_Globex_MDP3.0_LE_FUT_Definition.zip`,
`CME_Globex_MDP3.0_LE_FUT_OHLCV.zip` — `GLBX.MDP3` dataset, `LE.FUT` parent symbol,
`stype_in="parent"`, 2010-06-06 → 2026-07-13. One CSV (Zstandard-compressed) per
calendar day per schema: 4,050 `ohlcv-1d` files, 5,036 `definition` files.

Tooling: [polars](https://pola.rs) (not pandas — see "Why polars" at the end).

---

## Headline findings

1. **The raw `symbol` field is decade-ambiguous — do not use it as a contract key.**
   CME's raw symbology uses a single-digit year code, which collides across a pull
   this long. Confirmed and fixed (see below); this was the most consequential finding.
2. **A parent-symbol query returns mostly spread/butterfly instruments, not outrights**
   — 79% of rows for this asset. Not a targeting mistake; genuinely how CME organizes
   the instrument universe. The transform-stage design now keeps all three instrument
   classes instead of discarding 79% of the pull (see `WORKFLOW.md` Phase 3/4).
3. **Otherwise clean.** Zero non-positive prices, zero zero-volume outright rows, zero
   OHLC-relationship violations (`high >= low/open/close`, `low <= open/close`), 0.39%
   session coverage gap overall — and even that residual gap is mostly explained by
   calendar-library staleness (Juneteenth) rather than real missing data. One genuine
   5-day gap found and isolated (below).

---

## 1. Symbol decade-ambiguity (critical)

CME's raw symbol format is `{ROOT}{MONTH_CODE}{SINGLE_DIGIT_YEAR}` — e.g. `LEZ5` for a
December contract in a year ending in 5. Over a 16-year pull (2010–2026), that digit is
ambiguous: both December 2015 *and* December 2025 report as `LEZ5`.

Confirmed directly on the highest-volume outright observed:

| `instrument_id` | Date range | Close range | Real contract |
|---|---|---|---|
| `7780` | 2014-07-02 → 2015-12-31 | $118.55 – $159.88 | December 2015 |
| `42041124` | 2024-07-02 → 2025-12-31 | $173.35 – $247.75 | December 2025 |

Both report `symbol == "LEZ5"`. The price ranges alone confirm these are different
contracts (2015 vs. the 2024-2025 record-high cattle market), not a data artifact.
**47 of 60** raw outright symbols observed had exactly 2 distinct `instrument_id`s for
this reason — every contract-month whose 10-years-earlier sibling also falls inside the
pull window.

**Fix, verified clean:** the `definition` schema's `maturity_year` field is an
unambiguous 4-digit integer (confirmed `2015` vs. `2025` for the two `instrument_id`s
above). Joining `ohlcv-1d` → `definition` on `(date, instrument_id)` and building
`contract_symbol = {asset}{month_code}{maturity_year}` instead of trusting the raw
`symbol` string resolves it completely:

- Raw unique symbol strings: **60**
- Unique disambiguated `contract_symbol`s: **107** (= 47×2 + 13×1 — exact match with
  the collision count above)
- `contract_symbol`s with more than one `instrument_id` after the fix: **0**

`instrument_id` itself is *not* the problem — it never gets reused across contract
instances (confirmed: `7780` only ever appears for the 2015 contract's trading life,
`42041124` only for 2025's). It's specifically the symbol string's year digit that's
lossy. Any future use of this data — outright, spread, or butterfly — needs the same
`maturity_year`-based fix; the collision applies equally to spread/butterfly symbols
(`LEV6-LEZ6`, `LE:BF Z6-G7-J7`) since they're built from the same raw month codes.

---

## 2. Parent-symbol queries bundle spreads and butterflies

`instrument_class` (from `definition`, authoritative — not a regex guess on the symbol
string) splits the joined data as:

| Instrument class | Rows | % of total |
|---|---|---|
| Outright futures (`F`) | 35,142 | 21.1% |
| Spread/combo (`S`) — calendar spreads + butterflies | 131,122 | 78.9% |

Splitting `S` further by leg count (butterflies carry a `:BF` marker in their symbol):

| Type | Rows | Unique symbols | Example |
|---|---|---|---|
| Outright | 35,142 | 107 (disambiguated) | `LE` Dec 2026 |
| Calendar spread (2-leg) | 108,996 | 807 | `LEV6-LEZ6` |
| Butterfly (3-leg) | 22,126 | 84 | `LE:BF Z6-G7-J7` |

This happens because `stype_in="parent"` asks Databento for everything CME lists under
the `LE` product — and CME lists exchange-tradeable spread and butterfly instruments
directly, not just outrights, so traders don't have to leg in manually. Pairs
combinatorially outnumber individual months, and triplets outnumber pairs, which is
why spreads+butterflies dwarf outrights by rows even though there are fewer *distinct*
butterfly listings than spread listings per month-set.

**This is not wasted data.** The calendar spread prices are a direct, exchange-traded
quote on `F_near − F_deferred` — the exact quantity the carry formula
(`WORKFLOW.md` Phase 4) needs — without the leg-alignment risk of back-differencing two
separately-sourced outright prices. Sample (`LEV6-LEZ6`, 2015):

```
file_date  symbol      close    volume
20150702   LEV6-LEZ6  -0.500    11
20150708   LEV6-LEZ6  -0.850    2
20150710   LEV6-LEZ6  -0.350    7
```

Small-magnitude, real-looking spread quotes (order of $0.30–0.85/cwt) with genuine
trade volume — usable as-is once the same disambiguation fix is applied to the two
symbol legs. Butterflies are a distinct, currently out-of-scope signal (curve
curvature) — logged as a candidate in `WORKFLOW.md` Phase 3, not built.

**Note for future asset pulls:** Databento almost certainly bills by data volume, and
79% of this pull was unusable for the outright term structure specifically (now usable
for carry, per above, but still worth knowing before repeating this exact query shape
across the other 32 CME assets).

---

## 3. Definition schema has duplicate rows per (date, instrument_id)

The join fans out slightly: 166,264 joined rows vs. 166,093 source `ohlcv-1d` rows
(+171, or +0.10%). Root cause: **1,648** `(date, instrument_id)` pairs in the
`definition` files have more than one row — genuinely identical duplicates, not
conflicting values. Example (`instrument_id=42057126`, 2025-11-16): 3 byte-for-byte
identical rows (`raw_symbol`, `instrument_class`, `maturity_year` all match).

This is consistent with `definition` being an event-sourced schema (each row is a
security-definition *message*, not a deduplicated daily snapshot) — CME appears to
occasionally re-publish an unchanged definition intraday. Harmless for our purposes
(the duplicates carry identical values, so the join isn't corrupted, just slightly
inflated), but the transform script should `.unique()` the definition frame on
`(date, instrument_id)` before joining, rather than silently accept the row inflation.
Self-consistency check: the 36-row difference between this session's polars-with-join
outright count (35,142) and an earlier pandas-with-regex-only count (35,106, no join)
matches this artifact almost exactly.

---

## 4. OHLC / price sanity — clean

On the 35,142 outright rows:

- Non-positive `open/high/low/close`: **0**
- Zero-volume rows: **0** (every outright row has real trade volume)
- Rows violating `high ≥ low/open/close` or `low ≤ open/close`: **0**
- Duplicate `(date, contract_symbol)` rows (post-disambiguation): **36** — fully
  explained by the definition-duplication artifact in section 3, not a real duplicate
  print.
- Price scale confirmed: raw integers are fixed-point, ×1e-9 (Databento's standard DBN
  price scale) — scaled outright closes range $76.60–$258.48, consistent with real
  Live Cattle prices across 2010-2026.

---

## 5. Date coverage

Checked against the real exchange calendar (`pandas_market_calendars`,
`CMEGlobex_Livestock` — the same calendar the QA dashboard's OHLCV Coverage page uses
for the yfinance-sourced panel):

| | Count |
|---|---|
| Expected trading sessions, 2010-06-07 → 2026-07-13 | 4,058 |
| Actual distinct days with outright data | 4,050 |
| Missing (calendar expected, no data) | 16 (0.39%) |
| "Extra" (data present, calendar didn't expect a session) | 8 |

Of the 16 missing:

- **5 are Juneteenth** (2022-06-20, 2023-06-19, 2024-06-19, 2025-06-19, 2026-06-19).
  CME doesn't trade Live Cattle on Juneteenth (a federal holiday since 2021); the
  `CMEGlobex_Livestock` calendar package hasn't been updated to reflect it. **This is a
  gap in our own calendar tooling, not in the Databento data** — a real, concrete
  instance of exactly the caveat already written on the dashboard's OHLCV Coverage page
  ("a calendar mismatch... worth spot-checking outliers rather than treating every
  missing session as a confirmed defect").
- **6 are explained by `condition.json`** reporting `degraded`/`missing` status for
  those dates (Databento's own data-quality flag).
- **5 are a genuine, unexplained gap: 2012-02-06 through 2012-02-10** (a full trading
  week). Verified directly — both `ohlcv-1d` *and* `definition` files are simply absent
  for all 5 dates (not empty; the files don't exist), while 2012-02-03 and 2012-02-13
  immediately bracket it with normal data. `condition.json` claims `available` for
  these dates, so Databento's own metadata doesn't flag this as a known gap. **Worth
  raising with Databento if 2012-era history matters for anything specific** — flagged
  here rather than silently accepted, per this project's own standing rule about
  labeling gaps honestly instead of patching around them.

The 8 "extra" dates (data present on days the calendar didn't expect) are consistent
with actual abbreviated holiday sessions CME held that the generic calendar package
marks as fully closed (e.g. 2010-07-05, 2010-11-25 Thanksgiving, 2011-01-17 MLK Day) —
another instance of the calendar package being the imprecise side, not the data.

---

## 6. Why polars (not pandas) for this exploration

Measured, this session, reading + joining all 9,086 files (~948k total rows across both
schemas):

| Step | Time |
|---|---|
| Read 4,050 OHLCV files (166,093 rows) | 2.5s |
| Read 5,036 definition files, 8 of 65 columns selected (782,482 rows) | 29.5s |
| Join on `(date, instrument_id)` | 0.97s |
| **Total, including all checks in this report** | **35.1s** |

This was a single-asset, ~200MB exploration — not a scale where pandas would have
struggled. The reason to do it in polars anyway: this exploration is the prototype for
the real transform script, which has to run this same join+filter logic across the
**full 33-CME-asset backfill** (roughly 30-40x this volume — likely tens of millions of
definition rows alone). Polars' multi-threaded CSV parsing and column-selection-on-read
(only 8 of 65 definition columns were ever materialized) are what pay off at that scale,
not at this one. Scoped to `databento/` only — the rest of the repo stays pandas
(`cleanup.md`).

**Real head-to-head measurement (2026-07-19), not just estimated.** The paragraph
above justified polars at design time without ever running the pandas equivalent
side by side. After 16 assets actually processed this session, benchmarked both
directly on the identical task — read all OHLCV + definition files for one asset,
select the same columns, join on `(date, instrument_id)` — using Copper (`HG`, 4,997
OHLCV + 5,042 definition files, ~172k joined rows, one of the larger single-asset
pulls done this session):

| Approach | Time | Result shape |
|---|---|---|
| polars (`scan_csv`, lazy, multi-file, column projection) | **7.00s** | `(171869, 11)` |
| pandas (per-file `read_csv` loop + `pd.concat`, the direct equivalent) | 47.80s | `(171869, 11)` |

**6.8x speedup, identical output shape** (confirms the comparison is apples-to-apples
on correctness, not just speed — both approaches used `usecols`/`.select()` to read
only the needed columns, so the gap is not just "polars skips more columns," it's the
multi-threaded multi-file scan itself vs. pandas' inherently single-threaded
per-file-then-concat loop).

**Back-of-envelope projection across the full 42-asset universe (2026-07-19).**
Converted the Copper measurement into a per-file rate (`time ÷ 10,039 files`) and
applied it to the real total file count across all 84 downloaded zips — counted
directly via `zipfile.namelist()`, not estimated: **349,977 files, 2.64GB raw**,
across all 42 assets × 2 schemas.

| | Rate per file | Projected total (349,977 files) |
|---|---|---|
| polars | 0.697 ms | **244s ≈ 4.1 minutes** |
| pandas | 4.761 ms | 1,666s ≈ 27.8 minutes |
| **Projected time saved** | | **≈ 23.7 minutes**, one full pass over the raw archive |

Caveat, stated plainly: this is a linear extrapolation from a single asset's
measured rate, not a re-run of all 42 — file-count-to-row-density varies somewhat
by asset (grains/livestock list far more spread/butterfly combos per file than
metals/rates, per every per-asset finding above), so the true number could plausibly
land anywhere in the 15-30 minute band, not exactly 23.7. The order of magnitude
(tens of minutes saved on one pass, ~7x per-file) is the load-bearing claim, not the
decimal.

**Other benefits beyond raw speed** — split explicitly into what this session
actually observed vs. general, architectural properties not independently
re-measured here:

*Observed this session:*
- **Correctness held throughout a genuinely complex pipeline.** 16 assets, several
  requiring new resolution logic built mid-session (condor spreads, the anchor-leg
  algorithm's generalization) — validated at full scale every time (e.g. KC_Wheat's
  24/24 condors cross-checked against its own real outright universe with 0
  mismatches) using polars' expression API (`pl.when/then`, `.str.contains`,
  `.group_by`) for the classification and validation logic itself, not just the file
  I/O. The same lazy-scan + `.select()` pattern that gave the 6.8x also made ad hoc
  diagnostic queries (e.g. "how many `_Z`-suffixed instrument_ids does Coffee have")
  fast enough to run interactively while diagnosing an anomaly, not as a separate
  batch step.
- **One code path scales from a single day's diagnostic peek to the full 16-year
  pull.** The exact same `scan_definition`/`scan_ohlcv` functions were used for a
  single-file peek (checking ZN's fractional-price fields) and for the full
  5,042-file Copper pull — no separate "small data" vs. "big data" code path needed.

*General/architectural, not independently benchmarked this session:*
- **Apache Arrow columnar backend + Rust core** — the structural reason polars
  parallelizes CSV parsing across files/threads without extra code, and avoids
  pandas' per-column Python-object overhead for string-heavy columns like
  `raw_symbol`.
- **Lazy evaluation with query optimization** — `scan_csv(...).select(...).join(...)`
  builds a query plan and only executes on `.collect()`, letting polars fuse and
  reorder steps (e.g. pushing column selection down into the CSV parse itself)
  rather than materializing an intermediate DataFrame at every step the way eager
  pandas code does.
- **Larger-than-memory scaling headroom** — the same lazy-scan pattern extends to
  streaming/out-of-core execution if a future pull exceeds available RAM, without a
  rewrite; not a constraint this project has hit yet (2.64GB total raw), but relevant
  given the roadmap (Databento definition schema alone will be tens of millions of
  rows across the full 33-CME-asset backfill, per the original design note above).

---

## Recommendations for the transform script (`databento/`, not yet built)

1. Join `ohlcv-1d` → `definition` on `(date, instrument_id)`, not a static
   `instrument_id → symbol` map — `definition` must be deduplicated on that same key
   first (section 3).
2. Derive `contract_symbol` from `maturity_year`/`maturity_month`, never from the raw
   `symbol` string (section 1) — applies to outrights, spreads, and butterflies alike.
3. Write three outputs, not one, all keyed by the same disambiguated contract
   identity (`WORKFLOW.md` Phase 3/4):
   - `term_structure.parquet` — outrights
   - `term_structure_spreads.parquet` — calendar spreads
   - `term_structure_butterflies.parquet` — butterflies
4. Don't assume `condition.json` catches every gap — the Feb-2012 gap (section 5) shows
   Databento's own metadata can miss a real hole. A basic post-transform coverage check
   against a real exchange calendar (the same pattern the QA dashboard already uses) is
   worth running per asset, not just for LE.
5. Prefer calendar-spread prices over back-differenced outrights for carry
   construction once `term_structure_spreads.parquet` exists (section 2).

---

---

# Asset 2: KC (KC_Wheat) — 2026-07-19

CBOT-cleared (`GLBX.MDP3`, same dataset as LE). First asset run through the built
`transform_asset()` outside of LE, per the revised asset-by-asset rollout
(`WORKFLOW.md` Phase 4). Result: `status=OK`, 24,431 outright rows, 87,566 spread
rows, 11,418 butterfly rows, 92 condor rows (new, below), 0 join-unmatched, 0
non-positive prices, 0 OHLC violations, 8 missing sessions (0.25%).

## Headline finding: CME condor spreads (4-leg), not a data-quality bug

A prior (reverted) run against this asset hit `spread_not_2_legs=24` — 24 unique
`instrument_class="S"` instrument_ids whose `raw_symbol` didn't split into exactly 2
dash-separated legs the way every LE calendar spread did. Never investigated at the
time (the run was rolled back before looking into it); diagnosed properly this
session before touching any more assets.

**Root cause:** these are genuine CME **condor spreads** — a 4-leg exchange-listed
combo instrument, alongside the already-known 2-leg calendar spread and 3-leg
butterfly. Example: `KE:CF H4K4N4U4` (Mar14/May14/Jul14/Sep14 KC Wheat). Format is
`{ROOT}:CF {leg1}{leg2}{leg3}{leg4}` — legs concatenated with no separator (unlike
the butterfly's hyphen-separated `{ROOT}:BF {leg1}-{leg2}-{leg3}`), each leg a
2-character month-code+single-digit-year code. `resolve_spread_legs`'s dash-split
correctly recognized these as not-a-2-leg-spread and logged+skipped them — the same
"log and skip, don't guess" behavior already used for ICE's unsupported schema, i.e.
the code was working as designed, not corrupting anything.

**Materiality:** small — 92 total rows across 24 instruments over the full
2014-2026 pull (all with real, if thin, volume; avg ~4 trading days per instrument
over 12 years), vs. this same asset's 11,418 real butterfly rows across 83
instruments.

**Decision (direct instruction): build condor support rather than leave it as a
permanent skip**, since the fix is generic (`resolve_condor_legs`, in
`transform_databento.py`) — it applies to every future CME asset run through
`transform_asset()`, not just KC_Wheat, so the one-time cost compounds into reusable
infra. Generalizes the same anchor-leg-plus-modular-offset algorithm already
validated for spreads/butterflies to 4 legs. **Validated at full scale: 24/24
condors resolved, 0 errors, every leg cross-checked against KC_Wheat's own real,
independently-disambiguated outright universe (72 contract-months) with 0
mismatches** — same validation bar LE's spreads/butterflies were held to. New output
table: `Data/term_structure_condors.parquet`, `near`/`mid1`/`mid2`/`far`
leg-decomposed (mirrors the butterfly schema's near/mid/far, extended to 4 legs).
Single-root assumption carried over from butterflies (0 cross-root condors observed
here, same as LE's butterflies) — flagged, not proven, for other assets.

---

# Asset 3: KC (Coffee) — 2026-07-19

**First ICE-cleared asset (`IFUS.IMPACT`) run through the transform** — a
structurally different dataset from CME/`GLBX.MDP3`. Outrights were assumed (per the
original module docstring) to "transform identically for both... the fields they
need are common to both schemas." **That assumption was wrong** — this asset is the
first real test of it, and it failed in three distinct, previously-undocumented ways.
All three are now fixed generically in `transform_databento.py` (two apply to every
asset, one is ICE-specific) and validated before merging. Final result:
`status=PARTIAL` (expected — the three fixes below all log as anomalies rather than
silently disappearing), 6,877 outright rows, 0 non-positive prices, 0 OHLC
violations, 14 missing sessions (2.7% — wider than LE/KC_Wheat's CME sessions,
plausibly ICE's own holiday calendar differing from `pandas_market_calendars`'
`CMEGlobex_Livestock`-style mapping, not independently confirmed this session).

## Finding 1: Databento's `UNDEF_PRICE` sentinel leaking into `ohlcv-1d` OHLC fields

261 rows (across all instrument classes, this asset) had `int64::MAX`
(`9223372036854775807`) directly in `open`/`high`/`low`/`close` — Databento's own DBN
"field not applicable" sentinel (documented for the `definition` schema in Appendix
B below, but not previously observed inside `ohlcv-1d` price fields themselves). The
existing transform had no guard for this: dividing it by 1e9 produces a *positive*,
finite-looking number (~9.2 billion) that a plain `<= 0` positivity check can't
catch — it would have silently written a nonsense price into `term_structure.parquet`
undetected. **Fix:** filter on the raw, pre-scaling integer value, applied to every
instrument class (a sentinel is never valid data regardless of outright vs.
spread/butterfly/condor) — `UNDEF_PRICE` constant + a filter immediately after the
join, before the `/1e9` rescale.

## Finding 2: a second, non-representative instrument per contract month (ICE-only)

Each real Coffee contract month has **two** `instrument_id`s, both tagged
`instrument_class="F"` (so `instrument_class` alone can't distinguish them, unlike
CME) — one `raw_symbol` ending in `!` (e.g. `KC  FMU0024!`, the genuine tradable
outright: 36 instrument_ids, 8,981 daily rows) and one ending in `_Z` (e.g.
`KC  FMU0024_Z`: 13 instrument_ids, 1,780 rows). **97% of the `_Z` variant's rows
(1,731/1,780) are non-positive or the sentinel from Finding 1** — confirmed not a
calendar spread (that's the separate dash-combo pattern, e.g.
`KC FMU0024_Z-KC FMZ0024_Z`, already out of scope for ICE spreads/butterflies) but a
distinct third thing whose true meaning (settlement-only record? post-expiry
delivery/cabinet bookkeeping?) is **not confirmed** — flagged rather than guessed,
consistent with this project's standing rule. Actionable regardless of the exact
semantics: it is clearly not the tradable outright, so it's excluded from the
outright table by `raw_symbol` suffix. ICE-specific fix (`exchange ==
ICE_EXCHANGE_SUFFIX` gate) — ruled out as a CME concern since LE's own validation
found 0 non-positive rows with no such suffix pattern.

## Finding 3: incomplete daily bars even on the genuine outright instrument

Of the 8,981 genuine (`!`-suffix) rows, **1,890 (21%) have `open`/`high` populated
with real, sensible coffee prices (e.g. 245.05, 251.75 ¢/lb) but `low` and/or `close`
exactly `0`.** Volume on these rows is real (16-739 contracts observed) — not a
no-trading day — so this looks like ICE's daily-bar aggregation genuinely not
publishing a complete 4-field print for thinner back-months, a real gap in the
upstream feed rather than a join/scaling bug on our side. Left in, a `close=0` would
register as a fabricated ~-100% single-day return in any consumer (vol, momentum,
carry) touching this asset's close series — worse than dropping the row outright,
per `CLAUDE.md` Rule 4's "label missing as missing, don't fake it" principle. **Fix:
drop any outright row where `open`/`high`/`low`/`close` isn't all strictly positive**
— scoped to outrights only, not spreads/butterflies/condors, since those
legitimately trade negative (LE's own calendar spreads closed at -0.50/-0.85 in
Section 2 above; a positivity filter there would wrongly discard real data).

## Verified negative-price hypothesis was checked, not assumed

Before concluding Finding 3's near-zero/negative values were an artifact rather than
a real market event (cf. WTI's genuine -$37.63 print in April 2020), confirmed via
web search that coffee futures have no documented history of negative or near-zero
prints — coffee lacks crude oil's acute physical storage-constraint dynamic, and the
52-week real trading range (~239-438 ¢/lb) is nowhere near the ~0 values found here.

## Not yet investigated: OJ (OrangeJuice) has 3 pre-existing non-positive rows

Found incidentally while verifying Coffee's merge didn't disturb other assets: 3 rows
for `OrangeJuice` (`open=high=low=close=0.0`, 2025-09-30, three different contract
months) already existed in the yfinance forward-capture archive **before this
session's Databento work** (confirmed against the earliest backup taken this
session). Unrelated to Coffee — OrangeJuice hasn't been run through
`transform_asset()` yet. Flagged here so it isn't forgotten; likely resolves on its
own once OrangeJuice gets its own Databento pass (Databento wins on overlap), but
worth checking `jobs/capture_term_structure.py`'s OJ handling specifically if it
doesn't.

## ICE spread/butterfly/condor rows: still correctly out of scope

143 `instrument_class="S"` instrument_ids logged and skipped
(`ICE_schema_unsupported_spreads=143`) — ICE's `definition` schema uses
`leg_instrument_id`/`leg_raw_symbol` fields structurally different from CME's
embedded-symbol-string shape (module docstring); the CME-validated anchor-leg
algorithm was never applied to them. Still unvalidated for ICE, not attempted this
session — the outright fixes above didn't touch this.

---

# Asset 4: SB (Sugar) — 2026-07-19

Second ICE-cleared asset, run specifically to test whether Coffee's three findings
(Asset 3) were Coffee-specific or general to `IFUS.IMPACT`. **Confirmed general** —
same shape, similar magnitude, no new anomaly types:

| | Coffee | Sugar |
|---|---|---|
| Sentinel price rows dropped | 261 | 319 |
| `_Z`-variant outright instrument_ids excluded | 13 | 14 |
| Incomplete-print outright rows dropped | 1,890 | 1,871 |
| ICE unsupported spread instrument_ids | 143 | 148 |
| Non-positive prices post-fix | 0 | 0 |

Merged clean (`status=PARTIAL`, expected, same as Coffee — the three fixes log as
anomalies rather than disappearing). 8,009 outright rows. No new investigation needed;
the generic fixes built for Coffee handled this asset unchanged. Reasonable prior now:
Cocoa/Cotton/OJ likely share this shape too, but not confirmed until each is actually
run — per this project's own standing rule against assuming a pattern holds without
checking (`CLAUDE.md` Rules 1/2's spirit, applied here to a data characteristic rather
than a statistical test).

---

# Asset 5: CT (Cotton) — 2026-07-19

Fourth ICE-cleared asset. Same shape as Coffee/Sugar, no new findings: 534 sentinel
rows dropped, 14 `_Z`-variant instrument_ids excluded, 1,077 incomplete-print rows
dropped, 133 unsupported spread instrument_ids logged/skipped. Merged clean
(`status=PARTIAL`, expected), 5,289 outright rows, 0 non-positive prices post-fix.

---

# Asset 6: OJ (OrangeJuice) — 2026-07-19

Fifth and last ICE-cleared asset. Same shape again: 316 sentinel rows dropped, 13
`_Z`-variant instrument_ids excluded, 322 incomplete-print rows dropped, 64
unsupported spread instrument_ids logged/skipped. Merged clean, 2,546 outright rows.

## Resolved: the pre-existing 3 non-positive rows flagged under Coffee (Asset 3)

Asset 3 flagged 3 non-positive `OrangeJuice` rows (`open=high=low=close=0.0`,
2025-09-30, contracts `OJH27`/`OJU26`/`OJX26`) already present in the yfinance
forward-capture archive before any Databento work, deferred pending OJ's own pass.
**Root cause now confirmed, not fixed by the Databento merge — correctly so.**
Checked Databento's raw `ifus-impact-20250930.ohlcv-1d.csv.zst` directly: it has no
outright row at all for those three specific far-dated contracts that day (only
`FMX0025!`/`FMF0026!`/`FMH0026!`/`FMK0026!` printed) — a genuine no-trade day for
those thin, far-out months in the real exchange feed, not a Databento gap. Since
there's no Databento `(date, contract_symbol)` row to win the overlap, the spurious
yfinance-sourced zero rows pass through unmodified.

**This is a `jobs/capture_term_structure.py` (yfinance daily forward-capture) bug,
not a Databento-transform issue** — that job apparently wrote a `0.0` print for a
contract with no real trade that day, rather than skipping it, most plausibly a
consequence of yfinance's undocumented individual-contract-ticker behavior
(`WORKFLOW.md` Phase 4 already flags this path as "unofficial... riding on Yahoo's
internal API"). Out of scope to fix in this pass (belongs to the yfinance job, not
`transform_databento.py`) — logged here so it isn't rediscovered as a mystery later.
Materiality is negligible (3 rows out of 215,605 total, one calendar date, contracts
so far-dated they had no real trading anyway).

## All 5 ICE softs now done — pattern fully confirmed, no per-asset surprises left

| Asset | Sentinel dropped | `_Z`-variant excluded | Incomplete-print dropped | ICE spreads skipped | Outright rows |
|---|---|---|---|---|---|
| Coffee | 261 | 13 | 1,890 | 143 | 6,877 |
| Sugar | 319 | 14 | 1,871 | 148 | 8,009 |
| Cocoa | 481 | 13 | 1,226 | 114 | 5,315 |
| Cotton | 534 | 14 | 1,077 | 133 | 5,289 |
| OrangeJuice | 316 | 13 | 322 | 64 | 2,546 |

Consistent `_Z`-variant count (13-14 across all five) strongly suggests it's tied to
a fixed number of currently/recently-listed contract months per product rather than
random data corruption — consistent with the "second instrument per real contract
month" theory from Asset 3, not yet independently confirmed beyond that. ICE spread
support (the `leg_instrument_id`-based schema) remains unbuilt for all 5 — still
correctly out of scope, not attempted.

---

# Asset 7: GC (Gold) — 2026-07-19

First non-agricultural CME asset (metals category) — a deliberate category jump from
LE/KC_Wheat (both CBOT/CME agricultural) to check for category-specific idiosyncrasies
before assuming the transform generalizes cleanly across all of CME. Result:
`status=OK`, 0 anomalies of any kind. 52,398 outright rows, 100,425 spread rows, **0
butterflies, 0 condors**.

**Process note:** this run was executed without a pre-run backup, breaking the
discipline followed for every other asset so far. Confirmed safe after the fact by
exact arithmetic reconciliation (post-run total minus last-known-clean total equals
exactly Gold's own row delta, 51,124 = 51,124 — no other asset moved) rather than
assumed — but flagging the lapse plainly rather than quietly correcting it, and a
proper backup was taken immediately after for the next asset.

## Checked, not assumed: 0 butterflies/condors is a real COMEX fact, not a missed marker

Zero butterflies and zero condors is exactly the kind of silent-miss risk this
project's own hard rules warn against (assuming a pattern generalizes without
checking). Verified directly against the raw `definition` data rather than trusted the
zero counts at face value: **all 12,638 unique `instrument_class="S"` raw_symbols for
Gold are plain single-dash 2-leg strings — zero contain `:BF` or `:CF`, zero have any
other shape.** COMEX Gold genuinely doesn't list exchange-traded butterfly or condor
combos (at least not under this naming convention); the transform isn't missing a
marker format, there's nothing to miss.

## Other checks, all clean

- **No inter-commodity spreads** (`near_root`/`far_root` both `GC` for all 100,425
  spread rows) — unlike LE, which had real Live-Cattle/Lean-Hogs cross-commodity
  pairs (~35% of its spreads). Gold apparently doesn't have an actively-quoted
  cross-commodity combo partner in this dataset; not assumed to generalize to the
  other 4 metals without checking each.
- **Outright price levels sane**: $1,050.40-$5,957.70/oz across the full pull,
  consistent with gold's real historical range through the current bull market — no
  scaling or sentinel-leakage red flags like Coffee's Finding 1.
- 0 non-positive prices, 0 OHLC violations, 6 missing sessions (0.14% — tighter than
  LE's 0.39%, plausible given COMEX metals trade later local holidays differently
  than CBOT livestock, not independently confirmed).

---

# Assets 8-10: SI (Silver), PL (Platinum), PA (Palladium) — 2026-07-19

Same category, checked individually rather than assumed from Gold (Gold's own 0
inter-commodity-spread result was explicitly flagged as not-yet-generalized). All
three: `status=OK`, 0 anomalies, 0 butterflies, 0 condors, 0 cross-root spreads,
sane price levels (Silver $11.85-$127.10/oz; Platinum $595.10-$2,806.50/oz; Palladium
$423.00-$3,289.00/oz — all consistent with real historical ranges through the current
metals bull market). Confirms metals as a category is clean and structurally simple
relative to the ICE softs and CBOT grains explored so far — no per-asset surprises
across 4 of 5 metals.

# Asset 11: HG (Copper) — 2026-07-19, BLOCKED (not a transform issue)

**`CME_Globex_MDP3.0_HG_FUT_Definition.zip` is corrupted — truncated, missing its
End-Of-Central-Directory record.** `zipfile.ZipFile` fails with `BadZipFile: File is
not a zip file`; `file`'s magic-byte check alone reports "Zip archive data" (checks
only the header, not the tail), which is why this wasn't caught until actually
opening it. The OHLCV zip is unaffected (5,000 files, opens cleanly) — this is
isolated to one of the two required files.

**Confirmed reproducible, not a one-off network glitch:** re-downloaded the file:
identical size (135,376,146 bytes) and identical truncation point (byte-for-byte tail
match, cut off mid-way through the same central-directory entry,
`glbx-mdp3-20260626.definition.csv.zst`). Getting the exact same corruption twice
rules out a random interrupted transfer.

**Also confirmed via the Databento API directly, not assumed:** the underlying batch
job (`GLBX-20260715-5858LNQKED`) shows `state=done`, not expired
(`ts_expiration=2026-08-14`, well past today) — the job itself completed
successfully. `client.batch.list_files()` shows Databento serves this job as
**individual per-day files** (`glbx-mdp3-20260606.definition.csv.zst`, etc.), never
as a single zip. **No script in this repo bundles those into a `.zip`** — meaning the
local `CME_Globex_MDP3.0_HG_FUT_Definition.zip` file came from a portal download or a
manual step outside the tracked pipeline (`databento/submit_databento_jobs.py`,
`databento/retry_databento_jobs.py`), and the truncation is happening somewhere in
that untracked path, not in anything `transform_databento.py` or this project's own
scripts control.

**Status: blocked, logged to the manifest (`asset=Copper, status=BLOCKED`), not
worked around.** Per direct instruction, following up with Databento support
directly rather than guessing at a fix on this end. No merge attempted — the
extraction step fails before any write, so `term_structure.parquet` is untouched by
this asset.

---

# Asset 12: ZN (US_10Y) — 2026-07-19

First rates asset — a deliberate category jump to test fixed-income structure before
assuming it. Verified three real differences against the raw data *before* running
the transform, per direct instruction:

1. **Fractional price-display fields are genuinely populated** (`main_fraction=32`,
   `sub_fraction=2`, `price_display_format=3`, vs. sentinel/null for LE) — confirms
   Treasury futures display in half-32nds, the classic bond convention. **Doesn't
   require any code change**: the raw OHLC price fields are still stored as plain
   fixed-point decimals (÷1e9) regardless of display convention —
   `min_price_increment=15625000` → `0.015625` = exactly 1/64 point, the real tick.
   The transform never reads `main_fraction`/`price_display_format`, only
   `maturity_year`/`month`/`instrument_class`/raw OHLC, so this difference is real
   but inert for our purposes.
2. **Quarterly-only listing cycle** (only `H`/`M`/`U`/`Z` months ever list — 3
   outright contracts visible on any given date, vs. 12+ for Gold/LE). Real economic
   fact, not a gap — but it caps the combinatorial spread/butterfly count per date
   much lower than agriculturals/metals.
3. **Contract size correctly captured** via `unit_of_measure_qty` → $100,000 face
   value (the real 10-Year Note contract size), same field LE used (40,000 lbs), not
   `contract_multiplier` (sentinel for both).

Result: `status=PARTIAL`, 10,503 outright rows, 4,955 spread rows, **0 butterfly
rows, 0 condor rows**. Two anomalies found, both investigated and confirmed
legitimate, not bugs:

## Finding 1: 0 butterfly rows is real zero trading volume, not a parser miss

Unlike Gold (genuinely 0 butterfly *instruments* exist), ZN's `definition` file does
list real butterfly instruments (e.g. `ZN:BF U6-Z6-H7`, confirmed present on
2026-07-13) — but checked directly whether that specific `instrument_id` (42063318)
ever appears in `ohlcv-1d` across the **entire** 2010-2026 pull: **zero rows.** CME
lists rate butterfly combos (for margin/quoting purposes), but they see essentially
no executed volume — a genuine, if surprising, market-structure fact. Institutional
curve trades in rates are apparently done via calendar spreads (real, liquid — 4,955
rows) rather than the 3-leg butterfly combo, unlike KC_Wheat's condors (92 real
trades) or LE's butterflies (22,102 real rows).

## Finding 2: `spread_not_2_legs=957` is CME's User-Defined Spread facility, not a bug

Raw symbols like `UD:ZN: TL 0825829457` — 2,568 unique such instrument_ids exist in
the `definition` file across the pull (957 of them produced an actual OHLCV row).
This is CME's **User-Defined Spread (UDS)** facility: participant-negotiated, bespoke
combo instruments identified by an internal reference number, not a standard
exchange-listed calendar spread. **Checked, not assumed, whether the legs are
recoverable another way**: CME's `definition` schema for `GLBX.MDP3` genuinely does
**not** have `leg_count`/`leg_instrument_id`/`leg_raw_symbol` fields at all —
confirmed by a real `ColumnNotFoundError` attempting to select them. Those fields are
specific to ICE's `IFUS.IMPACT` schema (module docstring), not CME's — a different
finding from KC_Wheat's condors, which *were* resolvable from the existing symbol
string. There is no data-driven way to decompose a UDS instrument's legs from what
this pull provides. Correctly logged and skipped, same "don't guess" principle as
ICE's unsupported spread schema — not a fix candidate, a permanent limitation of
this data source for this instrument type.

## Finding 3: `spread_unknown_root_exchange=31` is a real inter-commodity spread partner outside our universe

Raw symbols like `ZNU2-N1UU2` — `N1U` is CME's Ultra 10-Year Note futures, a real,
separate CME product not currently in the 42-asset `UNIVERSE`. Same pattern as LE's
real LE-HE cross-commodity spreads, except here the partner root isn't tracked at
all, so `_leg_symbol` correctly returns `None` and the row is logged/skipped rather
than guessed at. Not a bug — would only resolve if Ultra 10-Year Note were added to
`UNIVERSE`, which is a universe-composition decision, not a transform fix.

---

# Asset 11 update: HG (Copper) — RECOVERED (2026-07-19)

Root cause found: `client.batch.download()`'s default path fetches a server-side
zip bundle (`_download_batch_zip`), which returned `504 The remote gateway timed
out` when attempted directly on this job — very likely the same failure that
produced the original truncated zip (a partial response saved before the gateway
killed the connection). **Fix: bypass the zip-bundle endpoint entirely** using the
SDK's `filename_to_download` parameter to fetch each of the job's 5,042 individual
`*.definition.csv.zst` files directly (confirmed via `client.batch.list_files()`
that Databento serves this job as individual files, never a zip, so nothing about
the underlying data required a zip in the first place). All 5,042 files downloaded
and **SHA256-verified against Databento's own reported hash**, 0 failures, $0 cost
(re-fetching an already-completed, unexpired job's output, not a new query). Built a
fresh zip from the verified files (`zipfile.ZIP_STORED`, matching the original's
compression method), confirmed `testzip()` reports no bad entries, and swapped it in
under the expected filename — the original corrupted zip preserved alongside as
`..._CORRUPTED_2026-07-19.zip.bak`, not deleted. Re-ran the transform: `status=OK`,
0 anomalies, 56,728 outright rows, 115,141 spread rows, 0 butterflies/condors/
cross-root spreads — same clean shape as the other 4 metals. Price range
$1.9365-$7.0815/lb, consistent with copper's real historical range.

---

# Assets 13-16: UB (UltraBond), ZF (US_5Y), ZT (US_2Y), ZB (US_30Y) — 2026-07-19

Remaining CME Treasury-complex assets, checked individually rather than assumed from
ZN. All confirm the exact same shape as US_10Y — no new anomaly types:

| Asset | Outright rows | Spread rows | Butterflies | `not_2_legs` (all `UD:`) | Unknown-root spreads |
|---|---|---|---|---|---|
| UltraBond (UB) | 7,695 | 2,939 | 0 (none even listed) | 226 | 19 (`B1U`, `MWN`) |
| US_5Y (ZF) | 9,177 | 4,233 | 0 | 1,239 | 28 (`F1U`) |
| US_2Y (ZT) | 9,158 | 4,153 | 0 | 610 | 0 |
| US_30Y (ZB) | 10,614 | 4,859 | 0 | 236 | 0 |

Each confirmed, not assumed: `not_2_legs` instrument_ids are 100% `UD:` User-Defined
Spreads (verified per-asset, not extrapolated from ZN), and every unknown-root
partner is a real, separate CME product outside the 42-asset universe (Ultra
10-Year-adjacent tickers, not typos or parsing failures) — `US_2Y`/`US_30Y` have none
at all, unlike the other three. All merged `status=PARTIAL` (expected), 0
non-positive prices, 0 OHLC violations. **All 5 CME Treasury-complex assets now
done** (ZN, UB, ZF, ZT, ZB) — rates as a category is fully explored, structurally
consistent across all five: quarterly-only listing, populated-but-inert fractional
price fields, zero-to-negligible butterfly volume, and the UDS/unknown-root spread
pattern as the only recurring "anomaly," always legitimate.

---

# Asset 17: CL (WTI Crude) — 2026-07-19, energy category, most consequential asset this session

First energy asset. Produced a real correctness bug caught before it could spread
to other assets, plus two genuinely new, economically important spread grammars
worth building support for (not skip-and-log) given their direct overlap with
`WORKFLOW.md` Phase 2d's highest-priority candidates.

## Finding 1 (critical, caught before merging): the ICE-derived "incomplete price"
## filter would have silently discarded the real April 2020 negative-price event

The generic outright filter built for Coffee's Finding 3 (drop rows where any OHLC
field is non-positive) flagged 2 WTI outright rows for `CLK20.NYM` (May 2020) on
2020-04-20/21. **Checked before accepting the drop, not assumed safe**: these are
the actual, historically documented WTI negative-settlement event -
`low=-40.32, close=-2.67` on 2020-04-20, continuing `open=-3.01, low=-10.0,
close=9.06` on 2020-04-21 - confirmed via web search that this is real (WTI's own
well-known -$37.63 settlement print, the first negative futures settlement in its
history) and that coffee/most other commodities have no comparable precedent
(Asset 3's negative-price check). **Root cause of the near-miss:** the filter used
"any field `<= 0`" as its incomplete-bar signature, conflating two different things
- ICE's real bug (some field(s) left at exactly `0` while others are populated) and
a genuine negative price (no field is ever exactly zero in either WTI row - checked
directly). **Fix:** narrowed the filter to check for an exact `0` in any field,
not "non-positive" - this preserves genuine negative prints while still catching
the original ICE pattern. **Verified the fix doesn't retroactively break the 5
already-merged ICE assets**: re-ran the old vs. new filter side by side against all
five - `old_caught == new_caught` exactly, 0 negative-but-nonzero rows in any of
them, so every previously-dropped ICE row really was an exact-zero case. No
re-merge needed there; this was a purely forward-looking fix that happened to
matter immediately for the very next asset.

## Finding 2: WTI's far-dated curve needs 2-digit year codes, and it's a real CME
## convention, not a data artifact

WTI lists contracts out past 2036 - far enough that CME's own single-digit year
code becomes ambiguous even within one pull, so far-dated legs (and, rarely, whole
outrights) use a literal 2-digit year suffix instead (e.g. `CLU36` = Sept 2036,
confirmed directly against real outright symbols: `maturity_year % 100` matches
exactly). This is unambiguous, not a second decade-collision problem to solve -
confirmed by checking that our own pull window never needs interpretation across
centuries. **Fix, generalized across spreads and butterflies** (found 2 real
2-digit butterfly legs too, not just spreads): `_SPREAD_LEG_RE`/`_BF_LEG_RE` now
accept 1-or-2-digit trailing codes; a shared `_resolve_leg_year()` treats a 2-digit
code as an absolute year and a 1-digit code via the existing anchor-relative
modular offset, unchanged. 26 instruments (~2,728 rows) affected.

## Finding 3: two new spread grammars, both real, liquid, and directly relevant to
## the Phase 2d roadmap - built, not skipped

Confirmed via real definition rows before writing any regex, per this session's
standing discipline:

- **`CL:BZ F0-G0`** - a labeled inter-commodity spread: root pair stated once,
  each leg keeps its own month code *and* digit (can differ in both month and
  year, e.g. `CL:BZ F5-G6` = Jan-2025 CL vs Feb-2026 BZ). This is the literal,
  exchange-quoted **WTI-Brent spread** - `WORKFLOW.md` Phase 2d's explicitly
  highest-priority relative-value candidate ("strongest of the 3... also the
  direct overlap point with the port congestion project's Track A/C"). A second,
  simpler convention for the same relationship (`CLK7-BZK7`, both legs' roots
  embedded per-leg) already resolved via the existing plain-dash parser - between
  the two formats, **66,291 real CL-BZ spread rows** now exist in
  `term_structure_spreads.parquet`.
- **`CL:C1 HO-CL K9`** - a crack spread: single shared month/year across two named
  roots (real crack-spread economics - the *same* delivery month compared across
  products, not two different months). `:C1` is a spread-type marker, not a
  product root (no CME product is named "C1"). Directly the 2-leg components of
  `WORKFLOW.md` Phase 2d's **3:2:1 crack spread candidate** - **24,065 RB-CL rows
  and 21,871 HO-CL rows** now captured.
- **Both generalized in the shared `_resolve_2leg_generic()` helper**, reusing the
  same anchor-offset logic already validated for the plain format - not a parallel
  implementation to maintain separately.
- **Validated at scale**: spread resolution went from 3,681/4,361 (`leg_parse_fail`
  on everything above) to **4,352/4,361 (99.8%)** - the remaining 9 are `UD:`
  User-Defined Spreads (2) and two never-before-seen, vanishingly small patterns
  (`CL:FS` "strip", `CL:SA` unknown - 7 instruments, ~11 rows, ~22 contracts of
  volume combined) correctly left as log-and-skip, not worth parsing for that
  volume. Butterflies: **445/445 resolved, 0 errors** - no new grammar needed
  there. 13 CL-side leg "mismatches" against the real outright universe all traced
  to genuine far-dated (2029-2032) contracts that are listed and spread-traded but
  have *never* had a standalone outright print - same "listed but zero outright
  volume" phenomenon already confirmed for `ZN`'s butterfly (Asset 12), not a
  resolution bug. `MCL` (Micro WTI) and `WS` correctly return `None` from
  `_leg_symbol` (untracked roots, not in the 42-asset universe) and are logged as
  `spread_unknown_root_exchange`, not silently mis-resolved.

## Net result

`status=PARTIAL` (expected - `spread_unknown_root_exchange=71` for `MCL`/`WS` and
`spread_not_2_legs=9` for the negligible strip/unknown patterns are both
legitimate, understood, log-and-skip cases), `non_positive_price_rows=2` now
correctly **visible** rather than silently dropped or invisible - a feature of the
fix, not a residual problem. 108,208 outright rows, 577,027 spread rows (up from
508,091 pre-fix), 70,965 butterfly rows, 0 condors. `term_structure_spreads.parquet`
now holds real, exchange-quoted WTI-Brent and crack-spread data ready for Phase 2d
to use directly, per the same "prefer the directly-quoted spread over
back-differencing" principle already established for carry.

---

# Asset 18: BZ (Brent) — 2026-07-19

Second energy asset, run specifically to independently cross-validate the CL:BZ
WTI-Brent spread parser built for WTI (Asset 17) - and to check whether WTI's
crack-spread grammar generalizes as-is.

## Finding: Brent's crack spreads need a more general grammar than WTI's

WTI's crack spreads only ever showed a single shared month/year across both legs
(`CL:C1 HO-CL K9`). Brent's real data has the same marker but a richer shape -
**both legs can have their own explicit month code, sometimes different months
and/or years** (e.g. `BZ:C1 HO H6-BZ J6` = HO March-2016 vs BZ April-2016). Checked
against real definition rows before assuming WTI's narrower pattern would
generalize, per this session's standing discipline - it didn't, cleanly.
**Fix**: made the first leg's month+digit optional in `_CRACK_RE`
(`(?: ([FGHJKMNQUVXZ]\d{1,2}))?`); when present, resolves via the same
`_resolve_2leg_generic()` anchor-offset logic as the labeled inter-commodity
format; when absent, falls back to WTI's original shared-month behavior. **Verified
no regression on WTI** (re-ran the exact same check: still 4,352/4,361, identical
error breakdown) before trusting the generalization, and confirmed the fix on
Brent: spread resolution went from 2,128/2,725 to **2,725/2,725 (100%)** - the only
remaining anomalies are two genuinely different, untracked NYMEX products (`OQD`,
`DCD`, confirmed via the plain dash format which already resolves fine — checked
their raw symbols directly rather than assumed).

## The real validation payoff: WTI's CL:BZ legs cross-checked against Brent's own data

Asset 17's `CL:BZ` resolution was built and merged *before* Brent's own outright
universe existed - the same structural gap LE's original inter-commodity
cross-check had ("the non-LE leg... doesn't have that same independent check
available"). Now that Brent is in, closed that gap: **all 66,291 CL-BZ spread
rows' `BZ`-side legs cross-checked against Brent's real, freshly-pulled outright
universe - 0 mismatches.** This is the strongest evidence yet that the labeled
inter-commodity parser is correct, not just internally self-consistent.

`status=PARTIAL` (expected, `spread_unknown_root_exchange=113` for the untracked
`OQD`/`DCD`), 0 non-positive prices, 0 zero-volume outrights, 49,531 outright rows,
236,724 spread rows (up from 205,188 pre-fix), 6,997 butterfly rows (188/188
resolved cleanly, no new grammar needed there), 0 condors.

---

# Assets 19-21: RB (RBOB), HO (HeatingOil), NG (Natural Gas) — 2026-07-19

Closing out the energy category. RBOB and HeatingOil are the crack spread's other
two legs (Asset 17); Natural Gas is structurally unrelated to the WTI/Brent/RBOB/
HeatingOil complex.

## RBOB and HeatingOil: the crack-spread cross-validation loop closes

Both merged clean (`status=PARTIAL`, only the negligible `FS`/`SA` strip patterns
remaining - 8 for RBOB, 49 for HeatingOil, same treatment as WTI's). The real
payoff: **both RB-CL (24,420 rows) and HO-CL (22,251 rows) crack-spread legs -
resolved during WTI's own run, before RBOB/HeatingOil had any data - cross-checked
against their own freshly-pulled real outright universes: 0 mismatches in either
case.** Combined with Brent's 0/66,291 (Asset 18), every cross-commodity leg built
this session for the energy complex is now independently confirmed correct, not
just internally self-consistent.

## Natural Gas: two more spread grammars found, deliberately left unbuilt

Two new marker patterns (`:XS`, `:SB`), both small enough and low-enough-value that
building parsers wasn't worth it, unlike WTI's CL:BZ/crack spreads:

- **`NG:XS 05M NG-HH X9`** (3,558 rows, 49,327 contracts volume) - always pairs
  `NG` against `HH` (Henry Hub, the physical natural-gas spot-price index). **Even
  if parsed, the `HH` leg could never be used or cross-validated** - Henry Hub is
  not a futures product with its own outright series in this project (not in the
  42-asset `UNIVERSE`, no `HH.NYM` price data exists anywhere in this pipeline).
  Structurally different from WTI's crack spreads, where *both* legs are real,
  already-tracked assets - the payoff here is much lower.
- **`NG:SB 07M J4-J5`** (361 rows, 2,795 contracts volume) - genuinely tiny, and
  the `07M` token's exact meaning wasn't confirmed (unlike every other marker in
  this project, which was pinned down before writing a parser).
- Both logged and skipped as `spread_leg_parse_fail` (63 instruments combined) -
  a deliberate cost/benefit call, not a gap left by oversight.
- Unknown-root spreads (1,759 - the largest anomaly count this session) are almost
  entirely `HH` (1,680) surfacing again via the plain dash format, plus small
  counts of other untracked NYMEX gas products (`MNG`, `QG`, `NN`) - all correctly
  resolved-but-unknown-root, not silently wrong.
- **Condors: 14/14 resolved, 0 leg mismatches** - the KC_Wheat condor algorithm
  (Asset 2) generalizes to a completely different asset category (energy vs.
  grains) with zero changes needed. Butterflies: 158/158 resolved.

## Energy category complete - summary

| Asset | Outright | Spread | Butterfly | Condor | Status |
|---|---|---|---|---|---|
| WTI Crude (CL) | 108,208 | 577,027 | 70,965 | 0 | PARTIAL |
| Brent (BZ) | 49,531 | 236,724 | 6,997 | 0 | PARTIAL |
| RBOB (RB) | 64,212 | 273,151 | 18,921 | 0 | PARTIAL |
| HeatingOil (HO) | 79,981 | 280,730 | 27,123 | 0 | PARTIAL |
| Natural Gas (NG) | 134,614 | 556,416 | 37,108 | 236 | PARTIAL |

All 5 merged with 0 non-positive prices among their own data (the only non-positive
rows anywhere in the full 867,496-row table are the 3 pre-existing, already-
explained OrangeJuice rows and the 2 genuine WTI April-2020 negative prints - Asset
17). Energy produced this session's most consequential single-asset finding (the
exact-zero filter fix) and its most valuable new capability (real WTI-Brent and
crack-spread data, directly serving `WORKFLOW.md` Phase 2d) - a strong argument for
the deliberate, no-fixed-pace, category-by-category approach adopted after Coffee.

---

# Cross-cutting fix: `jobs/capture_term_structure.py` gets the same exact-zero
# filter as the Databento transform (2026-07-20)

Found while investigating a *new* non-positive row for Cocoa (`CCU27.NYB`,
2025-10-01) that appeared mid-session without any Databento work touching Cocoa -
traced to the separate, independently-scheduled daily job
(`CTA_TermStructureCapture`) firing on its own schedule while this session's work
continued (confirmed directly via `Get-ScheduledTaskInfo`, not inferred - real
elapsed wall-clock time during this session was much longer than it felt,
spanning past that job's 6:15PM trigger and eventually past midnight into the next
day).

**Hypothesis raised and directly tested, not just assumed either way**: the user
noted the run that introduced this row happened on a Sunday, and asked whether
weekend runs specifically trigger the bug. **Disproven by direct reproduction**:
querying the same ticker live via `yfinance` on the *following Monday* returned the
identical zero-print for the same historical date - ruling out day-of-week as the
cause.

**Real root cause, confirmed by inspecting a wider date window**: every row in the
window (2025-09-22 through 2025-10-10) for this far-dated, thinly-listed OJ
contract has `Volume=0` - zero real trading the entire stretch. Almost every day
shows `Open=High=Low=Close` (a flat, carried-forward quote) - the exact
"synthetic/theoretical settlement print" pattern this project already documented
on 2026-07-14 for a *different* contract (`CLQ26.NYM`, logged in `DATA_SCHEMA.md`
section 1), well before this session. One day within that stretch (2025-09-30)
glitches to `Open=High=Low=0` instead of repeating the flat value, while `Close`
picks up the *next* day's flat value - an isolated artifact within an
already-known-unreliable no-volume stretch, not a new or day-of-week-linked bug.

**Fix**: `pull_contract_history()` now drops any row where `Open`/`High`/`Low`/
`Close` has an exact `0` while others don't - the identical rule (and identical
reasoning) already built and validated for `transform_databento.py`'s outright
filter (Asset 17): exact zero signals a fabricated/incomplete print; a genuine
negative price (the real WTI April-2020 event) never has an exact zero in any
field, only a truly missing one does. Also cleaned the 4 already-known bad rows
(3 OrangeJuice + this new Cocoa row) directly out of `Data/term_structure.parquet`
- confirmed safe first: applying the exact-zero rule retroactively to the *entire*
907,525-row archive removed exactly those 4 rows and nothing else, leaving the 2
genuine WTI negative-price rows untouched (neither has an exact zero in any
field). This fix protects only *future* runs from writing a new exact-zero row -
it doesn't retroactively fix anything the dedup logic doesn't re-pull, which is
why the manual one-time cleanup above was also needed.

---

# Generic fix: nearest-prior-definition fallback join (2026-07-20)

Found on FeederCattle (`GF`): 385 outright/spread/butterfly OHLCV rows
(2026-06-26 through 2026-07-13, right at the tail of the pull) had no matching
`definition` row on the *exact* date — but checked, not assumed unfixable: **all
44 unique `instrument_id`s involved do have a `definition` entry, just on an
earlier date.** Example: `instrument_id=42014336` has an OHLCV print on
2026-07-08, but its `definition` coverage stops at 2026-06-25 — CME apparently
stopped republishing this instrument's definition shortly before its last real
trade, likely near expiry/delisting from the actively-republished set. This is a
real gap in the underlying feed's republishing cadence, not a code bug.

**Safe to fix, not a guess**: `raw_symbol`/`instrument_class`/`maturity_year`/
`maturity_month` are immutable properties of a given `instrument_id` for its
entire life (established Section 1/Asset 1 - `instrument_id` is never reused
across contracts), so reusing an *older* definition record for the same
`instrument_id` is exactly as correct as an exact-date one would have been - it's
recovering already-true metadata, not inferring anything new.

**Fix**: `_join_ohlcv_definition()` now tries the exact `(date, instrument_id)`
join first (unchanged for every row that already worked), and only for rows that
fail that exact match, falls back to a `join_asof` (backward strategy) - the most
recent definition entry *for that same instrument_id* on or before the OHLCV
date. Generic, not FeederCattle-specific: wired into `transform_asset()`'s main
join, so it applies to every future asset with this same trailing-print-after-
definition-stops pattern. **Verified**: recovered all 385 FeederCattle rows (0
remaining unmatched), and re-ran WTI (previously 0 unmatched) to confirm no
regression - still 0 unmatched, identical row count, exact match unaffected for
rows that already worked.

---

# Assets 22-23: HE (LeanHogs), GF (FeederCattle) — 2026-07-20, livestock complete

Closing out livestock (LE was Asset 1). Both checked individually rather than
assumed to mirror LE, per this session's standing discipline.

## LeanHogs (HE): confirms LE's own cross-commodity finding, from the other side

`status=OK`, 0 anomalies of any kind - 41,284 outright rows, 114,125 spread rows,
26,301 butterfly rows, 0 condors. The real payoff: LE's original validation
(Asset 1) found real LE-HE inter-commodity spreads but could only cross-check the
LE-side leg (HE's own outright data didn't exist yet). Now it does - **11,766
LE-HE cross-commodity spread rows exist** (10,767 with `near_root=LE`, 999 with
`near_root=HE` - both orderings, expected from the near/far-by-date tie-break
logic), all real, liquid, and now checkable from both sides.

## FeederCattle (GF): surfaced the definition-republishing gap, fixed generically

Found the session's newest generic fix here: 385 rows (0.3% of GF's data,
2026-06-26 through 2026-07-13) had no `definition` row on the exact matching
date, though every instrument involved had one on an earlier date - see the
"Generic fix: nearest-prior-definition fallback join" section above for the full
investigation and fix. After the fix: `status=OK`, 0 unmatched, 31,498 outright
rows, 81,899 spread rows, 17,105 butterfly rows, 0 condors, 0 non-positive
prices. **Notably, 0 cross-commodity spread rows** - unlike LE-HE, FeederCattle
doesn't appear to have an actively-quoted cross-commodity spread partner in this
dataset. Not assumed to generalize from LE/HE - a real, asset-specific
market-structure difference within the same livestock category.

**Livestock category complete**: LiveCattle, LeanHogs, FeederCattle all done.

---

# Assets 24-28: ZC (Corn), ZW (Wheat), ZS (Soybeans), ZR (Rice), ZO (Oats) —
# 2026-07-20, grains complete

Closing out the remaining CBOT grains (KC_Wheat was Asset 2). All five checked
individually rather than assumed to mirror KC_Wheat.

## The real payoff: a validated grain-complex cross-commodity spread network

Unlike metals/rates/most energy, grains have a rich, real cross-commodity spread
network - directly relevant to `WORKFLOW.md`'s own Phase 2d candidate table:

| Pair | Rows | Economic relationship |
|---|---|---|
| ZC-ZW (Corn/Wheat) | 3,939 | The exact Phase 2d "Corn/Wheat spread" candidate - grain-complex substitution (feed use, planting-acreage competition) |
| KE-ZC (KC_Wheat/Corn) | 5,543 | Cross-grain substitution |
| KE-ZW (KC_Wheat/Wheat) | 29,636 | Two wheat variants (hard red winter vs. Chicago soft red winter) - the single largest cross-commodity spread found in this entire session |

**ZC-ZW legs cross-checked against Wheat's own real outright universe (resolved
during Corn's run, before Wheat had any data): 0 mismatches** - same validation
rigor as every other cross-commodity pair this session (LE-HE, CL-BZ, RB/HO-CL).

**Soybeans (`ZS`) has zero cross-commodity spread rows** - checked directly, not
assumed. Its natural spread partners (Soybean Meal `ZM`, Soybean Oil `ZL`, the
Soybean Crush) aren't in this pull at all, because those two products aren't in
the 42-asset `UNIVERSE` yet (`WORKFLOW.md` open decision #3, still unresolved -
this session's grain work doesn't change that). Soybeans' own
`spread_unknown_root_exchange=15` is Micro Soybeans (`MZS`) and a legacy code
(`XK`), not a hidden crush-spread partner.

## Category-wide results

| Asset | Outright | Spread | Butterfly | Condor | Unknown-root | Status |
|---|---|---|---|---|---|---|
| Corn (ZC) | 49,138 | 127,956 | 23,569 | 563 | `MZC`:10, `XC`:2 | PARTIAL |
| Wheat (ZW) | 38,397 | 106,625 | 19,451 | 747 | 13 (untracked) | PARTIAL |
| Soybeans (ZS) | 50,597 | 142,791 | 29,906 | 2,276 | `MZS`:13, `XK`:2 | PARTIAL |
| Rice (ZR) | 15,162 | 15,255 | 75 | 0 | none | **OK** |
| Oats (ZO) | 14,995 | 14,000 | 381 | 0 | none | **OK** |

Rice and Oats are the cleanest assets processed all session - `status=OK` with
zero anomalies of any kind, not even a Micro-product unknown-root spread. All
five had 0 non-positive prices, 0 join-unmatched rows, 0 OHLC violations. Condor
volume scales with how actively each grain's curve is traded as combos - Soybeans
(2,276) and Wheat (747) noticeably higher than KC_Wheat's original 92, Rice and
Oats (thinner markets) showing none at all.

**Grains category complete**: KC_Wheat, Corn, Wheat, Soybeans, Rice, Oats all done
(6 of 6).

---

# Asset 29: SR3 (SOFR) — 2026-07-20, richest single-asset exploration this session

Explored the raw data thoroughly *before* running anything (per direct correction
mid-session — every prior category's first asset got this treatment, this one
initially didn't and should have). Real payoff: SOFR turned out structurally
richer than every other asset combined.

## Structural differences confirmed before touching the transform

- **46 simultaneously-listed outright contracts** (monthly, not quarterly -
  consecutive month codes J6/K6/M6/N6/Q6... extending to Dec-2035), vs. 3 for
  Treasury note/bond futures. Real: SOFR lists serial (monthly) contracts for
  the front years, unlike deliverable Treasury futures.
- **`main_fraction=255` (sentinel/null)** - decimal-quoted, not fractional-32nds
  like Treasuries. Confirms the price-display convention genuinely varies within
  "rates" as a category, not just between rates and other asset classes.
- **Four new spread-family markers beyond `:BF`/`:CF`**, none seen in any other
  asset this session: `:AB`, `:DF`, `:BB`, `:SB`.

## `:DF` and `:BB` - free wins, zero new logic

Structurally identical to existing formats, just different CME/rates-market
terminology:
- **`:DF`** (e.g. `SR3:DF H1M1U1Z1`) - 4 concatenated leg codes, no separator -
  exactly the condor shape. **99/99 resolved using the existing
  `resolve_condor_legs()` unchanged**, once `:DF` was added as a second trigger
  marker alongside `:CF`. Real, liquid: 15,307 rows, 2.5M contracts of volume.
- **`:BB`** (e.g. `SR3:BB H7-H8-H9`) - 3 dash-separated leg codes - exactly the
  butterfly shape (here representing same-month-different-years "bundle"
  combinations, not adjacent-months-same-year, but the anchor-matching algorithm
  doesn't assume either pattern, so it resolves correctly regardless). **12/12
  resolved using `resolve_butterfly_legs()` unchanged.** Small (133 rows) but
  free.

Both wired in by adding the alternate marker string to each resolver's marker
check (`":BF " if ... else ":BB " if ...`) - the anchor-leg-plus-modular-offset
algorithm itself needed zero changes for either.

## `:AB` - CME SOFR Average futures, a genuinely new single-quote instrument type

`SR3:AB 02Y U8` = the 2-year SOFR average anchored to Sept-2028. **Not a
multi-leg spread at all** - a single, real, actively-traded instrument (verified:
prices in SOFR's genuine ~95-97 quoting range, tens of thousands of contracts/day
- 78.6M total volume, the largest of any combo type found this session).
Confirmed the trailing month+digit always equals the row's own
`maturity_year`/`month` directly (336/336, 0 mismatches) - simpler than every
other combo type here, no anchor-relative offset needed, just a sanity check.

**Built a new table, `Data/term_structure_averages.parquet`** (`date, OHLCV,
asset, average_symbol, root, avg_years, anchor_contract_symbol,
anchor_expiry_year, anchor_expiry_code`) - `resolve_average_instrument()`,
**121/121 resolved, 0 mismatches**, 19,663 rows merged.

## `:SB` - CME Pack/Bundle spreads, a second new instrument type

Packs (4 consecutive quarters starting anywhere) and Bundles (`NY` = N years of
consecutive quarters from a given start), spread against each other and
referenced by just their starting quarter on each side. Real but moderate volume
(90 instruments, 674K contracts) - explicitly discussed with the user given it's
smaller than `:AB` and doesn't fit the near/far leg schema conceptually (each
"leg" is an aggregate of 4 quarters, not a single contract) - **decision: build
it anyway**, treating each side's *starting* quarter as sufficient to identify
the strategy (same simplification already implicit in how the raw symbol itself
references packs/bundles).

**Two distinct raw_symbol grammars for the identical strategy, confirmed to
cover all 315 real `:SB` symbols with no third variant** - checked exhaustively,
not assumed to be one format with parsing noise:
- `PK H7-H8` / `4Y Z2-Z6` - space-separated, one type token, then a plain
  leg-code pair.
- `4YZ1-4YZ5` - no space, the bundle length repeated as a prefix on each leg
  independently. Never seen with `PK` repeated this way - packs only use the
  first form.

**Built a new table, `Data/term_structure_packs.parquet`** (`date, OHLCV, asset,
pack_symbol, pack_type, near/far_root, near/far_contract_symbol,
near/far_expiry_year, near/far_expiry_code`) - `resolve_pack_spread()` tries both
grammars in turn, then reuses `_resolve_2leg_generic()` (the same function behind
the plain and labeled-inter-commodity spread formats) for the actual year
resolution. **90/90 resolved, 0 mismatches**, 6,998 rows merged. Confirmed this
was the *only* remaining `spread_leg_parse_fail` source for SOFR - 0 unresolved
of that type after the fix.

## Also found: generic join-fallback triggered again (537 rows)

Same "definition stopped republishing before the last real print" pattern first
found on FeederCattle (Asset 22-23's "Generic fix" section) - the existing fix
handled it automatically, no new code needed.

## Net result

`status=PARTIAL` (expected - `spread_unknown_root_exchange=104` for genuinely
untracked products `GE` [legacy Eurodollar], `TBF3`, `BSB` is the only remaining
anomaly), 0 non-positive prices anywhere. 45,795 outright rows, 218,567 spread
rows, 64,696 butterfly rows (combining `:BF`+`:BB`), 23,267 condor rows
(combining `:CF`+`:DF`), 19,663 average rows (new table), 6,998 pack rows (new
table). **Two new output tables added to the pipeline in one asset** -
`term_structure_averages.parquet` and `term_structure_packs.parquet` - both
generic (not hardcoded to SOFR), so they'll pick up automatically if any other
future asset uses the same markers.

**Also noted, not fixed**: `coverage_missing_sessions`/`_pct` came back `None`
for SOFR - traced to `ASSET_CALENDAR` (in `jobs/update_dashboard_summary.py`) not
having an entry for SOFR yet. Pre-existing, already-logged gap (`WORKFLOW.md`'s
2026-07-17 universe-expansion note: "not yet done: adding these 4 to... the
dashboard's `ASSET_CALENDAR` mapping"), not introduced by this work - flagged so
it isn't rediscovered as a new mystery.

---

# Assets 30-31: ES (SP500), NQ (Nasdaq100) — 2026-07-20, equity index begins

## SP500 (ES): the cleanest category start this session

Explored first, per direct instruction. Quarterly-only listing (21 contracts on
one sample day, out to 2031 - same cadence as rates/FX, not the monthly pattern
SOFR has), decimal-quoted (`main_fraction` sentinel, not populated), and a new
`unit_of_measure` value never seen before: `IPNT` (Index Point) -
`unit_of_measure_qty` ÷1e9 = $50, the real E-mini S&P 500 multiplier. All 20
S-class symbols on the sample day were plain 2-leg calendar spreads - confirmed
at full scale: **`status=OK`, 0 anomalies of any kind** (0 butterflies, 0
condors, 0 averages, 0 packs) - cash-settled index futures don't have the
physical-delivery or rate-curve conventions that produce those elsewhere. 0
cross-index spreads (no ES-NQ combo in this data). 17,105 outright rows, 12,791
spread rows.

## Nasdaq100 (NQ): a real, pre-existing data-corruption bug, unrelated to any
## code from this session

First transform attempt returned `join_unmatched_rows` equal to 100% of
`rows_read` - every single row unmatched. **Traced directly, not guessed**:
`CME_Globex_MDP3.0_NQ_FUT_OHLCV.zip`'s `definition` counterpart correctly showed
real `NQ` symbols, but the **OHLCV file for the same dates contained `ES`
symbols instead** (`ESU6-ESZ6`, `ESZ6`, `ESM7`, etc.) - confirmed reproducible
across two different dates (2026-07-13 and 2026-01-05), so not a one-off
artifact. The two zips are NOT literal duplicates (different SHA256/MD5,
coincidentally identical file size), so this wasn't a simple copy-paste of the
wrong file wholesale - something in the original local download/assembly
process (predating this session - the file's mtime was 2026-07-17) mixed up
which job's output went into which zip.

**Confirmed via the Databento API directly, not assumed**: the original batch
job (`GLBX-20260717-9D5SJSA8EX`) was correctly submitted for `NQ.FUT`, completed
successfully, and isn't expired - so the mixup happened in the local
download/assembly step, not at submission or on Databento's servers. Verified
by downloading one file fresh, directly from that job: correct `NQ` symbols
(`NQH7`, `NQU6`, `NQZ6`). **Recovered the same way as Copper (Asset 11)**: all
4,996 individual OHLCV files re-downloaded directly via `filename_to_download`
(bypassing whatever assembled the original wrong zip), each SHA256-verified
against Databento's own reported hash, 0 failures, $0 cost (re-fetching an
already-completed, unexpired job). Rebuilt a valid zip from the verified files,
cleared the stale (wrong-content) local extraction cache, and re-ran: `status=OK`,
0 anomalies, 0 unmatched, 13,302 outright rows, 6,149 spread rows. Price range
$1,712-$32,378 matches the real Nasdaq-100 index's historical growth, confirming
the recovered data is genuinely NQ, not a repeat of the ES mixup.

**Also fixed while diagnosing this: a real, generalizable Windows file-locking
bug in the write path itself.** `_merge_append`/`_merge_outrights` read the
existing output parquet, then immediately try to overwrite that same path -
once `term_structure_spreads.parquet` grew large enough (after SOFR's rich
data) that polars' reader memory-maps it internally, Windows refused the
overwrite ("The requested operation cannot be performed on a file with a
user-mapped section open") - a self-conflict between the read and the write
within a single process, confirmed by the error persisting even with zero other
processes running. **Fix**: all 6 output-table writes now go through
`_write_parquet_atomic()` - write to a temp file, then `os.replace()` into the
real path - which never asks the OS to overwrite a path it just memory-mapped
for reading. Protects every future asset's merge+write, not just this one.

---

# Assets 32-33: YM (Dow), RTY (Russell2000) — 2026-07-20, equity index complete

Both checked for the same NQ-style OHLCV/definition symbol mismatch *before*
running anything, given it had just been found in this exact category - both
clean (`YMU6`/`YMZ6`/`YMU6-YMZ6` and `RTYU6`/`RTYZ6`/`RTYU6-RTYZ6` respectively,
correct roots throughout). Both confirm SP500's structural pattern: quarterly
listing, decimal-quoted, `IPNT` unit (Dow $5/point, Russell $50/point - both
match real contract specs).

Dow's definition sample showed 2 real `:BF` butterfly symbols (unlike SP500's
zero) - but neither ever had a real OHLCV print, same "listed but zero
volume" phenomenon already confirmed for `ZN`/`WTI`/`SOFR` (Assets 12, 17, 29).
Both merged `status=OK`, 0 anomalies of any kind: Dow 12,024 outright / 4,956
spread rows; Russell2000 5,771 outright / 2,475 spread rows. 0 cross-index
spreads for either (matching SP500 and Nasdaq100 - equity index futures don't
appear to have an actively-quoted cross-index combo in this dataset, unlike the
grain or energy complexes).

**Equity index category complete**: SP500, Nasdaq100, Dow, Russell2000 all done
- the cleanest category overall this session (0 butterflies/condors/averages/
packs with real volume across all four), aside from the one real, pre-existing
data-corruption incident on Nasdaq100 (Asset 31).

---

# Asset 34: LBR (Lumber) — 2026-07-20, a category of one, and the 42nd asset

Checked first for the same NQ-style OHLCV/definition mismatch given it's a new
category - clean (`LBRX6`, `LBRN6`, `LBRU6`, correct root throughout).

## New unit, new real cross-commodity partner, sparse listing confirmed

- **`unit_of_measure = "BDFT"`** (Board Feet) - a unit never seen before,
  correct for the physical lumber measurement. `unit_of_measure_qty` ÷1e9 =
  27,500 board feet, the real modern CME Lumber contract size (the
  post-2022-redesign contract this project already confirmed using, per
  `WORKFLOW.md`'s "not `LBS`, discontinued ~2022" note).
- **Sparse listing confirmed directly**: only 8 outright months listed at once
  (odd months - N/U/X/F/H/K, not a clean quarterly or monthly cycle), matching
  the "sparser listing cycle than most" note already logged when Lumber was
  added to the universe (2026-07-17).
- **Real cross-commodity spread partner found: `SYP`** (a related lumber
  product, correctly logged as `spread_unknown_root_exchange` since it isn't in
  the 42-asset `UNIVERSE`) - plus a handful of spreads against the old,
  discontinued `LBS` contract (also correctly untracked).

All 35 S-class symbols on the sample day were plain 2-leg format - no new
grammar needed. `status=PARTIAL` (expected - `spread_unknown_root_exchange=13`,
fully explained), 0 non-positive prices, 2,915 outright rows, 2,377 spread
rows. Price range $352.50-$740.00, a plausible magnitude for the modern
contract. `coverage_missing_sessions`/`_pct` came back `None` - same
pre-existing `ASSET_CALENDAR` gap already logged for SOFR (Asset 29), not a
new issue.

**35th asset transformed via Databento** (checked against the manifest
directly, not assumed from the outright table's asset count, which also
includes yfinance-only rows for not-yet-transformed FX assets). **Lumber's own
category is now fully done** - the remaining 7 are all FX (EURUSD, JPYUSD,
GBPUSD, AUDUSD, CADUSD, SwissFranc, MexicanPeso).

---

# Assets 35-41: 6E (EURUSD), 6J (JPYUSD), 6B (GBPUSD), 6A (AUDUSD), 6C (CADUSD),
# 6S (SwissFranc), 6M (MexicanPeso) — 2026-07-20, FX complete, and the final
# category in the 42-asset universe

All 7 checked for the NQ-style OHLCV/definition root mismatch before running
anything, given the precedent from Nasdaq100 (Asset 31). Found a different,
genuinely new bug on `CADUSD`.

## Structure (EURUSD explored first, representative of the category)

37 outright contracts listed at once - a hybrid cycle, monthly for the near
years then quarterly further out (structurally similar to SOFR's monthly-then-
quarterly pattern, Asset 29, but over a much shorter horizon). New
`unit_of_measure` value: the foreign currency itself (`EUR` for 6E, etc.) -
`unit_of_measure_qty` ÷1e9 = 125,000 EUR, the real CME EUR/USD contract size;
`min_price_increment` = 0.5 pip, also real. All 69 S-class symbols on the
sample day were plain 2-leg format - none of the six special markers
(`:BF`/`:CF`/`:AB`/`:DF`/`:BB`/`:SB`) found anywhere in FX. Confirmed at full
scale for EURUSD: `status=OK`, 0 anomalies, 25,289 outright / 21,204 spread
rows. **0 cross-currency-pair spreads across all 7 FX assets** (checked
explicitly, e.g. no 6E-vs-6J combo) - matching equity index's pattern, unlike
grains/energy's rich cross-commodity networks.

## CADUSD (6C): a real, pre-existing bug - the Definition and OHLCV zips were
## simply swapped

Root-mismatch check on `6C` raised an exception before even getting to the
symbol comparison: `CME_Globex_MDP3.0_6C_FUT_OHLCV.zip` contained only
`*.definition.csv.zst` files, and `..._Definition.zip` contained only
`*.ohlcv-1d.csv.zst` files - **the two zips' filenames were swapped**, a
different failure mode from Nasdaq100's (wrong *asset's* data under a correct
schema name) but caught by the same "verify the actual file content, don't
trust the filename" discipline. Confirmed both sides held genuine `6C`
(CADUSD) data before fixing - not a data-loss risk, purely a local
naming mistake predating this session. **Fix: renamed the two files back to
their correct names** (no re-download needed, unlike Nasdaq100 - both real
datasets were already present, just mislabeled). Re-ran clean: `status=OK`, 0
anomalies, 23,446 outright / 16,487 spread rows.

## SwissFranc, MexicanPeso: confirm the pre-existing `ASSET_CALENDAR` gap

Both show `coverage_missing_sessions`/`_pct` as `None`, same as SOFR and
Lumber (Assets 29, 34) - the pre-existing, already-logged gap (these 4
assets were added to the universe 2026-07-17 but never added to
`jobs/update_dashboard_summary.py`'s `ASSET_CALENDAR`). Not a new issue, not
fixed this pass (out of scope, already tracked).

## FX category and full-universe summary

All 7 merged `status=OK`, 0 non-positive prices, 0 join-unmatched rows beyond
the one bug found and fixed:

| Asset | Outright | Spread |
|---|---|---|
| EURUSD (6E) | 25,289 | 21,204 |
| JPYUSD (6J) | 20,032 | 14,629 |
| GBPUSD (6B) | 18,558 | 13,927 |
| AUDUSD (6A) | 18,540 | 12,748 |
| CADUSD (6C) | 23,446 | 16,487 |
| SwissFranc (6S) | 11,039 | 3,448 |
| MexicanPeso (6M) | 10,218 | 2,619 |

**This completes the entire 42-asset universe.** Final state across all five
output tables, verified directly against the actual files (not the manifest
alone):

| Table | Rows | Assets represented |
|---|---|---|
| `term_structure.parquet` (outrights) | 1,318,677 | **42 / 42** |
| `term_structure_spreads.parquet` | 3,396,663 | 37 |
| `term_structure_butterflies.parquet` | 375,857 | 15 |
| `term_structure_condors.parquet` | 24,117 | 6 |
| `term_structure_averages.parquet` (new, Asset 29) | 19,663 | 1 (SOFR) |
| `term_structure_packs.parquet` (new, Asset 29) | 6,998 | 1 (SOFR) |

**Non-positive prices anywhere in the entire outright table: 2** - both the
genuine, historically-documented WTI April-2020 negative settlement (Asset
17), not an artifact. Every other known anomaly across the whole rollout
(sentinels, incomplete bars, decade-ambiguous symbols, condor/pack/average
markers, join gaps, corrupted/swapped/mislabeled zips) was found, diagnosed
against real evidence, and either fixed generically or explicitly logged as an
understood, deliberate skip - not left as an open question.

---

## Appendix A: OHLCV-1D schema, field by field

10 columns. One row per (instrument, trading day).

| Column | What it is | Example (`LEZ6`, 2026-07-13) |
|---|---|---|
| `ts_event` | Nanosecond timestamp (since 1970-01-01 UTC) marking the trading session this bar covers. | `1783900800000000000` → 2026-07-11 |
| `rtype` | Databento's internal record-type discriminator — identifies which schema this row belongs to. Constant within a single `ohlcv-1d` pull. | `35` |
| `publisher_id` | Numeric ID for the specific dataset+venue combination (Databento's publisher registry). | `1` (= `GLBX.MDP3`, CME Globex) |
| `instrument_id` | Databento/CME's numeric ID for this specific contract *instance*. Confirmed stable for a contract's entire life, never reused for a different contract (Section 1) — but not usable as a cross-time symbol lookup without the `definition` join. | `42005708` |
| `open` | Session opening price. Fixed-point integer, ÷1e9 = real price. | `230550000000` → `230.55` |
| `high` | Session high. Same scaling. | `232550000000` → `232.55` |
| `low` | Session low. Same scaling. | `230375000000` → `230.375` |
| `close` | Session close/settlement price. Same scaling. | `230825000000` → `230.825` |
| `volume` | Contracts traded that session. Plain integer, no scaling. | `5569` |
| `symbol` | The resolved, human-readable ticker for this instrument on this date. Present only because the job was submitted with `map_symbols: true` — decade-ambiguous, see Section 1. | `LEZ6` |

## Appendix B: Definition schema, field by field

65 columns. One row per instrument per publication event — not deduplicated per day
(Section 3). Values below are from a real row: the December 2026 Live Cattle outright
(`raw_symbol=LEZ6`, `instrument_id=42005708`), as published 2026-07-13.

**A note on the huge sentinel numbers below** (`2147483647`, `9223372036854775807`,
`65535`, `255`, `127`): these are not malformed data. Databento's underlying binary
format (DBN) uses fixed-width integers, and its convention for "not applicable to this
instrument" is the maximum value representable in that field's width, carried through
literally into the CSV rather than left blank:

| Sentinel | Bit width | Meaning |
|---|---|---|
| `9223372036854775807` | signed 64-bit max (2⁶³−1) | Field not applicable (e.g. `strike_price` on a future) |
| `2147483647` | signed 32-bit max (2³¹−1) | Field not applicable (e.g. `contract_multiplier` on this product) |
| `65535` | unsigned 16-bit max (2¹⁶−1) | Field not applicable (e.g. `decay_start_date`) |
| `255` | unsigned 8-bit max (2⁸−1) | Field not applicable (e.g. `maturity_day` when only year+month matter) |
| `127` | signed 8-bit max (2⁷−1) | Field not applicable (e.g. `contract_multiplier_unit`) |

Confirmed by example: `strike_price`, `price_ratio` = `9223372036854775807` — correctly
null, since outright futures don't have a strike price (that field exists in this
schema because `definition` covers CME options too, in the same table shape).

**1. Identity & versioning — which record is this, has it changed**

| Column | What it is | Example |
|---|---|---|
| `ts_recv` | Nanosecond timestamp Databento received this record. | `1783900800000000000` |
| `ts_event` | Nanosecond timestamp CME generated this record. | `1783857868281000000` |
| `rtype` | Record-type discriminator for the `definition` schema. | `19` |
| `publisher_id` | Venue/feed ID. | `1` (GLBX.MDP3) |
| `instrument_id` | CME's numeric ID for this instrument instance. Stable for life; the join key used throughout this report. | `42005708` |
| `raw_symbol` | CME's raw ticker string — the decade-ambiguous one (Section 1). | `LEZ6` |
| `symbol` | Same as `raw_symbol` in this pull (both decade-ambiguous). | `LEZ6` |
| `security_update_action` | Single-letter code for what kind of update this record is (`A`=add, `M`=modify, others exist for delete). Explains why duplicate rows exist per day (Section 3) — each republish is its own "update" event. | `A` |

**2. What kind of instrument this is**

| Column | What it is | Example |
|---|---|---|
| `instrument_class` | The field the transform filters on: `F`=outright future, `S`=spread/combo (both calendar spreads and butterflies, Section 2). | `F` |
| `security_type` | Plain-text instrument type. | `FUT` |
| `cfi` | ISO 10962 Classification of Financial Instruments code — a standardized 6-character instrument taxonomy (first letter `F`=Futures; the remaining letters encode finer detail this report doesn't decode with full confidence). | `FCAXSX` |
| `group` | Product group code — matches the asset root for this product. | `LE` |
| `exchange` | ISO 10383 Market Identifier Code (MIC). | `XCME` |
| `asset` | Product root — what this project keys `contract_symbol` on. | `LE` |
| `underlying` | The underlying instrument, for derivatives-of-an-instrument (e.g. options on a future). Null for an outright future. | *(empty)* |
| `secsubtype` | Further security subtype classification. Null here. | *(empty)* |
| `underlying_product` | Numeric code for the underlying product family. | `2` |

**3. Contract economics — how to interpret a raw price or quantity**

| Column | What it is | Example |
|---|---|---|
| `contract_multiplier` | Dollar value per price point, for products that use this convention. Null (sentinel) for Live Cattle — this product expresses size via `unit_of_measure_qty` instead. | `2147483647` (null) |
| `contract_multiplier_unit` | Unit the multiplier above is expressed in. Null here. | `127` (null) |
| `unit_of_measure` | The physical unit prices are quoted against. | `LBS` |
| `unit_of_measure_qty` | Contract size in that unit. Fixed-point, ÷1e9. **Verified: 40,000 lbs — the real, correct CME Live Cattle contract size.** | `40000000000000` → `40,000` |
| `price_ratio` | A price-conversion ratio used for certain product types. Null (sentinel) here. | `9223372036854775807` (null) |
| `display_factor` | A *separate* scale factor from the universal ÷1e9 used for OHLCV/limit prices — applies to currency-*amount*-typed fields like `min_price_increment_amount` below. Confirmed by arithmetic, not assumed (see the tick-value check below). | `1000000` |
| `min_price_increment` | The tick size, in quoted price units. Fixed-point, ÷1e9. | `25000000` → `0.025` |
| `min_price_increment_amount` | Dollar value of one tick. Scaled by `display_factor`, not the standard ÷1e9. **Verified two independent ways: (a) 0.025 (tick, quote units) × 40,000 lbs ÷ 100 = $10.00; (b) 10,000,000 ÷ `display_factor` (1,000,000) = $10.00. Both match, and both match the real, documented CME Live Cattle tick value.** | `10000000` → `$10.00` |

**4. Lifecycle dates — when this specific contract exists**

| Column | What it is | Example |
|---|---|---|
| `activation` | Nanosecond timestamp this contract was first listed. Decoded: 2025-06-06 21:30 UTC — about 18 months before expiration, consistent with CME's typical multi-year-out listing cycle for Live Cattle. | `1749245400000000000` |
| `expiration` | Nanosecond timestamp this contract stops trading. Decoded: 2026-12-31 18:00 UTC. | `1798740000000000000` |
| `maturity_year` | Unambiguous 4-digit contract year — **the field that fixes the decade-ambiguity problem (Section 1).** | `2026` |
| `maturity_month` | Contract month, 1-12. | `12` |
| `maturity_day` | Contract day, for products with day-specific maturities. Null (sentinel) for a standard monthly future. | `255` (null) |
| `maturity_week` | Contract week, for weekly-maturity products. Null here. | `255` (null) |
| `trading_reference_date` | The date `trading_reference_price` below was struck. Encoded differently from the timestamp fields above — as a **day count since 1970-01-01**, not nanoseconds. Decoded: 2026-07-10 (the prior trading day relative to this file's 2026-07-13 date). | `20644` |
| `decay_start_date` | Start date for instruments with an amortizing/decaying notional (e.g. some fixed-income products). Null (sentinel) — not applicable to a physically-settled livestock future. | `65535` (null) |

**5. Trading limits & risk parameters — exchange-set guardrails, not market prices**

| Column | What it is | Example |
|---|---|---|
| `high_limit_price` | Upper bound of the day's allowed price band. Fixed-point, ÷1e9. | `238775000000` → `238.775` |
| `low_limit_price` | Lower bound of the day's allowed price band. | `221775000000` → `221.775` |
| `max_price_variation` | Exchange parameter used to compute the limit band above/below the reference price. This report did not fully reverse-engineer the exact formula relating this value to the limit prices above — flagged rather than guessed. | `975000000` |
| `trading_reference_price` | The settlement/reference price the limit band is centered on. | `230275000000` → `230.275` |

**6. Lot sizing — minimum/maximum tradeable quantities**

| Column | What it is | Example |
|---|---|---|
| `min_lot_size` | Minimum order size. `0` observed — likely means "no restriction beyond the exchange default" rather than a literal zero-size order; not fully confirmed. | `0` |
| `min_lot_size_block` | Minimum size for a privately-negotiated block trade. | `0` |
| `min_lot_size_round_lot` | Minimum size considered a standard "round lot." | `0` |
| `min_trade_vol` | Minimum tradeable volume on the open market. | `1` |
| `max_trade_vol` | Maximum single-order size. | `500` |

**7. Currency**

| Column | What it is | Example |
|---|---|---|
| `currency` | Quote currency. | `USD` |
| `settl_currency` | Settlement currency, when different from quote currency. Null (same as `currency`) here. | *(empty)* |

**8. Options-specific fields — present in the schema, not applicable to futures**

| Column | What it is | Example |
|---|---|---|
| `strike_price` | Option strike price. Null (sentinel) — `definition` is a shared schema across CME futures *and* options; these fields exist for the option rows, not this one. | `9223372036854775807` (null) |
| `strike_price_currency` | Currency the strike is quoted in. Null here. | *(empty)* |

**9. Market microstructure & order-matching mechanics**

| Column | What it is | Example |
|---|---|---|
| `match_algorithm` | Single-letter code for which order-matching algorithm this instrument uses (CME publishes several — FIFO, pro-rata, and others). This report did not confirm the exact code-to-algorithm mapping with full certainty — flagged rather than guessed. | `K` |
| `md_security_trading_status` | Current trading status code (halted, open, etc.), as of this definition snapshot. Null (sentinel) — status is carried in a separate real-time status message, not this reference record. | `255` (null) |
| `market_depth` | Number of price levels disseminated in the outright order book. | `5` |
| `market_depth_implied` | Number of price levels disseminated for implied (derived-from-spread) pricing. | `2` |
| `market_segment_id` | Numeric ID for the CME market segment this instrument trades in. | `70` |
| `channel_id` | Numeric ID for the specific data channel this instrument is disseminated on. | `6` |

**10. Legacy / lower-relevance fields for this project**

Real CME/Databento fields, not currently needed by this project's transform:

| Column | What it is | Example |
|---|---|---|
| `inst_attrib_value` | A bitmask of instrument attribute flags (e.g. electronic-eligible, implied-pricing-eligible). This report did not decode individual bits. | `794699` |
| `underlying_id` | `instrument_id` of the underlying instrument, for derivatives-of-an-instrument. `0` for an outright. | `0` |
| `raw_instrument_id` | CME's own raw instrument ID, as opposed to Databento's `instrument_id`. Identical value in this row. | `42005708` |
| `appl_id` | Internal CME application/system ID. | `316` |
| `decay_quantity` | Notional decay amount, for amortizing instruments. Null (sentinel). | `2147483647` (null) |
| `original_contract_size` | Original size before any decay, for amortizing instruments. Null (sentinel). | `2147483647` (null) |
| `main_fraction` | Numerator for fractional price display (e.g. bond prices quoted in 32nds). Null (sentinel) — Live Cattle is decimal-quoted. | `255` (null) |
| `sub_fraction` | Denominator refinement for fractional price display. Null (sentinel). | `255` (null) |
| `price_display_format` | Code for how price should be displayed (decimal vs. fractional). Null (sentinel). | `255` (null) |
| `settl_price_type` | Code classifying the type of settlement price (e.g. final vs. preliminary). A real, populated code here — not decoded further in this report. | `3` |
| `flow_schedule_type` | Code relating to physical delivery/flow scheduling, relevant to some physically-delivered products. Null (sentinel) here. | `127` (null) |
| `tick_rule` | Code for any non-uniform tick-size rule (e.g. different tick sizes at different price levels). Null (sentinel) — Live Cattle has a single uniform tick. | `255` (null) |
| `user_defined_instrument` | Flag (`Y`/`N`) for whether this is an exchange-listed instrument or a user-defined combination. `N` — this is a real, exchange-listed instrument. | `N` |
