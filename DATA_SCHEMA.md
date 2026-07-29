# Data Infrastructure

This project's data layer is built the way a production research desk's would
be, not assembled from a single downloaded CSV: two independent price sources
chosen for what each is actually good at, per-contract daily exchange data
decoded directly from raw archives, a real continuous-futures construction, and a
point-in-time-correct macro layer that a backtest can't accidentally cheat
against. This document describes what's in `Data/`, how it's produced, and
why each design decision is what it is.

---

## 1. Core daily panel — 42 markets, commodities/FX/rates/equities

A scheduled `yfinance` pipeline pulls full-history daily OHLCV for 41 of 42
markets (the 42nd, SOFR, has no viable continuous ticker on this source — it's
covered instead by the term-structure pipeline in Section 2). Universe:

| Sector | Assets |
|---|---|
| Energy | WTI Crude, Brent, Natural Gas, RBOB, Heating Oil |
| Precious Metals | Gold, Silver, Platinum, Palladium |
| Industrial Metals | Copper |
| Grains | Corn, Soybeans, Wheat, KC Wheat, Rice, Oats |
| Softs | Coffee, Sugar, Cocoa, Cotton, Orange Juice |
| Livestock | Live Cattle, Lean Hogs, Feeder Cattle |
| Equity Index | S&P 500, Nasdaq 100, Dow, Russell 2000 |
| Rates | 2Y, 5Y, 10Y, 30Y Treasuries, Ultra Bond |
| FX | EUR, JPY, GBP, AUD, CAD, CHF, MXN vs. USD |
| Other | Lumber |

This panel runs daily on a scheduler, independently of any other job, and is
the baseline every signal and estimator in this project ultimately touches.

## 2. Daily per-contract term structure — Databento, decoded from raw exchange archives

The core panel above gives one continuous price per asset. Several signal
families (carry, relative-value spreads, anything that needs to reason about
the shape of the futures curve) need the individual contract months
themselves — which isn't something a continuous-ticker vendor feed provides.

This project sources that directly from **Databento's raw `ohlcv-1d` and
instrument-`definition` archives** (CME Globex `GLBX.MDP3` for financial and
agricultural futures, ICE Futures US `IFUS.IMPACT` for the softs) — the same
class of feed an institutional desk would consume, not a pre-cleaned
vendor panel. For 33 of the 38 CME-listed core assets this recovers the full
outright curve plus every exchange-quoted **spread, butterfly, condor,
average, and pack** instrument; the 4 ICE softs (thin, budget-scoped) fall
back to a labeled back-differenced proxy rather than being silently treated
as equivalent.

**The decoding problem this pipeline solves**: raw exchange symbology encodes
a contract's expiry with a single decade-ambiguous digit (`LEQ5` could be
2015 or 2025), and a multi-leg spread's symbol only names its *legs*, not
which one anchors the whole instrument's listed maturity. This project
reverse-engineers and applies an anchor-leg algorithm — the leg whose month
code matches the combo instrument's own listed maturity is the anchor, every
other leg's absolute year is resolved as a small forward offset from it, and
when multiple legs share the anchor's month code (same-month cross-commodity
spreads, SOFR's "bundle" butterflies), each candidate is tried and the one
producing the tightest total leg-to-leg year span wins — the economically
sensible resolution, since a real near-dated combo's legs are never many
years apart.

**Built on `polars`, not pandas, for the scale involved.** The raw archives
decompress in-memory (no filesystem extraction) directly into columnar
frames; every join, filter, and leg-resolution step operates on the full
frame rather than row-by-row Python. The heaviest single step — resolving
every quoted combo instrument's legs — is batched by *unique instrument*
rather than by row (the parse outcome is a pure function of an instrument's
symbol and listed maturity, so a market with 650,000+ raw combo rows reduces
to a few thousand unique instruments to actually resolve, then joins back
onto the full daily series). The full 42-market universe — several million
raw rows across outrights and every combo type — transforms end-to-end in
well under 15 minutes.

**Output (`Data/term_structure*.parquet`, 6 tables):** outrights, spreads,
butterflies, condors, averages, and packs, each leg-decomposed (every leg's
canonical root, contract symbol, expiry year, and expiry code broken out as
its own column) rather than left as an opaque symbol string — built so a
relative-value or carry signal can reason about *which* contracts it's
actually trading, not just a price series. A daily forward-capture job
(`yfinance`-sourced, since expired-contract history can't be retroactively
recovered from that source) keeps this current going forward; the
historical backfill fills in everything before that job started, with
Databento's data taking precedence on any date the two sources overlap.

## 3. Continuous futures curves — correctly back-adjusted

Momentum, breakout, and crossover signals need one continuous per-asset price
series, not a table of individual contracts. This project builds that series
directly from the term-structure data above (`src/data/continuous_curve.py`),
rather than relying on a vendor's opaque continuous ticker:

- **Roll detection by volume crossover**, with a short confirmation window so
  a single noisy day doesn't flip the designated front contract back and
  forth.
- **Ratio (proportional), not additive, back-adjustment.** An additive
  ("Panama") adjustment can push an old, heavily-adjusted segment through
  zero or negative — which isn't just cosmetically wrong, it creates a
  percentage-return singularity at the crossing point that corrupts every
  downstream return calculation for that stretch of history. A ratio
  adjustment is a product of positive numbers, so it's positive by
  construction, and it preserves percentage returns consistently across the
  full history — the correct choice for a project where every signal is
  built on percentage returns.

Both a raw (unadjusted, roll-dates marked) and a back-adjusted series are
kept side by side: the raw series feeds the Yang-Zhang volatility estimator
(which needs genuine OHLC ranges, not a smoothed adjustment), the
back-adjusted series feeds every return-based signal.

## 4. Data-quality handling — trusted era vs. legacy, not deletion

Not every era of every asset's history is equally trustworthy — Yahoo's
continuous-ticker construction and some Databento-side thin-contract prints
both carry known noise. Rather than deleting flagged data (which makes a
genuinely bad print indistinguishable from a holiday, and throws away
information that a rolling-window volatility or momentum estimator can still
use directionally), each asset is given a **trusted-since date**: the point
from which its term-structure coverage is genuine, exchange-quoted,
per-contract data. Everything from that date forward is the primary basis
for signal calibration; everything before it is retained as an
out-of-sample robustness check only — never used to calibrate or select a
signal, since it reflects a structurally different (pre-electronic,
single-series) market regime anyway.

## 5. Volatility

Yang-Zhang OHLC volatility (Yang & Zhang, 2000) — combining overnight,
open-to-close, and Rogers-Satchell range variance components — computed at
four horizons (21/63/126/252 days), strictly backward-looking, scheduled
daily immediately after the price pipeline it depends on.

## 6. Macro overlays — point-in-time correct by construction

Five independent macro sources feed the Value signal and a regime-context
layer: the US Treasury par yield curve, effective Fed funds and related
overnight reference rates, the NY Fed's Global Supply Chain Pressure Index,
a daily trade-policy-uncertainty index, CBOE VIX, and a 7-country CPI panel
(FRED for six countries, Japan sourced directly from its own government
statistics API after its FRED mirror was found to be running years stale).

**Every source here has a real publication lag**, and none of the raw files
distinguish "the date a value describes" from "the date it was actually
known" — joining a backtest directly against the raw date is a look-ahead
bug waiting to happen. `src/macro_point_in_time.py` provides `as_of(date)`
accessors that apply each source's confirmed publication lag, so a
backtested strategy can never see a macro value before it was actually
public. Built ahead of any consumer needing it, specifically so this class
of bug gets designed out rather than discovered after the fact.

## 7. Monitoring

A dedicated scheduled job computes pipeline health, coverage-vs-real-
exchange-calendar gaps, term-structure curve snapshots, and macro freshness,
writing small pre-computed artifacts that a 17-page Streamlit dashboard
reads and renders — the dashboard does no computation of its own for this
layer, so what it shows is exactly what the monitoring job measured.

## Known gaps

- Open interest is not yet pulled (requires event-level tick parsing, not
  a simple daily bar).
- ICE's spread/butterfly/condor schema is structurally different from CME's
  and not yet supported — ICE outrights transform normally, ICE combo
  instruments are skipped rather than guessed at.
- The 4 ICE softs' term-structure history is scoped to roughly the last two
  years, not the full available archive — a deliberate cost/scope tradeoff,
  not a technical limit.
- **Found 2026-07-28: for 3 of the 5 ICE softs (Coffee, Cotton, OrangeJuice),
  the scoped backfill above is missing rows for the contracts genuinely
  trading in 2023-2024** — the only contract-symbol present in
  `term_structure.parquet` on those early dates is a single far-dated one
  (e.g. Coffee's `KCN26.NYB`, a July-2026 contract, present back to
  2023-08-01 with volume=0). `continuous_curve.assign_front_contract()`
  didn't malfunction — it correctly picked "whichever contract has real
  data," there was just no genuine alternative to roll into, so it stayed
  parked on that one thin contract for 13-28 months (Coffee ~13mo, Cotton
  ~19mo, OrangeJuice ~28mo) before the daily forward-capture job started
  populating real near-term contracts. The gap extends past these 3 assets'
  own `trusted_since` cutoffs (`Data/asset_trusted_since.csv`), so the
  existing trusted-era mask doesn't fully cover it. Price still tracks the
  real commodity reasonably (checked directly against `Data/close.parquet` —
  no roll-jump artifacts, ratio back-adjustment works correctly), but volume
  is genuinely near-zero because that far-dated contract-month wasn't
  actively trading in the real world at those historical dates — not
  representative of genuine tradable liquidity. Sugar and Cocoa (the other 2
  of the 5 ICE softs) show no such defect. **Fix applied**:
  `src/data/universe.ICE_SOFTS_DATA_BLOCKED` excludes Coffee/Cotton/
  OrangeJuice from `get_liquid_universe()` independently of their computed
  ADV, so they stay excluded for the real, documented reason rather than as
  an accidental side-effect of the ADV threshold. Doesn't change any
  already-published momentum/breakout/crossover/portfolio result — all 3
  already failed the ADV floor under `continuous_curve`'s own (corrupted)
  volume anyway. Purchase-contingent, not a permanent liquidity judgment —
  remove from that list once the fuller Databento ICE history (confirmed
  available back to 2018-12-23, not yet purchased) is bought and the
  stuck-front pattern is reverified as gone.
- Classic cointegration/relative-value spreads (Corn/Wheat, Gold/Silver,
  Brent/WTI) have a rolling-window statistical foundation but no trading
  signal built on top yet.

## Overlap with a related project

A separate project researching an AIS-derived port-congestion signal against
commodity prices independently sources 11 of this project's markets (WTI,
Brent, and a 9-commodity basket) — no data-sharing is wired between the two
repos, by design, until that signal validates on its own terms.
