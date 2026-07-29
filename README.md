# Systematic Commodity & Macro Futures Research Platform

**[Live demo →](https://cta-research-platform-fflp3hcjqbvbss9mpnqymz.streamlit.app/)**

A systematic futures research platform spanning 42 markets across commodities,
FX, rates, and equity indices — built on a self-collected, two-source daily
data pipeline, seven independently-researched signal families, and a
covariance-aware portfolio construction layer with rigorous overfitting
controls.

This README describes what's actually built and validated today. Results are
reported honestly, including where signals are weak or mixed — the platform is
designed to make that distinction impossible to fudge, not to hide it.

## Data infrastructure

The foundation is a real market-data pipeline, not a single downloaded CSV:

- **Daily OHLCV across 41 of 42 markets** via a scheduled `yfinance` pipeline,
  with a coverage-audited, gap-masked panel feeding every downstream signal.
- **Daily per-contract term-structure data via Databento** (CME Globex
  `GLBX.MDP3` and ICE Futures US `IFUS.IMPACT` raw daily archives, `ohlcv-1d`
  — not tick/trade-level data) for 38 core assets — outright
  contracts plus every quoted spread, butterfly, condor, average, and pack
  instrument, decoded from raw exchange feeds rather than a vendor's
  pre-cleaned panel.
- **A fully vectorized `polars` transform pipeline** that resolves
  decade-ambiguous contract symbols via an empirically-derived anchor-leg
  algorithm, joins millions of raw daily bars against instrument definitions,
  and rebuilds all 6 term-structure tables for the full 42-market universe
  end-to-end in under 15 minutes.
- **A continuous futures curve** built with volume-crossover roll detection
  (with a confirmation-day buffer to avoid single-day noise flipping the front
  contract) and **ratio (proportional), not additive, back-adjustment** — the
  correct construction for return-based signal research, not the naive version
  that silently distorts historical returns.
- **Macro overlays** (yield curve, CPI, GSCPI, VIX, trade-policy uncertainty)
  feeding the Value signal and a point-in-time-correctness QA layer.
- An **18-page Streamlit dashboard** covering data QA, per-signal strategy
  performance, portfolio-construction health, and macro exploration.

## Signal research

Seven signal families, each implemented as pure functions with no optimizer
dependency, matched directly against their source papers rather than built
from memory:

| Family | Source | Construction |
|---|---|---|
| Time-series momentum | Moskowitz-Ooi-Pedersen (2012) | 12-month lookback, vol-targeted sign signal, full 8×8 lookback/holding robustness grid |
| Breakout | Classic Turtle Rules | Dual-channel Donchian system (20d/10d and 55d/20d), per-asset state machine |
| Moving-average crossover | Golden/death cross | Three SMA pairs (50/100, 50/200, 100/200) |
| Short-term reversal | Lehmann (1990), Nagel (2011) | Cross-sectional, sector-demeaned, VIX-conditioned sizing overlay with a Newey-West HAC significance check |
| Carry | Koijen-Moskowitz-Pedersen-Vrugt (2018) | Rank-weighted cross-sectional carry and carry-timing, matched to the paper's exact monthly-rebalance construction |
| Cross-sectional momentum | Asness-Moskowitz-Pedersen (2013) | Rank-weighted 12-month-skip-1-month momentum |
| Value | Asness-Moskowitz-Pedersen (2013) | Asset-class-specific construction — 5-year return default, yield-change for bonds, PPP-adjusted real FX return for currencies |

**Every backtest enforces the same discipline**: strict train / validation /
test splits with the test period touched once, no signal-spec selection based
on its own performance, and every signal `shift(1)`'d before being used to
trade "today." Results below are gross Sharpe, train / validation / test:

| Family | Headline spec | Train | Validation | Test |
|---|---|---:|---:|---:|
| Time-series momentum | 12-month lookback | 0.24 | 0.48 | 0.40 |
| Crossover | 50/200 (golden cross) | 0.14 | 0.37 | 0.27 |
| Carry (cross-sectional, 1-12mo) | Koijen et al. Eq. 19 | 0.33 | -0.36 | -0.06 |
| Cross-sectional momentum | MOM2-12 | -0.34 | -1.39 | 0.04 |
| Value | Negative-5yr-return, asset-class-adjusted | -0.01 | 0.13 | -0.69 |

Momentum is the one consistently positive family; the rest are genuinely
mixed. That's reported as found — the platform's value is in the discipline
that produced these numbers, not in engineering a cleaner-looking table.

## Single Strategy Portfolios

The 42-market research universe is now a defined two-Book investment mandate
— directional trend and curve-based carry, chosen because trend is the one
consistently-positive family above and carry has genuine (if mixed) evidence,
while relative-value stays research-only until its own trading signal exists
(only a rolling-window cointegration foundation today, no z-score entry/exit
built yet).

Each Book's construction was chosen by a small, validation-selected bake-off
across economically distinct alternatives — not a numerical hyperparameter
grid, and not the highest-Sharpe pick either:

| Book | Winning construction | Why | Train | Validation | Test |
|---|---|---|---:|---:|---:|
| Trend | TSMOM alone | Beat every blend tried (equal-weight, tilted, IC-weighted, risk-parity, a confirmation-filter gate, and a conviction-deadband construction) — TSMOM and the 50/200 crossover are too correlated (0.65 full-sample, DCC-GARCH) to diversify each other, so blending only diluted the stronger signal | 0.32 | 0.59 | 1.36 |
| Carry | Carry-timing (zero reference) | Best of the carry family's 4 existing specs on validation, though still negative there — consistent with carry's well-documented COVID-era underperformance | -0.37 | -0.94 | 0.29 |
| Combined (equal Book risk) | — | — | -0.14 | -0.51 | 0.87 |

Both Books use GJR-GARCH (not EWMA) to forecast their own realized-PnL
volatility for vol-targeting — tested head-to-head and adopted after GARCH
cut forecast loss (QLIKE) roughly 60-70% versus EWMA, the same asymmetric-
shock-response advantage GARCH already showed at the asset-price level, now
confirmed at the portfolio level too rather than assumed to transfer.

## Portfolio construction

- **`Book` / `Allocator` architecture**: each signal family owns its own
  alpha, covariance estimate, and vol target end-to-end; the Allocator
  combines Books and applies any regime conditioning *before* the optimizer
  runs (vol targeting silently cancels out any post-solve position scaling).
- **Ledoit-Wolf shrinkage covariance**, estimated on a rolling window with a
  minimum-coverage gate to avoid pricing risk off a stale, sparsely-populated
  cross-section.
- **Turnover-penalized mean-variance optimization** with position-size caps
  and a dollar-neutral constraint.
- **GJR-GARCH vol-targeting** (opt-in per Book, `vol_estimator="garch"`) —
  refits every 20 rebalance periods on realized PnL through the previous
  period only, filters forward with fixed parameters between refits, falling
  back to a plain EWMA recursion during warmup or on any fit failure. The
  EWMA default is preserved for every other Book in the codebase — this is
  additive, not a silent behavior change to already-published results.
- **Historical VaR / Expected Shortfall** at the combined-portfolio level.
- **Two distinct selection disciplines, not one**: choosing *which
  construction* to trade (the Single Strategy Portfolios bake-off above) is a
  small number of economically distinct alternatives selected on validation —
  a different statistical situation from tuning *sizing parameters* of an
  already-chosen construction across many Books, which is where the
  Bonferroni/FDR + CPCV machinery below applies. Per-Book sizing parameters
  (`target_vol`, `max_weight`) were tuned across all 20 Books with a
  two-stage Bonferroni + Benjamini-Hochberg FDR correction, then
  independently cross-checked with **Combinatorially Symmetric / Purged
  Cross-Validation** (Bailey, Borwein, López de Prado & Zhu, 2017) measuring
  Probability of Backtest Overfitting across ~70 recombinations of the full
  history. Both methods agree: none of the tuned parameter sets clear the bar
  for adoption over the flat default calibration — a result that held up
  under two independent statistical tests rather than being taken on faith
  from the first one.

## References

Every signal and methodology choice here is matched directly against its
source paper, not implemented from memory or a secondary summary:

| Paper | Used for |
|---|---|
| Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012), "Time Series Momentum," *Journal of Financial Economics* | Time-series momentum: 12-month lookback, vol-targeted sizing |
| Lehmann, B. (1990), "Fads, Martingales, and Market Efficiency," *Quarterly Journal of Economics* | Short-term reversal: cross-sectional, sector-demeaned construction |
| Nagel, S. (2011), "Evaporating Liquidity" | Short-term reversal: VIX-conditioned sizing overlay |
| Blitz, D., van der Grient, B., & Honarvar, I. (2023), "Reversing the Trend" | Short-term reversal: validates the lookback/sector-demean design choice |
| Koijen, R., Moskowitz, T., Pedersen, L. H., & Vrugt, E. (2018), "Carry," *Journal of Financial Economics* | Carry and carry-timing signals |
| Asness, C., Moskowitz, T., & Pedersen, L. H. (2013), "Value and Momentum Everywhere," *Journal of Finance* | Cross-sectional momentum and Value signals (both from this paper) |
| Fitzgibbons, S., Hecht, J., McQuinn, J., & Serban, A. (2017), "Portfolio Construction Matters," AQR | Mix-vs-integrate comparison for combining Value and cross-sectional momentum |
| Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017), "The Probability of Backtest Overfitting," *Journal of Computational Finance* 20(4) | Combinatorially Symmetric / Purged Cross-Validation (CPCV), used to test whether hyperparameter tuning generalizes across history rather than overfitting one window |

Background reading, informing design but not yet driving a shipped feature:

| Paper | Context |
|---|---|
| Kelly, B., Pruitt, S., & Su, Y., "Characteristics are Covariances" (IPCA) | Candidate covariance-estimation approach alongside DCC-GARCH |
| "Market Regime Identification for a Production CTA Research Platform" | Background for the not-yet-built regime classifier |
| "Portfolio Construction for CTA and Managed Futures Strategies" | General background for the portfolio-construction layer |

## Architecture

```
src/
  data/          # panels, universe filters, the polars-vectorized Databento transform, continuous-curve construction, vol estimators, macro loaders
  signals/       # one module per signal family — pure functions, no optimizer dependency
  backtest/      # engine, performance stats, train/val/test splits, costs, CPCV
  portfolio/     # Book (one signal sleeve), Allocator (combines Books), optimizer, covariance, risk metrics
  regime/        # regime-conditioning interface (classifier not yet built)
research/        # driver scripts — where new signal/portfolio research is actually run
dashboard/       # Streamlit QA + strategy-performance + portfolio-construction pages
jobs/            # scheduled data-refresh entry points (Windows Task Scheduler)
databento/       # thin entry-point scripts: job submission/retry, raw-archive backup, transform/curve-build drivers
tests/           # pytest, 246+ tests
```

## Testing

In a systematic trading platform, an untested backtest is a liability, not a
convenience: a single silent lookahead bug or an unhandled NaN can make a
strategy look profitable when it isn't. The 246-test suite (`tests/`, one
file per `src/` module) exists to make that class of error structurally
hard to ship:

- **No-lookahead guarantees** — confirming a signal is genuinely `shift(1)`'d
  before being used to trade "today," and that a rolling/expanding statistic
  never sees a future value. This is the single most common way a backtest
  silently lies, so it's checked directly rather than trusted by convention.
- **Numerical correctness** — a function's output checked against a manual,
  hand-computed answer (Sharpe ratio, Yang-Zhang volatility, back-adjustment
  ratio) on a small constructed input where the right answer is known in
  advance, not just "does it run without crashing."
- **Edge-case handling** — sparse trading calendars, all-NaN inputs,
  degenerate cases (constant price series, single-member sectors, empty date
  ranges) — the boundary conditions real market data hits that a happy-path
  test never would.
- **Regression protection** — once a specific failure mode is fixed, a test
  locks in that the fix stays fixed.

## Current state

| Component | Status |
|---|---|
| Core OHLCV data pipeline (41 of 42 assets) | Done, production quality |
| Term structure / carry data (Databento) | Done for 33 of 38 core assets (real spreads); 4 ICE softs on a labeled proxy |
| Continuous futures curves (42 assets) | Done — ratio back-adjusted, roll dates marked |
| Signal research (7 families) | Done, see table above |
| Portfolio mandate | Decided — two-Book: Trend + Carry, Relative Value deferred until its own trading signal exists |
| Single Strategy Portfolios (Trend Book, Carry Book) | Built and validation-selected, see table above |
| Portfolio construction (Ledoit-Wolf, optimizer, Book/Allocator) | Machinery built; first full pass was a 6-Book pilot, since superseded by the two selected Single Strategy Portfolios |
| Book-level vol-targeting (GJR-GARCH vs. EWMA) | Tested and decided — GARCH adopted for the two selected Books, opt-in, other Books unchanged |
| Hyperparameter tuning + overfitting controls (Bonferroni/FDR, CPCV/PBO) | Built and run across all 20 Books (sizing-parameter tuning only — construction selection uses a separate, smaller bake-off, see above) |
| Risk metrics (historical VaR / Expected Shortfall) | Built, portfolio-level |
| Regime detection | Interface only, no classifier yet |
| Live data / paper trading | Not started |

Classic cointegration / relative-value spreads (Corn/Wheat, Gold/Silver,
Brent/WTI) have a rolling-window Engle-Granger foundation but no trading
signal yet — sequenced after the signal families and portfolio-construction
work above.

## Getting started

```bash
pip install pandas numpy polars scikit-learn statsmodels streamlit plotly matplotlib \
            yfinance requests pandas_market_calendars xlrd

# Dashboard (QA pages + per-signal strategy performance + portfolio construction)
streamlit run dashboard/app.py

# Full test suite
pytest tests/

# Re-run a signal family's research driver
python research/momentum.py
```

`Data/` is not checked into this repo — it's regenerated by the jobs in
`jobs/` and documented field-by-field in [`DATA_SCHEMA.md`](DATA_SCHEMA.md).
Databento-sourced term-structure data requires a `DATABENTO_API_KEY`
environment variable; neither that nor a Japan CPI `ESTAT_APP_ID` is required
to run the signal research or dashboard against already-collected data.

## Further reading

- [`DATA_SCHEMA.md`](DATA_SCHEMA.md) — full data inventory: what's collected,
  what's collected-but-unused, and how to source what's missing.

## Scope note

This repo is deliberately self-contained. A related project researching an
AIS-derived port-congestion signal against commodity prices is a fully
separate repository by design, and is not wired in here.
