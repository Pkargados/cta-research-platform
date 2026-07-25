# Systematic Commodity & Macro Futures Research Platform

A from-scratch systematic futures research platform covering commodities, FX,
rates, and equity index futures — built to the standard of a production
research desk, not a coursework project. It researches trend, breakout,
short-term reversal, carry, and relative-value signals with volatility
targeting and covariance-based portfolio construction, on top of a real
daily data pipeline and a QA/monitoring dashboard.

**Target state:** a 42-market universe, six-plus independently-researched
signal families combined through a shared Book/Allocator portfolio layer,
with infrastructure eventually extending to live data ingestion and paper
trading. This README describes what is actually built today, not the
end-state — see [Current state](#current-state) for the honest line between
the two.

## Why this project looks the way it does

Most of the signal results below are weak, mixed, or net-negative after
costs. That is reported on purpose, not softened. The point of this project
is not "find a strategy that backtests well" — it's demonstrating the
research discipline that keeps a backtest honest: strict train/validation/test
splits with test touched once, no signal-spec selection by looking at its own
performance, no look-ahead in either the trading signal or the statistical
tests used to build it, and every negative or mixed result reported as found.
A handful of real, live-caught bugs (look-ahead in a sizing overlay, a
16-month unmonitored leverage blackout, a joint-completeness gate silently
starving a signal of data) are documented in [`WORKFLOW.md`](WORKFLOW.md) in
full, including what they looked like before they were caught — that record
is as much the point of this repo as any single Sharpe ratio in the table
below.

## Current state

| Component | Status |
|---|---|
| Core OHLCV data pipeline (`yfinance`, 41 of 42 assets) | Done, production quality |
| Term structure / carry data (Databento) | Done for 33 of 38 core assets (real spreads); 4 ICE softs on a labeled proxy |
| Yang-Zhang volatility (4 horizons) | Done |
| Macro/auxiliary data (yield curve, CPI, GSCPI, VIX, trade-policy uncertainty) | Collected, feeding the Value signal and a QA dashboard page |
| Time-series momentum | Done — rebuilt to match Moskowitz-Ooi-Pedersen (2012) exactly |
| Breakout (Donchian / Turtle Rules) | Done — weak/negative net-of-cost, reported honestly |
| Moving-average crossover | Done — mixed (one of three pairs consistently positive) |
| Short-term reversal (cross-sectional, VIX-conditioned) | Done — unprofitable net-of-cost across every spec tested |
| Carry (Koijen-Moskowitz-Pedersen-Vrugt 2018) | Done — genuinely mixed across four paper-matched specs |
| Cross-sectional momentum & Value (Asness-Moskowitz-Pedersen 2013) | Done — both weak/negative, reported as found |
| Portfolio construction (Ledoit-Wolf Σ, turnover-penalized optimizer, Book/Allocator) | First real pass run on 6 of ~20 Books — genuinely mixed vs. a naive blend |
| Risk metrics (historical VaR / Expected Shortfall) | First pass built, portfolio-level only |
| Hyperparameter tuning + multiple-testing correction (Bonferroni/FDR, then CPCV/PBO) | Built and run across all 20 Books — see [Methodology](#methodology-worth-reading) |
| Regime detection | Interface only, no classifier yet |
| Live data / paper trading | **Not started** |

Classic cointegration / relative-value spreads (Corn/Wheat, Gold/Silver,
Brent/WTI) have a rolling-window Engle-Granger foundation built but no
trading signal yet — deferred twice in favor of carry and cross-sectional
momentum, per direct instruction, not silently dropped (see `WORKFLOW.md`).

## Signal results (headline spec per family, gross Sharpe, train / validation / test)

| Family | Headline spec | Train | Validation | Test | Verdict |
|---|---|---:|---:|---:|---|
| Time-series momentum | 12-month lookback | 0.24 | 0.48 | 0.40 | Positive throughout — the one clean win |
| Breakout | Turtle System 1 (20d/10d) | — | — | — | Weak/negative net-of-cost; turnover ~50-60x/yr |
| Crossover | 50/200 (golden cross) | 0.14 | 0.37 | 0.27 | Most consistent of 3 pairs tested; others fade or sign-flip |
| Short-term reversal | Individual, 5-day lag | — | — | — | Net Sharpe -0.5 to -2.8 across all 6 specs; unprofitable |
| Carry (cross-sectional, 1-12mo) | Koijen et al. Eq. 19 | 0.33 | -0.36 | -0.06 | No spec robust across all three periods |
| Cross-sectional momentum | MOM2-12 | -0.34 | -1.39 | 0.04 | Weak/negative; validation spans the 2020 momentum crash |
| Value | Negative-5yr-return, asset-class-adjusted | -0.01 | 0.13 | -0.69 | Weak/negative |
| Combined portfolio (6-Book pilot, optimizer) | Ledoit-Wolf + turnover-penalized MVO | 0.46 | -1.21 | 0.09 | Beats a naive blend on test, loses on validation (2020-21) |

Every family has 1-6 parallel specs (not one cherry-picked winner) reported
side-by-side in the dashboard — see full per-spec numbers, methodology, and
every real bug found along the way in
[`CLAUDE.md`](CLAUDE.md#current-state-verified-against-source-not-the-resume-bullet)
and [`WORKFLOW.md`](WORKFLOW.md).

## Methodology (worth reading)

The most recent work on this project isn't a new signal — it's testing
whether the *hyperparameter tuning process itself* can be trusted:

1. **Per-Book sizing tuning** (`target_vol`, `max_weight`) was tested across
   all 20 Books with a two-stage Bonferroni + Benjamini-Hochberg FDR
   correction. Result: **0 of 19 evaluable Books' tuned parameters survive**
   — every Book keeps its flat default calibration.
2. That raised a real objection: a single fixed 2020-2021 validation window
   is thin evidence, and naive multiple-testing correction doesn't scale
   the way real multi-strategy shops operate. The fix, built and run this
   session: **Combinatorially Symmetric / Purged Cross-Validation**
   (Bailey, Borwein, López de Prado & Zhu, 2017 — implemented from the
   primary paper, not a summary) measures whether the *whole selection
   process* generalizes across ~70 independent recombinations of the full
   ~18-year history instead of one window.
3. Scaled to all 20 Books, this **independently corroborates** the original
   finding via a completely different method: mean Probability of Backtest
   Overfitting (PBO) across the roster is 0.45 (essentially a coin flip),
   with 10 of 19 Books above the 0.5 overfitting threshold. A 3-Book pilot
   run first looked much more reassuring (PBO 0.09-0.37) — the full
   20-Book run showed that pilot was, in hindsight, unrepresentative. That
   reversal is logged in full, not quietly corrected away.

Next steps (empirical-Bayes shrinkage, an effective-number-of-trials
correction, a Deflated Sharpe Ratio check) are scoped and sourced but not
yet built — see `WORKFLOW.md` Phase 7 for the full citation list and
reasoning.

## Architecture

```
src/
  data/          # panels, universe filters, continuous-curve construction, vol estimators, macro loaders
  signals/       # one module per signal family — pure functions, no optimizer dependency
  backtest/      # engine, performance stats, train/val/test splits, costs, CPCV
  portfolio/     # Book (one signal sleeve), Allocator (combines Books), optimizer, covariance, risk metrics
  regime/        # regime-conditioning interface (classifier not yet built)
research/        # driver scripts — where new signal/portfolio research is actually run
dashboard/       # Streamlit QA + strategy-performance + portfolio-construction pages
jobs/            # scheduled data-refresh entry points (Windows Task Scheduler)
databento/       # term-structure ingestion, transform, continuous-curve build
tests/           # pytest, 150+ tests, run before every methodology change
```

A `Book` owns one signal family's alpha, covariance, optimizer parameters,
and vol target end-to-end. An `Allocator` combines Books and would apply any
regime conditioning *before* the optimizer runs — vol targeting silently
cancels out any post-solve position scaling, a lesson carried over from an
earlier (retired) project rather than re-learned here. See `CLAUDE.md`'s
Architecture section for what does and doesn't carry over from that prior
build.

## Hard rules this project enforces on itself

(Full detail and the real incidents behind each one in `CLAUDE.md`.)

- Never edit the asset universe, or pick a signal spec, after looking at its
  own backtest performance.
- Never run a stationarity/cointegration test on the full sample — rolling
  or expanding window with a strict train/test boundary, always.
- Every signal is `shift(1)`'d before being used to trade "today."
- No faked carry from a residual-return proxy — label it missing/blocked
  instead, until a real term-structure source is found and verified.
- Continuous, vol-scaled signals only — binary signals were shown early on
  to underperform in this universe, and that finding isn't re-litigated
  without new evidence.

## Getting started

```bash
pip install pandas numpy scikit-learn statsmodels streamlit plotly matplotlib \
            yfinance requests pandas_market_calendars xlrd

# Dashboard (QA pages + per-signal strategy performance + portfolio construction)
streamlit run dashboard/app.py

# Full test suite
pytest tests/

# Re-run a signal family's research driver
python research/momentum.py
```

`Data/` is not checked into this repo (see `.gitignore`) — it's regenerated
by the jobs in `jobs/` and the notebooks in `notebooks/`, documented field-by-
field in [`DATA_SCHEMA.md`](DATA_SCHEMA.md). Databento-sourced term-structure
data requires a `DATABENTO_API_KEY` environment variable; the Japan CPI pull
requires a free `ESTAT_APP_ID`. Neither is required to run the signal
research or dashboard against already-collected data.

## Further reading

- [`CLAUDE.md`](CLAUDE.md) — full current-state table, hard rules, and the
  project's own working agreement with itself on what counts as a legitimate
  finding.
- [`WORKFLOW.md`](WORKFLOW.md) — the complete phased research log: every
  signal's exact construction, every real bug found and how it was fixed,
  every design decision and the reasoning behind it, in chronological order.
- [`DATA_SCHEMA.md`](DATA_SCHEMA.md) — full data inventory: what's collected,
  what's collected-but-unused, and how to source what's missing.

## Scope note

This repo is deliberately self-contained. A related project researching an
AIS-derived port-congestion signal against commodity prices is a fully
separate repository by design, and is not wired in here — see `CLAUDE.md`
for the integration plan if that signal ever validates on its own terms.
