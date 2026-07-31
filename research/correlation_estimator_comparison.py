"""
research/correlation_estimator_comparison.py — Diagnostic comparison of three
correlation estimators (rolling Pearson, EWMA Pearson, DCC-GARCH) across four
representative pairs, for the Technical Appendix's new Correlation Estimators
dashboard page. Per direct instruction, following straight from the
Multi-Strategy Portfolio page's DCC-GARCH sleeve-covariance wiring
(WORKFLOW.md decision #12) — this is the "between signals/Books" DCC-GARCH
escalation the earlier signal-level correlation work (`research/
signal_correlation.py`) already flagged as a next step, given the same
"Estimators" shape page 06 (Volatility Estimators) already established:
compare methods, don't pick a winner, surface where they agree and where
they don't.

Not new methodology — reuses, doesn't reimplement:
- `research/trend_correlation.py`'s exact data/signal construction for the
  three trend-family pairs (momentum_12mo, breakout_system1, crossover_50_200,
  daily standalone strategy returns) — imported as a module, not copy-pasted.
- `research/single_strategy_portfolios.py`'s exact Book construction for the
  fourth pair — the Trend/Carry weekly sleeve PnL, same construction
  `research/sleeve_risk_parity.py` already validated (decision #12).
- `portfolio.correlation.rolling_correlation`/`ewma_correlation` (the simple
  baseline, already used by both scripts above).
- `dcc_garch.garch.gjr_garch.fit_multivariate_gjr` + `dcc_garch.dcc.
  optimizer.fit` — the identical two-stage DCC-GARCH fit call `trend_
  correlation.py`'s own `dcc_report` and `portfolio.sleeve_covariance.
  dcc_garch_covariance` both already use.

The one genuinely new piece: neither prior script CACHED its DCC-GARCH
result (both only ever printed it) — a real per-pair MLE fit isn't
instant, so a dashboard page can't fit it live. This script precomputes and
caches both outputs (same never-recomputed-live convention as
`data.garch_volatility`'s own cache):
- Data/research/correlation_estimator_series.parquet — long-format
  (pair, estimator, date, value) rolling/EWMA/DCC-GARCH correlation paths.
- Data/research/correlation_estimator_summary.csv — one row per pair:
  full-sample correlation, each estimator's mean and its mean within two
  named stress windows (2020 COVID crash, 2022 inflation/rate shock — same
  windows `trend_correlation.py` already uses), and the DCC fit's own
  convergence flag.

Run: `python research/correlation_estimator_comparison.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Same two-line pattern (and the same shadow-avoidance ordering) as
# research/trend_correlation.py: src/ inserted AFTER the implicit research/
# (this script's own directory, already at sys.path[0] by default) so src/
# ends up first in resolution order, then the DCC sibling repo pushed to the
# very front — see dashboard/_single_strategy_pipeline.py's own docstring
# gotcha note for why this ordering matters (research/portfolio.py would
# otherwise shadow src/portfolio/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "DCC Garch Rompolis" / "src"))

import trend_correlation as tc
import single_strategy_portfolios as ssp
from portfolio.correlation import rolling_correlation, ewma_correlation

DAILY_ROLLING_WINDOW = tc.DAILY_ROLLING_WINDOW
DAILY_EWMA_HALFLIFE = tc.DAILY_EWMA_HALFLIFE
# Weekly analogues for the Trend/Carry sleeve pair - a labeled diagnostic
# default (~half a year of weekly observations), not a tuned choice, same
# discipline as this project's other unvalidated-but-reasonable window
# defaults (e.g. portfolio.covariance.DEFAULT_WINDOW).
WEEKLY_ROLLING_WINDOW = 26
WEEKLY_EWMA_HALFLIFE = 26

# Same two stress windows as research/trend_correlation.py's own
# STRESS_WINDOWS, with filesystem/column-safe keys instead of that dict's
# human-readable labels (which contain a space and a "/").
STRESS_WINDOWS = {
    "covid_2020": ("2020-02-15", "2020-04-30"),
    "shock_2022": ("2022-01-01", "2022-12-31"),
}

SERIES_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "correlation_estimator_series.parquet"
SUMMARY_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "correlation_estimator_summary.csv"


def _dcc_correlation_path(series_a: pd.Series, series_b: pd.Series):
    """Fit DCC-GARCH on two aligned return series, return (R[:, 0, 1] conditional
    correlation path, converged flag) — the identical two-stage fit call
    `trend_correlation.py`'s own `dcc_report` and `portfolio.sleeve_covariance.
    dcc_garch_covariance` both already use."""
    from dcc_garch.garch.gjr_garch import fit_multivariate_gjr
    from dcc_garch.dcc.optimizer import fit as fit_dcc

    aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1, join="inner").dropna()
    fit = fit_multivariate_gjr(aligned.values)
    dcc = fit_dcc(fit["Z"], fit["sigmas"], model="DCC")
    path = pd.Series(dcc["R"][:, 0, 1], index=aligned.index)
    return path, bool(dcc["converged"])


def build_pairs() -> dict:
    """Returns {pair_name: (series_a, series_b, rolling_window, ewma_halflife)}
    for all four representative pairs — three trend-family (daily standalone
    strategy returns) plus one Trend/Carry sleeve (weekly Book PnL)."""
    adj, raw = tc.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = tc.build_vol(raw)
    trend_returns = tc.build_trend_returns(close, returns, vol)

    s_adj, s_raw, s_included, s_sectors = ssp.load_and_prepare_data()
    s_close = s_adj["close"]
    s_returns = s_close.pct_change(fill_method=None)
    s_vol = ssp.build_vol(s_raw)
    carry_panel, _ = ssp.build_carry_panel(s_included)

    trend_flavors = ssp.build_trend_flavors(s_close, s_vol, s_returns)
    trend_book = ssp.build_book("trend_tsmom_alone", trend_flavors["tsmom_alone"], s_returns, vol_estimator="garch")
    trend_result = trend_book.run(s_returns)

    carry_flavors = ssp.build_all_carry_signals(carry_panel, s_sectors)
    carry_book = ssp.build_book("carry_carry_timing_zero", carry_flavors["carry_timing_zero"], s_returns, vol_estimator="garch")
    carry_result = carry_book.run(s_returns)

    names = list(trend_returns.keys())
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs[f"{names[i]} vs. {names[j]}"] = (
                trend_returns[names[i]], trend_returns[names[j]], DAILY_ROLLING_WINDOW, DAILY_EWMA_HALFLIFE,
            )
    pairs["Trend sleeve vs. Carry sleeve"] = (
        trend_result["pnl"], carry_result["pnl"], WEEKLY_ROLLING_WINDOW, WEEKLY_EWMA_HALFLIFE,
    )
    return pairs


def main():
    pairs = build_pairs()
    print(f"Pairs: {list(pairs.keys())}")

    series_rows = []
    summary_rows = []
    for pair_name, (series_a, series_b, window, halflife) in pairs.items():
        print(f"\n=== {pair_name} ===")
        rolling = rolling_correlation(series_a, series_b, window=window)
        ewma = ewma_correlation(series_a, series_b, halflife=halflife)
        dcc_path, converged = _dcc_correlation_path(series_a, series_b)
        print(f"  DCC-GARCH converged: {converged}")

        aligned_full = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1, join="inner").dropna()
        full_sample_corr = float(aligned_full["a"].corr(aligned_full["b"]))
        print(f"  full-sample corr: {full_sample_corr:.3f}  (n={len(aligned_full)})")

        estimators = {"rolling": rolling, "ewma": ewma, "dcc_garch": dcc_path}
        for est_name, path in estimators.items():
            clean = path.dropna()
            df = clean.rename("value").reset_index()
            df.columns = ["date", "value"]
            df["pair"] = pair_name
            df["estimator"] = est_name
            series_rows.append(df)

        row = {"pair": pair_name, "n_obs": len(aligned_full), "full_sample_corr": full_sample_corr, "dcc_converged": converged}
        for est_name, path in estimators.items():
            row[f"{est_name}_mean"] = float(path.mean()) if path.notna().any() else np.nan
            for window_id, (start, end) in STRESS_WINDOWS.items():
                window_mean = path.loc[start:end].mean()
                row[f"{est_name}_{window_id}"] = float(window_mean) if pd.notna(window_mean) else np.nan
        summary_rows.append(row)

    series_long = pd.concat(series_rows, ignore_index=True)
    SERIES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    series_long.to_parquet(SERIES_CACHE_PATH)
    print(f"\nSaved {len(series_long)} rows to {SERIES_CACHE_PATH}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CACHE_PATH, index=False)
    print(f"Saved summary to {SUMMARY_CACHE_PATH}")


if __name__ == "__main__":
    main()
