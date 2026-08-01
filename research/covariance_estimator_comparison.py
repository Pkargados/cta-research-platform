"""
research/covariance_estimator_comparison.py — Diagnostic comparison of five
covariance estimators (rolling sample, Ledoit-Wolf [current Book default],
Gerber statistic at c=0.5/0.7/0.9) for the Technical Appendix's new
Covariance Estimators dashboard page (25). WORKFLOW.md's "Gerber statistic
covariance" plan, Phase 7 — step 2 of 4. Same "compare, don't pick a
winner" shape as page 06 (volatility) and page 22 (correlation), and the
same diagnostic-only scope: nothing here swaps any live Book's actual
optimizer covariance (see the plan's own explicit scope boundary).

Universe: Trend's own already-adopted, compressed universe (WORKFLOW.md
decision #13, `data.universe.compress_for_family(included, "trend")`) — no
new universe invented for this comparison, per the plan. Further restricted
to assets with >= 90% overall non-NaN return coverage, the same threshold
`single_strategy_portfolios.py`'s own `_active_columns` already uses
(reused here as a plain returns-coverage filter, since this diagnostic
isn't tied to any one signal's alpha). This extra filter is necessary, not
cosmetic: checked directly, `build_cov_dict`'s Ledoit-Wolf fit produces
ZERO usable weekly dates on the raw 31-asset trend universe (it drops an
entire window row if ANY asset is missing that date, and several of those
31 assets are only ~60% covered — the union of everyone's scattered gaps
eats every window). Gerber's own per-pair T_ij tolerance does noticeably
better on that same raw panel (800 vs. 0 dates, confirmed live) — but
restricting to the 90%-covered ~21-asset subset here makes the comparison
apples-to-apples across all five estimators, not there to flatter
Ledoit-Wolf.

Cadence: weekly (`COV_FREQ="W-FRI"`), matching `Book`'s own cadence — NOT
monthly like the Gerber paper itself, a disclosed difference (see
WORKFLOW.md's plan). Window: 252 trading days, matching
`portfolio.covariance.DEFAULT_WINDOW`, identical across every estimator
compared here so none is confounded by a different lookback.

Forecast-accuracy metric — the multivariate analogue of page 06's own
QLIKE-against-realized-variance test (Patton 2011), per the plan: at each
formation date, form the GLOBAL MINIMUM-VARIANCE portfolio implied by that
estimator's own Sigma (closed form, w* = Sigma^-1 1 / (1' Sigma^-1 1)), then
compare that portfolio's FORECAST variance (w*' Sigma w*) against its
REALIZED variance over the following week (mean squared daily portfolio
return, held at w*, `shift(1)`-safe by construction — w* is formed at
`date` using only data through `date`, applied to returns strictly after
`date` through the next rebalance date). QLIKE = realized/forecast -
log(realized/forecast) - 1, zero at a perfect forecast, same formula as
page 06. Realized variance from ~5 daily observations per week is
genuinely thin (flagged, not hidden — the same kind of small-sample caveat
`risk_metrics.py`'s own monthly-PnL VaR/ES already carries), but it's the
same holding period every estimator is scored on, so the comparison
between them is still fair even if any single number is noisy.

Secondary diagnostic, matching page 22's style: pairwise correlation time
series (correlation IMPLIED by each estimator's own Sigma, not raw
returns) for four representative cross-sector pairs, viewed across the
same two stress windows `research/trend_correlation.py`/
`correlation_estimator_comparison.py` already use.

Cached to Data/research/ (never recomputed live by the dashboard page,
same convention as pages 06/22):
- covariance_estimator_qlike.parquet — long format (date, estimator,
  forecast_var, realized_var, qlike, condition_number).
- covariance_estimator_summary.csv — one row per estimator (n_dates,
  mean_qlike, win_rate, mean_condition_number).
- covariance_estimator_pairwise.parquet — long format (date, estimator,
  pair, correlation).

Run: `python research/covariance_estimator_comparison.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import single_strategy_portfolios as ssp
from data.universe import compress_for_family
from portfolio.covariance import build_cov_dict, real_period_end_dates, DEFAULT_MIN_FRAC
from portfolio.gerber_covariance import build_gerber_cov_dict, drop_until_complete
from correlation_estimator_comparison import STRESS_WINDOWS

COV_WINDOW = 252
COV_FREQ = "W-FRI"
MIN_VALID_FRAC = 0.90  # matches single_strategy_portfolios.py's own _active_columns coverage gate
GERBER_THRESHOLDS = (0.5, 0.7, 0.9)

# Four representative cross-sector pairs, same "reuse an existing sensible
# choice, don't invent a new one for its own sake" spirit as page 22's own
# pair selection - one pair per broad sector present in the active universe.
REPRESENTATIVE_PAIRS = [
    ("Gold", "Silver"), ("WTI Crude", "HeatingOil"), ("US_10Y", "US_30Y"), ("EURUSD", "JPYUSD"),
]

QLIKE_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "covariance_estimator_qlike.parquet"
SUMMARY_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "covariance_estimator_summary.csv"
PAIRWISE_CACHE_PATH = Path(__file__).resolve().parent.parent / "Data" / "research" / "covariance_estimator_pairwise.parquet"


def _active_columns(returns_df: pd.DataFrame, universe: list, min_valid_frac: float = MIN_VALID_FRAC) -> list:
    """Restrict `universe` to assets with >= `min_valid_frac` non-NaN
    coverage across the full `returns_df` history - the returns-coverage
    half of `single_strategy_portfolios.py`'s own `_active_columns` (no
    alpha_df here, since this diagnostic isn't tied to one signal)."""
    valid_frac = returns_df[universe].notna().mean()
    return [c for c in universe if valid_frac.get(c, 0.0) >= min_valid_frac]


def build_sample_cov_dict(returns_df: pd.DataFrame, window: int = COV_WINDOW, freq: str = COV_FREQ,
                           min_frac: float = DEFAULT_MIN_FRAC) -> dict:
    """Plain rolling sample covariance (no shrinkage) - the cheap baseline
    both the Gerber paper and page 06's own vol-estimator comparison score
    the more sophisticated estimators against. Identical warmup/min_frac
    gate to `build_cov_dict` so it's skipped on exactly the same dates
    Ledoit-Wolf would skip, for a clean side-by-side."""
    min_rows = max(2, int(window * min_frac))
    reb_dates = real_period_end_dates(returns_df.index, freq)
    cov_dict = {}
    for date in reb_dates:
        window_data = returns_df.loc[:date].iloc[-window:]
        if len(window_data) < window:
            continue
        clean = window_data.dropna(how="any")
        if len(clean) < min_rows:
            continue
        cov_dict[date] = clean.cov()
    return cov_dict


def _min_variance_weights(Sigma: pd.DataFrame) -> pd.Series:
    """Closed-form global minimum-variance portfolio: w* = Sigma^-1 1 /
    (1' Sigma^-1 1) - fully invested, long-short allowed, no other
    constraint (a diagnostic scoring tool, not a tradeable portfolio -
    `pinv` rather than `inv` since a near-singular Sigma is an expected
    occurrence on this panel, not a corner case to crash on)."""
    ones = np.ones(len(Sigma))
    raw = np.linalg.pinv(Sigma.values) @ ones
    denom = ones @ raw
    if denom == 0 or not np.isfinite(denom):
        return pd.Series(np.nan, index=Sigma.index)
    return pd.Series(raw / denom, index=Sigma.index)


def _qlike(realized: float, forecast: float) -> float:
    if not (np.isfinite(realized) and np.isfinite(forecast)) or realized <= 0 or forecast <= 0:
        return np.nan
    ratio = realized / forecast
    return float(ratio - np.log(ratio) - 1.0)


def _forecast_accuracy(cov_dict: dict, returns_df: pd.DataFrame, common_dates: list) -> pd.DataFrame:
    """One row per formation date in `common_dates` (skipping the last,
    which has no following period to realize against): forecast/realized
    variance of that date's global min-variance portfolio, QLIKE, and
    Sigma's own condition number (a raw stability diagnostic, not scored)."""
    rows = []
    for i in range(len(common_dates) - 1):
        date, next_date = common_dates[i], common_dates[i + 1]
        Sigma = drop_until_complete(cov_dict[date])
        if len(Sigma) < 2:
            continue
        w = _min_variance_weights(Sigma)
        if w.isna().any():
            continue
        forecast_var = float(w.to_numpy() @ Sigma.to_numpy() @ w.to_numpy())

        realized_window = returns_df.loc[date:next_date, w.index].iloc[1:].dropna(how="any")
        if len(realized_window) == 0:
            continue
        port_returns = realized_window.to_numpy() @ w.to_numpy()
        realized_var = float(np.mean(port_returns ** 2))

        rows.append({
            "date": date, "forecast_var": forecast_var, "realized_var": realized_var,
            "qlike": _qlike(realized_var, forecast_var),
            "condition_number": float(np.linalg.cond(Sigma.to_numpy())),
        })
    return pd.DataFrame(rows)


def _pairwise_correlations(cov_dict: dict, active: list, common_dates: list) -> list:
    rows = []
    for a, b in REPRESENTATIVE_PAIRS:
        if a not in active or b not in active:
            continue
        for date in common_dates:
            Sigma = cov_dict[date]
            if a not in Sigma.index or b not in Sigma.index:
                continue
            cov_ab, var_a, var_b = Sigma.loc[a, b], Sigma.loc[a, a], Sigma.loc[b, b]
            if not (np.isfinite(cov_ab) and np.isfinite(var_a) and np.isfinite(var_b)) or var_a <= 0 or var_b <= 0:
                continue
            rows.append({
                "date": date, "pair": f"{a} vs. {b}",
                "correlation": float(cov_ab / np.sqrt(var_a * var_b)),
            })
    return rows


def main():
    print("Loading data and building Trend's compressed universe...")
    adj, raw, included, sectors = ssp.load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    trend_universe = compress_for_family(included, "trend")
    active = _active_columns(returns, trend_universe)
    print(f"Active universe ({MIN_VALID_FRAC:.0%}+ coverage): {len(active)} of {len(trend_universe)} trend-universe assets")
    print(f"  {active}")
    r = returns[active]

    print("\nBuilding covariance estimator dicts (weekly cadence, 252-day window)...")
    cov_dicts = {
        "sample": build_sample_cov_dict(r),
        "ledoit_wolf": build_cov_dict(r, window=COV_WINDOW, freq=COV_FREQ),
    }
    for c in GERBER_THRESHOLDS:
        name = f"gerber_c{c:.1f}".replace(".", "")
        cov_dicts[name] = build_gerber_cov_dict(r, window=COV_WINDOW, freq=COV_FREQ, c=c)
    for name, d in cov_dicts.items():
        print(f"  {name}: {len(d)} dates")

    common_dates = sorted(set.intersection(*(set(d.keys()) for d in cov_dicts.values())))
    print(f"\nCommon dates across all {len(cov_dicts)} estimators: {len(common_dates)}")
    if len(common_dates) < 20:
        raise RuntimeError(f"Only {len(common_dates)} common dates across estimators - too few for a meaningful comparison, investigate before proceeding.")

    print("\nForecast-accuracy pass (global min-variance portfolio, QLIKE vs. realized)...")
    qlike_rows = []
    for name, cov_dict in cov_dicts.items():
        acc = _forecast_accuracy(cov_dict, r, common_dates)
        acc["estimator"] = name
        print(f"  {name}: {len(acc)} scored dates, mean QLIKE = {acc['qlike'].mean():.4f}")
        qlike_rows.append(acc)
    qlike_long = pd.concat(qlike_rows, ignore_index=True)

    print("\nPairwise correlation diagnostic...")
    pairwise_rows = []
    for name, cov_dict in cov_dicts.items():
        pairs_for_estimator = _pairwise_correlations(cov_dict, active, common_dates)
        for row in pairs_for_estimator:
            row["estimator"] = name
        pairwise_rows.extend(pairs_for_estimator)
    pairwise_long = pd.DataFrame(pairwise_rows)

    print("\nSummarizing...")
    win_matrix = qlike_long.pivot_table(index="date", columns="estimator", values="qlike")
    win_matrix = win_matrix.dropna(how="any")  # only score wins on dates every estimator has a real number
    wins = win_matrix.idxmin(axis=1).value_counts() if len(win_matrix) else pd.Series(dtype=int)

    summary_rows = []
    for name in cov_dicts:
        sub = qlike_long[qlike_long["estimator"] == name]
        summary_rows.append({
            "estimator": name,
            "n_dates": int(len(sub)),
            "mean_qlike": float(sub["qlike"].mean()),
            "win_rate": float(wins.get(name, 0) / len(win_matrix)) if len(win_matrix) else np.nan,
            "mean_condition_number": float(sub["condition_number"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    QLIKE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    qlike_long.to_parquet(QLIKE_CACHE_PATH)
    summary.to_csv(SUMMARY_CACHE_PATH, index=False)
    pairwise_long.to_parquet(PAIRWISE_CACHE_PATH)
    print(f"\nSaved {len(qlike_long)} qlike rows to {QLIKE_CACHE_PATH}")
    print(f"Saved summary to {SUMMARY_CACHE_PATH}")
    print(f"Saved {len(pairwise_long)} pairwise rows to {PAIRWISE_CACHE_PATH}")


if __name__ == "__main__":
    main()
