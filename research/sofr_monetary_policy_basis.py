"""
research/sofr_monetary_policy_basis.py — Monetary-policy basis (WORKFLOW.md
§11e item #2): compares SOFR futures' currently-priced forward rate against
the REALIZED daily SOFR rate, testing whether the gap has genuine, tradeable
information. Not a Fed-policy view — a mechanical bet on the well-established,
broad fixed-income principle that forward/futures-implied rates embed a
term/risk premium and are therefore a biased-HIGH predictor of the true
expected future short rate (the same general logic behind bond risk premia
and this project's own Carry family, applied here to a genuinely different
input: SOFR futures vs. the REALIZED rate series, not a calendar spread
between two futures - `data.macro.load_overnight_rate` (new this session)
is data never used by any prior signal).

**Construction and sign, fixed BEFORE looking at any backtest result**
(CLAUDE.md Hard Rule 1 - decide before performance, exactly the discipline
that rule exists to enforce):

    implied_rate_t  = 100 - SR3 front contract's RAW (not back-adjusted)
                      close - the genuine point-in-time market-implied
                      forward rate. The back-adjusted series is for RETURN
                      construction only, not for reading off a real
                      historical rate level (same distinction already
                      established for carry vs. the continuous curve).
    realized_rate_t = today's actual daily SOFR print.
    basis_t         = implied_rate_t - realized_rate_t
                      (>0: market currently prices a HIKE; <0: a CUT)

    signal_t = sign(basis_t)  ->  LONG SR3 when a hike is priced, SHORT when
    a cut is priced. Mechanically: being long SR3 profits if the eventual
    REALIZED rate for that quarter comes in BELOW today's implied level -
    i.e. this signal bets that priced-in moves (hikes or cuts) are somewhat
    overstated relative to what's eventually realized, consistent with a
    persistent term/risk premium rather than a forecast of Fed policy
    itself. This is the OPPOSITE of "trend with the priced-in path."

Two parallel specs, no headline pick (same discipline as Carry's own
`carry_timing_zero` vs. continuous, CLAUDE.md Rule 5's logged exception
precedent): `basis_timing` (simple ±1 direction, unscaled) and
`basis_continuous` (basis z-scored over a rolling 252-day window - a
monetary-policy regime persists over a multi-month/year horizon, unlike a
commodity mean-reversion spread's much shorter half-life, hence a longer
window than the RV sleeve's own 63-day default - then vol-targeted, this
project's usual house style). Both tested at daily AND weekly rebalancing
(the RV sleeve's own precedent: daily first as the a-priori default since
this is a fast-moving macro quantity, weekly added as a genuine second
spec, not selected after comparing).

Run: `python research/sofr_monetary_policy_basis.py` from the repo root.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sofr_carry as sc
from data.macro import load_overnight_rate
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import turnover as turnover_fn

Z_WINDOW = 252
MIN_FRAC = 0.7
VOL_WINDOW = 63
VOL_MIN_FRAC = 0.5
TARGET_VOL = 1.0
FREQUENCIES = ("daily", "weekly")


def _one_col(series: pd.Series, col: str) -> pd.DataFrame:
    return series.to_frame(col)


def build_basis(curve: pd.DataFrame, realized_rate: pd.Series) -> pd.Series:
    implied_rate = 100.0 - curve["raw_close"]
    realized = realized_rate.reindex(curve.index).ffill()
    return implied_rate - realized


def zscore(series: pd.Series, window: int = Z_WINDOW, min_frac: float = MIN_FRAC) -> pd.Series:
    min_periods = max(1, int(window * min_frac))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std


def realized_vol(returns: pd.Series, window: int = VOL_WINDOW, min_frac: float = VOL_MIN_FRAC) -> pd.Series:
    min_periods = max(1, int(window * min_frac))
    return returns.rolling(window, min_periods=min_periods).std() * np.sqrt(252)


def evaluate(signal: pd.Series, returns: pd.Series, cost_bps_value: float, frequency: str, label: str) -> dict:
    signal_df, ret_df = _one_col(signal, "sofr"), _one_col(returns, "sofr")
    cost_series = pd.Series({"sofr": cost_bps_value})
    gross = backtest_signal(signal_df, ret_df, frequency=frequency)
    net = backtest_signal(signal_df, ret_df, frequency=frequency, cost_bps=cost_series)
    g_tr, g_va, g_te = train_validation_test_split(gross)
    n_tr, n_va, n_te = train_validation_test_split(net)
    positions = normalized_positions(signal_df, frequency)
    annualized_turnover = float(turnover_fn(positions).sum(axis=1).mean() * 252)
    return {
        "spec": label, "frequency": frequency, "annualized_turnover": annualized_turnover,
        "train_gross": simple_sharpe(g_tr), "train_net": simple_sharpe(n_tr),
        "validation_gross": simple_sharpe(g_va), "validation_net": simple_sharpe(n_va),
        "test_gross": simple_sharpe(g_te), "test_net": simple_sharpe(n_te),
    }


def main():
    print("=== SOFR monetary-policy basis (futures-implied vs. realized rate) ===")

    curve = sc.build_databento_only_continuous_curve("SOFR")
    sofr_returns = curve["adj_close"].pct_change(fill_method=None)
    realized_rate = load_overnight_rate("SOFR")
    basis = build_basis(curve, realized_rate)

    print(f"Basis: {basis.notna().sum()} of {len(basis)} obs valid "
          f"({basis.dropna().index.min().date()} to {basis.dropna().index.max().date()})")
    print(f"Basis describe: mean={basis.mean():.3f} std={basis.std():.3f} "
          f"min={basis.min():.3f} max={basis.max():.3f} (percentage points)")

    _, volume, included, _ = sc.load_rates_universe()
    combined_volume = volume.copy()
    combined_volume["SOFR"] = curve["volume"].reindex(combined_volume.index)
    from backtest.costs import liquidity_tiered_cost_bps
    cost_bps = liquidity_tiered_cost_bps(combined_volume, window_start=sc.ADV_WINDOW_START)
    sofr_cost = float(cost_bps["SOFR"])
    print(f"SOFR liquidity-tiered one-way cost: {sofr_cost:.2f}bp")

    basis_timing = np.sign(basis)
    vol = realized_vol(sofr_returns)
    basis_z = zscore(basis)
    basis_continuous = TARGET_VOL * basis_z / vol

    rows = []
    for frequency in FREQUENCIES:
        rows.append(evaluate(basis_timing, sofr_returns, sofr_cost, frequency, "basis_timing"))
        rows.append(evaluate(basis_continuous, sofr_returns, sofr_cost, frequency, "basis_continuous"))

    result = pd.DataFrame(rows).set_index(["spec", "frequency"])
    print("\n" + result.round(3).to_string())
    return result


if __name__ == "__main__":
    main()
