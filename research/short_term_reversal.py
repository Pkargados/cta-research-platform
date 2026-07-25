"""
research/short_term_reversal.py — Short-term reversal driver: 6 parallel specs
({individual, sector} x {1, 5, 10}d), gross/net Sharpe, turnover, and the Nagel
(2011) VIX-conditioning regression (Newey-West HAC) + sizing-overlay comparison.

Reproduces CLAUDE.md's documented result: turnover 109-362x annualized, net Sharpe
deeply negative across every spec. VIX-conditioning HAC regression: individual-tier
NOT statistically distinguishable from zero (t=0.58, p=0.56); sector-tier IS
significant (t=2.33, p=0.02) despite a tiny R² (0.17%) — both at the 5-day
headline lag. VIX-adjusted sizing overlay is practically mixed (helps some
periods, hurts others), doesn't rescue net-of-cost profitability either way.

Run: `python research/short_term_reversal.py` from the repo root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.volatility import yang_zhang_volatility
from signals.short_term_reversal import build_all_reversal_signals, LAGS
from signals.vix_overlay import vix_size_multiplier, apply_size_multiplier
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
FREQUENCY = "daily"
HAC_MAXLAGS = 20
HEADLINE_LAG = 5
DATA_DIR = Path(__file__).resolve().parent.parent / "Data"


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    raw = load_continuous_raw()
    included, excluded = get_liquid_universe(adj["volume"], ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(adj['volume'].columns)} assets")
    adj = {f: df[included] for f, df in adj.items()}
    raw = {f: df[included] for f, df in raw.items()}
    sectors = sectors_for_universe(included)
    return adj, raw, sectors


def build_vol(raw):
    return yang_zhang_volatility(
        raw["open"], raw["high"], raw["low"], raw["close"], window=YZ_WINDOW, roll_mask=raw["is_roll_date"],
    )


def load_vix():
    vix = pd.read_csv(DATA_DIR / "vix_data.csv", parse_dates=["Date"], index_col="Date")
    return vix["Close"]


def annualized_turnover(signal):
    positions = normalized_positions(signal, FREQUENCY)
    return float(turnover_fn(positions).sum(axis=1).mean() * 252)


def evaluate_all_specs(signals, returns, cost_bps):
    rows = []
    for name, signal in signals.items():
        gross = backtest_signal(signal, returns, frequency=FREQUENCY)
        net = backtest_signal(signal, returns, frequency=FREQUENCY, cost_bps=cost_bps)
        g_train, g_val, g_test = train_validation_test_split(gross)
        n_train, n_val, n_test = train_validation_test_split(net)
        rows.append({
            "spec": name,
            "annualized_turnover": annualized_turnover(signal),
            "train_gross": simple_sharpe(g_train), "train_net": simple_sharpe(n_train),
            "validation_gross": simple_sharpe(g_val), "validation_net": simple_sharpe(n_val),
            "test_gross": simple_sharpe(g_test), "test_net": simple_sharpe(n_test),
        })
    return pd.DataFrame(rows).set_index("spec")


def hac_vix_regression(strategy_returns, vix, maxlags=HAC_MAXLAGS):
    """Plain OLS of the reversal book's daily return on the PRIOR day's VIX level
    (predetermined/known-in-advance, not contemporaneous — Nagel's state-variable
    framing: today's return is predicted by yesterday's already-known VIX, the
    same shift(1) convention `signals.vix_overlay.vix_size_multiplier` applies),
    Newey-West HAC standard errors (Nagel's own convention for daily, serially-
    correlated, overlapping-lookback data)."""
    lagged_vix = vix.shift(1)
    aligned = pd.concat([strategy_returns.rename("ret"), lagged_vix.rename("vix")], axis=1, join="inner").dropna()
    X = sm.add_constant(aligned["vix"].values)
    y = aligned["ret"].values
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"t": float(model.tvalues[1]), "p": float(model.pvalues[1]), "r_squared": float(model.rsquared), "n_obs": len(y)}


def vix_sizing_comparison(signal, returns, vix):
    """Simple vs. VIX-adjusted Sharpe — the multiplier applied AFTER gross-exposure
    normalization (signals.vix_overlay's own documented ordering fix)."""
    positions = normalized_positions(signal, FREQUENCY)
    simple_returns = (positions * returns).sum(axis=1).dropna()

    multiplier = vix_size_multiplier(vix)
    adjusted_positions = apply_size_multiplier(positions, multiplier)
    adjusted_returns = (adjusted_positions * returns).sum(axis=1).dropna()

    rows = []
    for label, series in [("simple", simple_returns), ("vix_adjusted", adjusted_returns)]:
        train, val, test = train_validation_test_split(series)
        rows.append({"sizing": label, "train": simple_sharpe(train), "validation": simple_sharpe(val), "test": simple_sharpe(test)})
    return pd.DataFrame(rows).set_index("sizing")


def main():
    adj, raw, sectors = load_and_prepare_data()
    close = adj["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(raw)
    cost_bps = liquidity_tiered_cost_bps(adj["volume"], window_start=ADV_WINDOW_START)
    vix = load_vix()

    signals = build_all_reversal_signals(close, vol, sectors, lags=LAGS)
    print(f"\nSpecs: {list(signals.keys())}")

    print("\n--- Gross/net Sharpe, all 6 specs ---")
    result = evaluate_all_specs(signals, returns, cost_bps)
    print(result.round(3).to_string())

    print(f"\n--- VIX-conditioning HAC regression (maxlags={HAC_MAXLAGS}), headline lag={HEADLINE_LAG}d ---")
    for tier in ("individual", "sector"):
        name = f"{tier}_{HEADLINE_LAG}d"
        positions = normalized_positions(signals[name], FREQUENCY)
        strategy_returns = (positions * returns).sum(axis=1).dropna()
        reg = hac_vix_regression(strategy_returns, vix)
        print(f"{tier}: t={reg['t']:.2f} p={reg['p']:.2f} R²={reg['r_squared']:.4f} (n={reg['n_obs']})")

    print(f"\n--- VIX-adjusted sizing overlay comparison, headline lag={HEADLINE_LAG}d ---")
    for tier in ("individual", "sector"):
        name = f"{tier}_{HEADLINE_LAG}d"
        print(f"\n{tier}:")
        print(vix_sizing_comparison(signals[name], returns, vix).round(3).to_string())

    print(
        "\nDocumented: individual-tier VIX regression NOT significant (t=0.58, p=0.56); "
        "sector-tier IS significant (t=2.33, p=0.02) despite tiny R² (0.17%). "
        "Net-of-cost Sharpe deeply negative across all 6 specs either way."
    )


if __name__ == "__main__":
    main()
