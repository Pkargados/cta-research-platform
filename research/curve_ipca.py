"""
research/curve_ipca.py — "Curve IPCA" (WORKFLOW.md §11e item #3): combine
Litterman-Scheinkman (1991)'s level/steepness/curvature finding with
Kelly-Pruitt-Su (2018) IPCA's characteristics-instrumented latent-factor
estimator + alpha significance test, applied to this project's own Rates
cross-section (US_2Y, US_5Y, US_10Y, US_30Y, UltraBond, SOFR - the same 6
names items #1/#2 already built real data for).

Characteristics (z_{i,t}), each already computed or newly available this
session:
  - duration: a static, labeled APPROXIMATE modified-duration figure per
    instrument (standard market-convention values, not fitted from data -
    see DURATION_YEARS below - same "label it, don't fake it" discipline as
    Carry's ICE-softs proxy).
  - carry: the real calendar-spread carry level already built for Carry/
    item #1 (data.term_structure.build_carry_panel / build_databento_only_carry).
  - vol: rolling realized volatility of each instrument's own returns
    (annualized, close-to-close - not Yang-Zhang, for uniform treatment
    across all 6 names including SOFR, which has no OHLC-based estimator
    built for it).

Each characteristic is cross-sectionally rank-transformed to [-0.5, 0.5] per
date (Kelly-Pruitt-Su's own convention, Section 4), plus a constant column -
a DIFFERENT convention from this project's own `signals.transforms.
cross_sectional_rank` (-(N+1)/2 integer scale, sector-scoped for Carry/XSMOM/
Value), re-derived here rather than reused (CLAUDE.md Rule 7) since IPCA's
own convention is a plain cross-sectional percentile with no sector concept.

See `src/signals/ipca.py`'s own module docstring for the small-N estimator
itself and its explicitly disclosed simplifications relative to the paper's
full large-cross-section machinery.

Run: `python research/curve_ipca.py` from the repo root.
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
from data.term_structure import build_carry_panel
from signals.ipca import (
    fit_ipca_restricted, total_r2, bootstrap_alpha_test, alpha_signal,
)
from backtest.engine import backtest_signal, normalized_positions
from backtest.performance import simple_sharpe
from backtest.splits import train_validation_test_split
from backtest.costs import liquidity_tiered_cost_bps, turnover as turnover_fn

RATES_6 = ["US_2Y", "US_5Y", "US_10Y", "US_30Y", "UltraBond", "SOFR"]

# Standard market-convention APPROXIMATE modified duration, years - a
# labeled, static assumption (not fitted from data), same "label it, don't
# fake it" discipline as Carry's ICE-softs proxy. US_30Y is the classic
# CBOT bond contract (~15-25y deliverable basket); UltraBond is the 25-30y+
# basket (see the fixed-income-RV discussion this session on the two
# contracts' different segments).
DURATION_YEARS = {
    "US_2Y": 1.9, "US_5Y": 4.5, "US_10Y": 7.8, "US_30Y": 17.5,
    "UltraBond": 22.0, "SOFR": 0.25,
}

K_FACTORS = 2
VOL_WINDOW = 63
VOL_MIN_FRAC = 0.5
FREQUENCY = "weekly"
TARGET_VOL = 1.0


def cross_sectional_rank_ipca(panel: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank in [-0.5, 0.5] per date - IPCA's own characteristic
    convention (Kelly-Pruitt-Su Section 4)."""
    ranked = panel.rank(axis=1, pct=True) - 0.5
    return ranked.where(panel.notna())


def build_characteristics(returns: pd.DataFrame, carry: pd.DataFrame) -> dict:
    duration = pd.DataFrame({a: DURATION_YEARS[a] for a in RATES_6}, index=returns.index)
    duration = duration.where(returns.notna())

    vol = returns.rolling(VOL_WINDOW, min_periods=max(1, int(VOL_WINDOW * VOL_MIN_FRAC))).std() * np.sqrt(252)

    return {
        "duration": cross_sectional_rank_ipca(duration),
        "carry": cross_sectional_rank_ipca(carry[RATES_6]),
        "vol": cross_sectional_rank_ipca(vol),
    }


def build_panels(chars: dict, returns: pd.DataFrame):
    """Stack characteristics (plus a constant) into a (T, N, L) numpy array,
    aligned so z_panel[t] pairs with r_next[t] = the return realized from t
    to t+1 (CLAUDE.md Rule 3's shift(1) discipline - characteristics known
    at t explain the NEXT period's return, never looking ahead)."""
    dates = returns.index[:-1]
    r_next = returns.shift(-1).loc[dates, RATES_6].to_numpy()

    l_names = list(chars.keys()) + ["const"]
    z_panel = np.full((len(dates), len(RATES_6), len(l_names)), np.nan)
    for li, name in enumerate(chars.keys()):
        z_panel[:, :, li] = chars[name].loc[dates, RATES_6].to_numpy()
    z_panel[:, :, -1] = 1.0
    return z_panel, r_next, dates, l_names


def evaluate_alpha_signal(signal: pd.DataFrame, returns: pd.DataFrame, cost_bps: pd.Series, label: str) -> dict:
    gross = backtest_signal(signal, returns, frequency=FREQUENCY)
    net = backtest_signal(signal, returns, frequency=FREQUENCY, cost_bps=cost_bps)
    g_tr, g_va, g_te = train_validation_test_split(gross)
    n_tr, n_va, n_te = train_validation_test_split(net)
    positions = normalized_positions(signal, FREQUENCY)
    annualized_turnover = float(turnover_fn(positions).sum(axis=1).mean() * 252)
    return {
        "spec": label, "annualized_turnover": annualized_turnover,
        "train_gross": simple_sharpe(g_tr), "train_net": simple_sharpe(n_tr),
        "validation_gross": simple_sharpe(g_va), "validation_net": simple_sharpe(n_va),
        "test_gross": simple_sharpe(g_te), "test_net": simple_sharpe(n_te),
    }


def main():
    print("=== Curve IPCA: Rates cross-section (US_2Y/5Y/10Y/30Y/UltraBond/SOFR) ===")
    close, volume, included, sectors = sc.load_rates_universe()
    sofr_curve = sc.build_databento_only_continuous_curve("SOFR")
    sofr_carry = sc.build_databento_only_carry("SOFR")
    carry_panel, is_proxy = build_carry_panel(included)

    returns = close.pct_change(fill_method=None).copy()
    returns["SOFR"] = sofr_curve["adj_close"].pct_change(fill_method=None).reindex(returns.index)
    returns = returns[RATES_6]

    combined_carry = carry_panel.reindex(returns.index).copy()
    combined_carry["SOFR"] = sofr_carry.reindex(returns.index)

    chars = build_characteristics(returns, combined_carry)
    z_panel, r_next, dates, l_names = build_panels(chars, returns)
    print(f"Characteristics: {l_names}")
    print(f"Panel: {len(dates)} dates x {len(RATES_6)} assets x {len(l_names)} characteristics")

    fit = fit_ipca_restricted(z_panel, r_next, k=K_FACTORS)
    print(f"Valid dates for managed portfolios: {int(fit['valid_t'].sum())} of {len(dates)}")
    print(f"Eigenvalues (all {len(l_names)}): {fit['eigvals'].round(4)}")
    print(f"Total R^2 ({K_FACTORS} factors, managed-portfolio space): {total_r2(fit):.4f}")

    test = bootstrap_alpha_test(fit, n_boot=1000, seed=0)
    print(f"\nGamma_alpha estimate: {dict(zip(l_names, test['gamma_alpha'].round(4)))}")
    print(f"W_alpha = {test['w_alpha']:.6f}, bootstrap p-value = {test['p_value']:.4f} (n_boot=1000)")

    print("\n=== Building the alpha-implied trading signal regardless of significance "
          "(reported honestly either way, per this project's own discipline) ===")
    alpha_vals = alpha_signal(z_panel, test["gamma_alpha"])  # (T, N)
    alpha_df = pd.DataFrame(alpha_vals, index=dates, columns=RATES_6).reindex(returns.index)
    alpha_df = TARGET_VOL * alpha_df.div(alpha_df.std(), axis=1)  # crude vol-scale per asset, continuous per Rule 5

    combined_volume = volume.copy()
    combined_volume["SOFR"] = sofr_curve["volume"].reindex(combined_volume.index)
    cost_bps = liquidity_tiered_cost_bps(combined_volume, window_start=sc.ADV_WINDOW_START)[RATES_6]

    result = evaluate_alpha_signal(alpha_df, returns, cost_bps, "curve_ipca_alpha")
    print(pd.DataFrame([result]).set_index("spec").round(3).to_string())

    return fit, test, alpha_df, result


if __name__ == "__main__":
    main()
