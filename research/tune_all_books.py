"""
research/tune_all_books.py — Per-Book hyperparameter tuning, ALL 20 Books
across every signal family, with a two-stage multiple-testing-corrected
selection rule.

Per direct instruction (2026-07-23), extending `research/tune_book_
hyperparameters.py`'s value/XSMOM-only pass (WORKFLOW.md Phase 7) to the full
Book roster already decided and logged there: momentum (3: 3/12/24mo),
breakout (2: System 1/2), crossover (3: 50/100, 50/200, 100/200), short-term
reversal (6: individual/sector tiers x 1d/5d/10d), carry (4: cross-sectional
1m/1-12, timing zero/mean), XSMOM (1), value (1) = 20 Books, all built from
each family's own already-established `src/signals/` construction — no new
signal logic, only the tuning/selection layer is new.

**Why this needed a properly designed multiple-testing correction, not just
"run the same thing 20 times"**: the value/XSMOM pass already showed real
overfitting on 2 Books even with a single HAC-gated t-test. Scaling to 20
Books x a large grid is up to hundreds of "does this beat default" questions.
If each is tested independently at a nominal 5% significance level, several
would look "significant" by pure chance even under a true null of "tuning
never helps anywhere" — not a hypothetical, confirmed by the FIRST run of this
script (2D grid, no costs): several Books' raw p-values looked promising
(breakout_system2 p=0.014, carry_timing_mean p=0.024, value p=0.0013) but
NONE survived the correction below — value came closest (Bonferroni p=0.032)
but failed the across-Book FDR step (p=0.615). 0 of 19 evaluable Books were
adopted. That result is superseded by THIS run (adds real transaction costs
and rebalance frequency as a third grid dimension), not deleted — see
WORKFLOW.md Phase 7 for both, side by side.

**Two-stage correction, applied per direct instruction:**

1. **Within-Book (Bonferroni, FWER control)**: for each Book, don't just take
   the grid's best validation result — the fact that it's the best of GRID_SIZE
   candidates already inflates how "good" it looks by chance. Test each grid
   combo's validation daily-marked PnL AGAINST the Book's own DEFAULT combo
   (target_vol=0.10, max_weight=0.30, weekly rebalancing — this project's
   actual current standard) — a PAIRED difference test (HAC t-test on the
   daily (combo - default) PnL series, `maxlags=25`) — more directly answers
   "did tuning help" than testing a combo's raw Sharpe against zero. Take the
   single best (lowest p-value, positive-mean-improvement) candidate, then
   Bonferroni-correct: `p_bonf = min(1, p_min * GRID_SIZE)`.
2. **Across-Book (Benjamini-Hochberg, FDR control)**: collect all (up to) 20
   Books' own `p_bonf` values and run `statsmodels.stats.multitest.
   multipletests(..., method="fdr_bh")` across them.

A Book's tuned hyperparameters are adopted ONLY if its best candidate survives
BOTH corrections. Every other Book keeps its flat default calibration.

**New in this run, per direct instruction, both added together:**

1. **Real transaction costs, wired into `Book.run()` itself** (`cost_bps`
   param, added to `portfolio/book.py` 2026-07-23) — `backtest.costs.
   liquidity_tiered_cost_bps` on this universe's own trailing ADV, the SAME
   cost assumption every standalone signal's net-of-cost number in this
   project already uses (`liquidity_tiered_cost_bps`), reused here, not a
   second/different cost model. **The alpha fed into every Book stays GROSS**
   — `build_all_signals()` below never touches `cost_bps` at any point; costs
   are deducted exactly ONCE, inside `Book.run()`/`daily_mark_pnl()`, not
   pre-baked into any signal upstream of the Allocator. This was a real
   prerequisite, not optional: without it, rebalancing MORE often could only
   ever look equal-or-better in a grid search (more chances to react, no
   downside charged for the extra trades) — exactly the blind spot flagged
   when the earlier monthly->weekly switch was first made without a cost
   model to check it against.
2. **Rebalance frequency as a third grid dimension** — `FREQUENCY_GRID`:
   monthly (`"ME"`, 12/yr) and weekly (`"W-FRI"`, 52/yr), each with its own
   `EWMA_HALFLIFE` rescaled to hold the same ~1.7yr calendar reactivity (12 and
   87 periods respectively — see `research/value_momentum_combine.py`'s own
   module docstring for why this rescaling matters, found live there).
   **Daily rebalancing deliberately excluded from this grid** — not hidden,
   a real, stated compute-practicality tradeoff: at 20 Books x 25 sizing
   combos, a daily-cadence tier would mean ~2,600 rebalance dates per
   `Book.run()` call instead of ~125-540, several times slower per call, on
   top of already being the single largest cost driver in ANY realistic
   scenario (most turnover, least alpha-per-trade). Worth a dedicated,
   narrower follow-up, not bundled into this already-large 20-Book x
   25-sizing-combo run.

GRID_SIZE is now `len(TARGET_VOL_GRID) * len(MAX_WEIGHT_GRID) *
len(FREQUENCY_GRID)` = 50, and the Bonferroni correction uses that larger
denominator.

**Evaluation granularity**: daily-marked PnL of each Book's solved weight path
(`portfolio.book.daily_mark_pnl`, now cost-aware — see that function's own
updated docstring), NOT the sparser period-level PnL `Book.run()` itself
reports — see `tune_book_hyperparameters.py`'s module docstring for why
(measurement precision, not a strategy change).

**Compute note**: `portfolio.covariance.build_cov_dict` is built ONCE per
(Book, frequency) pair — 2 per Book, not once per of the 50 grid points —
cov_dict only depends on the active asset set, returns, and rebalance
frequency, not on target_vol/max_weight.

Run: `python research/tune_all_books.py` from the repo root. Results written
to Data/research/tune_all_books/ (overwrites the prior no-cost/2D-grid run's
CSVs — see WORKFLOW.md Phase 7 for both results preserved side by side).
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.continuous_curve import load_continuous_backadjusted, load_continuous_raw
from data.universe import get_liquid_universe
from data.sectors import sectors_for_universe
from data.volatility import yang_zhang_volatility
from data.term_structure import build_carry_panel
from data.macro import load_yield_curve, load_cpi
from signals.momentum import build_momentum_features
from signals.breakout import TURTLE_SYSTEMS, build_breakout_regime
from signals.crossover import CROSSOVER_PAIRS, build_crossover_regime
from signals.short_term_reversal import (
    build_reversal_score, build_sector_returns, broadcast_sector_score_to_assets, sector_realized_vol,
)
from signals.carry import sampled_carry, cross_sectional_carry_signal, carry_timing_signal
from signals.xs_momentum import cross_sectional_momentum_signal
from signals.value import cross_sectional_value_signal
from signals.transforms import vol_targeted_sign_signal
from portfolio.covariance import build_cov_dict
from portfolio.book import Book, daily_mark_pnl
from backtest.costs import liquidity_tiered_cost_bps
from backtest.performance import simple_sharpe
from backtest.splits import TRAIN_END, train_validation_test_split

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Data" / "research" / "tune_all_books"

ADV_WINDOW_START = "2024-07-14"
ADV_THRESHOLD = 1000
YZ_WINDOW = 63
TARGET_VOL_SIGNAL = 0.40  # momentum/breakout/crossover's own per-instrument target, matching research/portfolio.py
MIN_GROUP_SIZE = 2
LAGS = (1, 5, 10)

# Book calibration — fixed (not tuned), reused verbatim from research/
# value_momentum_combine.py / research/tune_book_hyperparameters.py.
GAMMA = 20000.0
KAPPA = 1.0
LAMBD = 0.0
SCALE_MIN, SCALE_MAX = 0.1, 5.0
COV_WINDOW = 252

DEFAULT_TARGET_VOL = 0.10
DEFAULT_MAX_WEIGHT = 0.3
DEFAULT_FREQUENCY = "weekly"  # this project's current standard (WORKFLOW.md Phase 7)
TARGET_VOL_GRID = [0.05, 0.075, 0.10, 0.125, 0.15]
MAX_WEIGHT_GRID = [0.15, 0.2, 0.3, 0.4, 0.5]

# (name, pandas freq, periods_per_year, ewma_halflife) — EWMA_HALFLIFE rescaled
# per frequency to hold the SAME ~1.7yr calendar reactivity (12 periods at
# monthly cadence, 87 at weekly — 20 * periods_per_year/12, see research/
# value_momentum_combine.py's own module docstring for why this matters).
# Daily deliberately excluded — see module docstring.
FREQUENCY_GRID = [
    ("monthly", "ME", 12, 12),
    ("weekly", "W-FRI", 52, 87),
]

GRID_SIZE = len(TARGET_VOL_GRID) * len(MAX_WEIGHT_GRID) * len(FREQUENCY_GRID)  # 50

DAILY_HAC_MAXLAGS = 25

ALPHA_FWER = 0.05  # within-Book Bonferroni threshold
ALPHA_FDR = 0.05   # across-Book Benjamini-Hochberg threshold


def load_and_prepare_data():
    adj = load_continuous_backadjusted()
    raw = load_continuous_raw()
    volume = adj["volume"]
    included, excluded = get_liquid_universe(volume, ADV_WINDOW_START, ADV_THRESHOLD)
    print(f"Excluded (ADV < {ADV_THRESHOLD}): {excluded}")
    print(f"Universe: {len(included)} of {len(volume.columns)} assets")

    adj = {f: df[included] for f, df in adj.items()}
    raw = {f: df[included] for f, df in raw.items()}
    groups = sectors_for_universe(included)

    return {
        "close": adj["close"], "high": adj["high"], "low": adj["low"], "volume": adj["volume"],
        "raw_open": raw["open"], "raw_high": raw["high"], "raw_low": raw["low"], "raw_close": raw["close"],
        "is_roll_date": raw["is_roll_date"],
        "included": included, "groups": groups,
        "yield_curve": load_yield_curve(), "cpi": load_cpi(),
    }


def build_vol(prepared):
    return yang_zhang_volatility(
        prepared["raw_open"], prepared["raw_high"], prepared["raw_low"], prepared["raw_close"],
        window=YZ_WINDOW, roll_mask=prepared["is_roll_date"],
    )


def build_all_signals(prepared, vol):
    """All 20 Books' alpha DataFrames — every construction reused verbatim
    from its own family's already-established src/signals/ module, matching
    the exact spec set WORKFLOW.md Phase 7's Book-count decisions already
    logged. No new signal-construction logic anywhere in this function."""
    close, high, low, groups = prepared["close"], prepared["high"], prepared["low"], prepared["groups"]
    returns = close.pct_change(fill_method=None)
    signals = {}

    # Momentum: 3 Books (3/12/24mo, research/momentum.py's own LOOKBACK_DAYS convention)
    mom_days = {3: 63, 12: 252, 24: 504}
    mom_features = build_momentum_features(close, lookbacks=tuple(mom_days.values()))
    for months, days in mom_days.items():
        signals[f"momentum_{months}mo"] = vol_targeted_sign_signal(mom_features[f"mom_{days}"], vol, target_vol=TARGET_VOL_SIGNAL)

    # Breakout: 2 Books (System 1, System 2)
    for name, spec in TURTLE_SYSTEMS.items():
        regime = build_breakout_regime(high, low, close, spec["entry"], spec["exit"])
        signals[f"breakout_{name}"] = vol_targeted_sign_signal(regime, vol, target_vol=TARGET_VOL_SIGNAL)

    # Crossover: 3 Books (50/100, 50/200, 100/200)
    for name, spec in CROSSOVER_PAIRS.items():
        regime = build_crossover_regime(close, spec["fast"], spec["slow"])
        signals[f"crossover_{name}"] = vol_targeted_sign_signal(regime, vol, target_vol=TARGET_VOL_SIGNAL)

    # Short-term reversal: 6 Books (individual/sector tiers x 1d/5d/10d)
    for lag in LAGS:
        signals[f"reversal_individual_{lag}d"] = build_reversal_score(close, lag, vol, groups, MIN_GROUP_SIZE)
    sector_returns = build_sector_returns(returns, groups)
    sector_close = (1 + sector_returns.fillna(0)).cumprod()
    sector_vol = sector_realized_vol(sector_returns, window=YZ_WINDOW)
    all_sectors_group = {"AllSectors": list(sector_returns.columns)}
    for lag in LAGS:
        sector_score = build_reversal_score(sector_close, lag, sector_vol, all_sectors_group, MIN_GROUP_SIZE)
        signals[f"reversal_sector_{lag}d"] = broadcast_sector_score_to_assets(sector_score, groups, prepared["included"])

    # Carry: 4 Books (cross-sectional 1m/1-12, timing zero/mean)
    carry_panel, _ = build_carry_panel(prepared["included"])
    carry1m = sampled_carry(carry_panel, holding_months=1)
    carry1_12 = sampled_carry(carry_panel, holding_months=12)
    signals["carry_cross_sectional_1m"] = cross_sectional_carry_signal(carry1m, groups, MIN_GROUP_SIZE)
    signals["carry_cross_sectional_1_12"] = cross_sectional_carry_signal(carry1_12, groups, MIN_GROUP_SIZE)
    signals["carry_timing_zero"] = carry_timing_signal(carry1m, groups, reference="zero")
    signals["carry_timing_mean"] = carry_timing_signal(carry1m, groups, reference="mean")

    # XSMOM: 1 Book
    signals["xs_momentum"] = cross_sectional_momentum_signal(close, groups, lookback=12, skip=1, min_group_size=MIN_GROUP_SIZE)

    # Value: 1 Book
    signals["value"] = cross_sectional_value_signal(
        close, prepared["yield_curve"], prepared["cpi"], groups, lookback_months=60, min_group_size=MIN_GROUP_SIZE,
    )

    assert len(signals) == 20, f"expected 20 Books, got {len(signals)}"
    return signals


def _active_columns(alpha_df, returns_df, min_valid_frac=0.90):
    has_alpha = alpha_df.notna().any()
    returns_valid_frac = returns_df.notna().mean()
    return [c for c in alpha_df.columns if has_alpha.get(c, False) and returns_valid_frac.get(c, 0.0) >= min_valid_frac]


def hac_mean_tstat(returns: pd.Series, maxlags: int = DAILY_HAC_MAXLAGS) -> dict:
    y = returns.dropna()
    if len(y) < maxlags * 2:
        return {"mean": np.nan, "t": np.nan, "p": np.nan, "n_obs": len(y)}
    X = np.ones((len(y), 1))
    model = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"mean": float(model.params[0]), "t": float(model.tvalues[0]), "p": float(model.pvalues[0]), "n_obs": len(y)}


def make_book(name, alpha_scaled, cov_dict, target_vol, max_weight, periods_per_year, ewma_halflife, cost_bps):
    return Book(
        name=name, alpha_df=alpha_scaled, cov_dict=cov_dict,
        gamma=GAMMA, kappa=KAPPA, lambd=LAMBD, max_weight=max_weight,
        target_vol=target_vol, ewma_halflife=ewma_halflife,
        scale_min=SCALE_MIN, scale_max=SCALE_MAX,
        periods_per_year=periods_per_year, dollar_neutral=False,
        cost_bps=cost_bps,
    )


def tune_one_book(name, alpha_df, returns, cost_bps):
    """Returns a grid_df (target_vol x max_weight x frequency, GRID_SIZE rows)
    or None if the Book's active universe is too small / has no usable
    covariance window at EITHER frequency to build anything meaningful."""
    active = _active_columns(alpha_df, returns)
    if len(active) < 3:
        print(f"  {name}: skipped — only {len(active)} active assets (<3)")
        return None

    alpha_active = alpha_df[active]
    train_std = alpha_active.loc[:TRAIN_END].stack().std()
    if not train_std or np.isnan(train_std) or train_std < 1e-12:
        train_std = 1.0
    alpha_scaled = alpha_active / train_std

    # One cov_dict per FREQUENCY (not per grid point) — sizing doesn't affect
    # which dates are usable, only the frequency does.
    cov_dicts = {}
    for freq_name, freq_str, ppy, halflife in FREQUENCY_GRID:
        cd = build_cov_dict(returns[active], window=COV_WINDOW, freq=freq_str)
        if len(cd) >= 20:
            cov_dicts[freq_name] = cd
    if not cov_dicts:
        print(f"  {name}: skipped — no frequency has >=20 rebalance dates with a full covariance window")
        return None

    default_key = (DEFAULT_TARGET_VOL, DEFAULT_MAX_WEIGHT, DEFAULT_FREQUENCY)
    daily_pnls = {}
    viable_frequencies = set()
    for freq_name, freq_str, ppy, halflife in FREQUENCY_GRID:
        if freq_name not in cov_dicts:
            continue
        # Probe once per frequency (target_vol/max_weight don't affect which
        # dates are usable) — Book.run()'s own `len(common) < 20` early-return
        # (no "weights" key) is a real gap the cov_dict >=20 check above
        # doesn't catch, since `common` also intersects alpha_df.dropna(),
        # which can be materially smaller (found live on
        # carry_cross_sectional_1_12). Skip this FREQUENCY, not the whole
        # Book, if it fails — the other frequency may still be viable.
        probe = make_book(name, alpha_scaled, cov_dicts[freq_name], DEFAULT_TARGET_VOL, DEFAULT_MAX_WEIGHT,
                           ppy, halflife, cost_bps).run(returns)
        if "weights" not in probe or len(probe.get("pnl", [])) == 0:
            print(f"  {name} ({freq_name}): skipped — only {probe.get('n_rebalance_dates_valid', 0)} usable dates (<20)")
            continue
        viable_frequencies.add(freq_name)
        daily_pnls[(DEFAULT_TARGET_VOL, DEFAULT_MAX_WEIGHT, freq_name)] = daily_mark_pnl(probe["weights"], returns, cost_bps=cost_bps)

    if DEFAULT_FREQUENCY not in viable_frequencies:
        print(f"  {name}: skipped — the default frequency ({DEFAULT_FREQUENCY}) itself isn't viable, nothing to compare against")
        return None

    for freq_name, freq_str, ppy, halflife in FREQUENCY_GRID:
        if freq_name not in viable_frequencies:
            continue
        for tv, mw in itertools.product(TARGET_VOL_GRID, MAX_WEIGHT_GRID):
            key = (tv, mw, freq_name)
            if key == default_key:
                continue
            book = make_book(name, alpha_scaled, cov_dicts[freq_name], tv, mw, ppy, halflife, cost_bps)
            result = book.run(returns)
            daily_pnls[key] = daily_mark_pnl(result["weights"], returns, cost_bps=cost_bps)

    default_val_pnl = train_validation_test_split(daily_pnls[default_key])[1]

    rows = []
    for (tv, mw, freq_name), daily_pnl in daily_pnls.items():
        train_pnl, val_pnl, test_pnl = train_validation_test_split(daily_pnl)
        is_default = (tv, mw, freq_name) == default_key
        diff = pd.Series(dtype=float) if is_default else (val_pnl - default_val_pnl).dropna()
        hac = {"mean": 0.0, "t": np.nan, "p": 1.0, "n_obs": 0} if is_default else hac_mean_tstat(diff)
        rows.append({
            "book": name, "target_vol": tv, "max_weight": mw, "frequency": freq_name, "is_default": is_default,
            "n_active_assets": len(active),
            "train_sharpe": simple_sharpe(train_pnl), "val_sharpe": simple_sharpe(val_pnl), "test_sharpe": simple_sharpe(test_pnl),
            "diff_vs_default_mean": hac["mean"], "diff_vs_default_t": hac["t"],
            "diff_vs_default_p": hac["p"], "diff_n_obs": hac["n_obs"],
        })
    return pd.DataFrame(rows)


def select_best_candidate(grid_df):
    """Best (lowest-p, positive-improvement) non-default candidate in one
    Book's grid, Bonferroni-corrected across the GRID_SIZE comparisons made
    within that Book. Returns a dict (best row + p_bonf) or None if no combo
    improved on default at all (nothing to correct/adopt)."""
    candidates = grid_df[(~grid_df["is_default"]) & (grid_df["diff_vs_default_mean"] > 0)]
    if len(candidates) == 0:
        return None
    best = candidates.loc[candidates["diff_vs_default_p"].idxmin()].to_dict()
    best["p_bonf"] = min(1.0, best["diff_vs_default_p"] * GRID_SIZE)
    return best


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prepared = load_and_prepare_data()
    close = prepared["close"]
    returns = close.pct_change(fill_method=None)
    vol = build_vol(prepared)

    # Real transaction costs — same liquidity-tiered ADV-based assumption
    # every standalone signal's net-of-cost number in this project already
    # uses, not a new/different cost model.
    cost_bps = liquidity_tiered_cost_bps(prepared["volume"], window_start=ADV_WINDOW_START)
    print("\nLiquidity-tiered cost assumption (bps, one-way):")
    print(cost_bps.sort_values().to_string())

    print("\n--- Building all 20 signal alphas ---")
    all_signals = build_all_signals(prepared, vol)
    print(f"Books: {list(all_signals.keys())}")
    # Explicit check, not just a claim: build_all_signals() must never touch
    # cost_bps — costs are applied exactly once, inside Book.run()/
    # daily_mark_pnl(), not pre-baked into any signal fed to the Allocator.
    assert "cost_bps" not in build_all_signals.__code__.co_names, \
        "build_all_signals must not reference cost_bps — costs belong only inside Book.run()"

    all_grids = {}
    candidates = {}
    print(f"\n--- Tuning each Book ({GRID_SIZE}-point grid: {len(TARGET_VOL_GRID)} target_vol x "
          f"{len(MAX_WEIGHT_GRID)} max_weight x {len(FREQUENCY_GRID)} frequency, daily-marked, "
          f"cost-inclusive, HAC-tested vs. default) ---")
    for name, alpha_df in all_signals.items():
        print(f"\n{name}:")
        grid_df = tune_one_book(name, alpha_df, returns, cost_bps)
        if grid_df is None:
            continue
        all_grids[name] = grid_df
        cand = select_best_candidate(grid_df)
        if cand is None:
            print(f"  no combo improved on default (target_vol={DEFAULT_TARGET_VOL}, max_weight={DEFAULT_MAX_WEIGHT}, "
                  f"frequency={DEFAULT_FREQUENCY}) at all")
            candidates[name] = {"book": name, "p_bonf": 1.0, "found_candidate": False}
        else:
            print(f"  best: target_vol={cand['target_vol']:.3f} max_weight={cand['max_weight']:.2f} "
                  f"frequency={cand['frequency']} diff_t={cand['diff_vs_default_t']:.2f} "
                  f"raw_p={cand['diff_vs_default_p']:.4f} bonferroni_p={cand['p_bonf']:.4f} (n={cand['diff_n_obs']:.0f})")
            candidates[name] = {**cand, "found_candidate": True}

    full_grid = pd.concat(all_grids.values(), ignore_index=True)
    full_grid.to_csv(OUTPUT_DIR / "full_grid_all_books.csv", index=False)

    cand_df = pd.DataFrame(candidates.values())
    p_bonf = cand_df["p_bonf"].values
    reject, p_fdr, _, _ = multipletests(p_bonf, alpha=ALPHA_FDR, method="fdr_bh")
    cand_df["p_fdr"] = p_fdr
    cand_df["adopt_tuned"] = reject & cand_df["found_candidate"]

    print(f"\n=== Across-Book Benjamini-Hochberg FDR correction (alpha={ALPHA_FDR}, {len(cand_df)} Books tested) ===")
    print(cand_df[["book", "found_candidate", "p_bonf", "p_fdr", "adopt_tuned"]].sort_values("p_bonf").to_string(index=False))
    cand_df.to_csv(OUTPUT_DIR / "candidate_selection.csv", index=False)

    n_adopted = int(cand_df["adopt_tuned"].sum())
    print(f"\n{n_adopted} of {len(cand_df)} Books survive BOTH corrections and get tuned hyperparameters adopted.")
    if n_adopted > 0:
        print(cand_df[cand_df["adopt_tuned"]][["book", "target_vol", "max_weight", "frequency", "diff_vs_default_t", "p_bonf", "p_fdr"]].to_string(index=False))
    else:
        print(f"No Book's tuning survives both the within-Book Bonferroni and across-Book FDR corrections — "
              f"every Book keeps its flat default calibration (target_vol={DEFAULT_TARGET_VOL}, "
              f"max_weight={DEFAULT_MAX_WEIGHT}, frequency={DEFAULT_FREQUENCY}).")

    print(f"\nResults written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
