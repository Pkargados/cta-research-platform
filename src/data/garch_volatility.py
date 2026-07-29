"""
data/garch_volatility.py — Walk-forward, point-in-time GJR-GARCH(1,1,1)
annualized volatility, wrapping the author's own public `dcc-garch-python`
package (github.com/Pkargados/dcc-garch-python) -- cloned on demand into a
gitignored local cache, not vendored into this repo's own git history.

The public repo has no pyproject.toml/setup.py (a plain `python/` source
tree, not currently pip-installable), so this clones it directly -- the
exact same on-demand `git clone` pattern dashboard/_bootstrap_data.py
already uses for the private data repo, just without auth, since this repo
is public. This is a real improvement over an earlier version of this
module that pointed at a local sibling folder only present on one machine:
a fresh clone of THIS repo, anywhere, including Streamlit Cloud, can now
reach the GARCH code too, not just this developer's own computer.

Refit every REFIT_FREQ_DAYS trading days (default 20, ~monthly) using only
data available through the refit date; between refits, extend the
conditional volatility path with FIXED parameters via the package's own
filter_gjr_garch (its documented ~100x-faster no-optimization path) rather
than re-fitting daily. This matches standard GARCH practice (parameters
don't need daily re-estimation -- the recursion itself already updates the
variance forecast day by day using newly realized returns) and mirrors that
package's own live/daily_run.py production pattern (monthly refit there;
20 trading days here per direct instruction, a reasonable practice-matching
default, not a backtested/tuned choice).

Sigma is returned in this project's standard convention (annualized,
decimal) -- the package's native units are % daily:
annualized_vol = sigma_pct_daily / 100 * sqrt(252).

Slow by construction (a real MLE fit per asset per refit window) -- meant
for offline/research use via research/vol_estimator_comparison.py, not live
dashboard computation.

The clone-and-import is deferred (inside _get_garch_functions(), not at
module load time) rather than eager at the top of this file. Reading an
already-cached GARCH result (research/vol_estimator_comparison.py's
load_or_compute_garch(), the only path the public dashboard should ever
hit) never touches this at all -- only an actual cache-miss fit does, and
even then it's now a network clone of a public repo, not a hard dependency
on a folder that has to already exist locally.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REFIT_FREQ_DAYS = 20
MIN_WARMUP_OBS = 500
ANNUALIZATION = 252

_DCC_GARCH_REPO = "https://github.com/Pkargados/dcc-garch-python.git"
_DCC_GARCH_CLONE_DIR = Path(__file__).resolve().parent.parent.parent / ".external" / "dcc-garch-python"


def _ensure_dcc_garch_clone() -> Path:
    marker = _DCC_GARCH_CLONE_DIR / "python" / "garch" / "gjr_garch.py"
    if marker.exists():
        return _DCC_GARCH_CLONE_DIR
    _DCC_GARCH_CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", _DCC_GARCH_REPO, str(_DCC_GARCH_CLONE_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone {_DCC_GARCH_REPO}: {result.stderr}")
    return _DCC_GARCH_CLONE_DIR


def _get_garch_functions():
    clone_dir = _ensure_dcc_garch_clone()
    module_path = clone_dir / "python" / "garch" / "gjr_garch.py"
    spec = importlib.util.spec_from_file_location("dcc_garch_gjr_garch", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_gjr_garch, module.filter_gjr_garch


def _asset_walk_forward_vol(returns: pd.Series, verbose: bool = False) -> pd.Series:
    """Point-in-time GJR-GARCH annualized vol for one asset. NaN before
    MIN_WARMUP_OBS (not enough data for a stable fit) and on any refit
    window that fails to converge (skipped, not crashed -- logged and
    continued, same discipline as breakout's per-asset SKIPPED handling).

    Per-asset dynamic rescale before fitting: fit_gjr_garch/filter_gjr_garch
    apply a fixed x100 scale internally, tuned for the ETF returns dcc_garch
    was originally validated on. Some assets in this project's universe
    (e.g. US_2Y -- 2yr Treasury note price returns) have a much smaller
    native return scale, leaving the effective input far outside arch's own
    documented stable range ("estimation works better when [the scale] is
    between 1 and 1000") even after that x100 -- confirmed live via a real
    breakdown: US_2Y's GARCH vol collapsed to ~0.000004 annualized in
    several windows and spiked to 22.4 elsewhere, exploding its QLIKE to
    27,407 against a normal ~0.2-0.6 range for every other asset. Centering
    every asset's effective scale at 100 (comfortably mid-range) before
    fitting, then inverting the extra factor on the way out, fixes the root
    cause the library's own DataScaleWarning was already flagging, not a
    workaround around it. The scale factor is computed from the warmup
    window only (not the full history) so it stays a point-in-time-safe
    numerical-conditioning choice, not a look-ahead one -- it never depends
    on the sign or timing of a return this estimate would need to forecast,
    only a rough magnitude for optimizer stability."""
    clean = returns.dropna()
    if len(clean) < MIN_WARMUP_OBS:
        return pd.Series(np.nan, index=returns.index)

    fit_gjr_garch, filter_gjr_garch = _get_garch_functions()

    values = clean.values
    dates = clean.index

    raw_std = np.std(values[:MIN_WARMUP_OBS])
    if raw_std <= 0 or not np.isfinite(raw_std):
        return pd.Series(np.nan, index=returns.index)
    extra_scale = 1.0 / raw_std
    scaled_values = values * extra_scale

    sigma_pct = np.full(len(values), np.nan)

    refit_points = list(range(MIN_WARMUP_OBS, len(values), REFIT_FREQ_DAYS))
    for i, start in enumerate(refit_points):
        end = refit_points[i + 1] if i + 1 < len(refit_points) else len(values)
        try:
            fit = fit_gjr_garch(scaled_values[:start])
            filtered = filter_gjr_garch(scaled_values[:end], fit["params"])
            sigma_pct[start:end] = filtered["sigmas"][start:end]
        except Exception as exc:
            if verbose:
                print(f"    refit at obs {start} failed: {type(exc).__name__}: {exc}")
            continue

    # sigma_pct is "% daily" of (returns * extra_scale) -- undo extra_scale,
    # then the package's own %-daily convention (/100), then annualize.
    annualized = pd.Series(sigma_pct, index=dates) / extra_scale / 100.0 * np.sqrt(ANNUALIZATION)
    return annualized.reindex(returns.index)


def gjr_garch_volatility(adj_returns: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Walk-forward GJR-GARCH annualized vol, one column per asset -- same
    (T x N) shape/units as data.volatility's Yang-Zhang and
    data.ewma_volatility's EWMA, so it plugs directly into the existing
    vol-estimator comparison framework."""
    cols = {}
    for asset in adj_returns.columns:
        if verbose:
            print(f"Fitting GJR-GARCH: {asset}...")
        cols[asset] = _asset_walk_forward_vol(adj_returns[asset], verbose=verbose)
    return pd.DataFrame(cols)
