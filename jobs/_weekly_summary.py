"""
Weekly Trend / RV / Multi-Strategy backtest + written summary -- the
"evaluation output" step of jobs/weekly_databento_pipeline.py. Reuses
existing, already-validated construction (CLAUDE.md Rule 6) rather than
re-deriving anything:
  - single_strategy_portfolios.build_adopted_books() -> the adopted Trend
    Book (compressed universe, tsmom_alone, GARCH vol-targeted) -- no
    re-run of the flavor bake-off.
  - multi_strategy_relative_value.build_rv_book() -> the pooled RV Book.
  - multi_strategy_relative_value's own EWMA risk-parity pattern
    (ewma_covariance -> risk_parity_weights -> combine_static), fit ONCE on
    TRAIN sleeve PnL and applied as a FIXED weight -- matches the
    already-validated Trend+Carry precedent (WORKFLOW.md decision #12),
    not re-fit walk-forward. Chosen over DCC-GARCH per direct instruction:
    decision #12's own real numbers show no measurable edge from DCC-GARCH
    over EWMA at n=2 sleeves, and EWMA is far cheaper to compute weekly.

Writes Data/reports/weekly_<week_id>.md (human-readable, diffed against the
prior saved JSON) and Data/reports/weekly_<week_id>.json (machine-readable,
becomes next week's diff baseline).
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from single_strategy_portfolios import build_adopted_books, PERIODS_PER_YEAR  # noqa: E402
import multi_strategy_relative_value as msrv  # noqa: E402
from portfolio.sleeve_covariance import ewma_covariance  # noqa: E402
from portfolio.risk_parity import risk_parity_weights  # noqa: E402
from portfolio.risk_metrics import historical_var, expected_shortfall  # noqa: E402
from backtest.splits import TRAIN_END, train_validation_test_split  # noqa: E402
from backtest.performance import simple_sharpe  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent.parent / "Data" / "reports"
EWMA_HALFLIFE = msrv.EWMA_HALFLIFE  # 87, weekly-observation units -- single source of truth


def _pnl_extra_stats(pnl: pd.Series) -> dict:
    """Annualized vol + max drawdown, computed the same way
    multi_strategy_relative_value.py's own pnl_stats() does for an arbitrary
    combined series -- reused construction, not a second definition of
    "max drawdown" that could silently diverge from the standalone one."""
    clean = pnl.dropna()
    if len(clean) < 2:
        return {"ann_vol": float("nan"), "max_dd": float("nan")}
    ann_vol = float(clean.std() * np.sqrt(PERIODS_PER_YEAR))
    cumret = (1 + clean).cumprod()
    max_dd = float(((cumret - cumret.cummax()) / cumret.cummax()).min())
    return {"ann_vol": ann_vol, "max_dd": max_dd}


def _stats(name: str, pnl: pd.Series, turnover=None) -> dict:
    if len(pnl.dropna()) == 0:
        return {"name": name, "insufficient_data": True}
    train, validation, test = train_validation_test_split(pnl)
    extra = _pnl_extra_stats(pnl)
    return {
        "name": name,
        "sharpe_train": simple_sharpe(train, periods_per_year=PERIODS_PER_YEAR),
        "sharpe_validation": simple_sharpe(validation, periods_per_year=PERIODS_PER_YEAR),
        "sharpe_test": simple_sharpe(test, periods_per_year=PERIODS_PER_YEAR),
        # Turnover isn't a well-defined single number for a COMBINED
        # multi-book pnl series (Allocator doesn't track it) -- left NaN
        # rather than faked when not supplied by a real per-Book result.
        "turnover": turnover if turnover is not None else float("nan"),
        "max_dd": extra["max_dd"],
        "ann_vol": extra["ann_vol"],
        "var95": historical_var(pnl, confidence=0.95),
        "es95": expected_shortfall(pnl, confidence=0.95),
        "n_obs": int(len(pnl.dropna())),
    }


def run_backtests() -> dict:
    """Builds Trend, RV, and the EWMA-risk-parity Trend+RV combination.
    Pure composition of already-validated functions -- no new backtest
    logic."""
    returns_ts, trend_book, _carry_book = build_adopted_books()
    rv_book, returns_rv = msrv.build_rv_book()

    trend_result = trend_book.run(returns_ts)
    rv_result = rv_book.run(returns_rv)

    sleeve_returns = pd.concat({"trend": trend_result["pnl"], "relative_value": rv_result["pnl"]}, axis=1)
    train_sleeve_returns = sleeve_returns.loc[:TRAIN_END]
    cov_ewma = ewma_covariance(train_sleeve_returns, halflife=EWMA_HALFLIFE)
    w_ewma = risk_parity_weights(cov_ewma.values)
    combined_pnl = msrv.combine_static(
        {"trend": 2 * w_ewma[0], "relative_value": 2 * w_ewma[1]}, sleeve_returns
    )

    return {
        "trend": _stats("trend", trend_result["pnl"], turnover=trend_result.get("turnover")),
        "relative_value": _stats("relative_value", rv_result["pnl"], turnover=rv_result.get("turnover")),
        "combined_ewma_risk_parity": _stats("combined_ewma_risk_parity", combined_pnl),
        "combination_weights": {"trend": float(w_ewma[0]) * 2, "relative_value": float(w_ewma[1]) * 2},
    }


def _load_prior_report() -> dict:
    """Most recently-dated weekly_*.json in Data/reports/, if any -- the
    diff baseline. None on the very first run, handled explicitly."""
    if not REPORTS_DIR.exists():
        return None
    candidates = sorted(REPORTS_DIR.glob("weekly_*.json"))
    if not candidates:
        return None
    with open(candidates[-1], "r") as f:
        return json.load(f)


def _diff_line(label: str, key: str, current: dict, prior: dict) -> str:
    cur_val = current.get(key)
    if not prior or key not in prior:
        return f"{label}: {cur_val:.3f}" if isinstance(cur_val, (int, float)) and not isinstance(cur_val, bool) else f"{label}: {cur_val}"
    prior_val = prior.get(key)
    if isinstance(cur_val, (int, float)) and isinstance(prior_val, (int, float)):
        delta = cur_val - prior_val
        sign = "+" if delta >= 0 else ""
        return f"{label}: {cur_val:.3f} ({sign}{delta:.3f} vs. last week)"
    return f"{label}: {cur_val}"


def write_report(week_id: str, results: dict) -> tuple:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prior = _load_prior_report()

    json_path = REPORTS_DIR / f"weekly_{week_id}.json"
    md_path = REPORTS_DIR / f"weekly_{week_id}.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    lines = [f"# Weekly Pipeline Report -- {week_id}", ""]
    for book_key in ("trend", "relative_value", "combined_ewma_risk_parity"):
        stats = results[book_key]
        prior_stats = (prior or {}).get(book_key, {})
        lines.append(f"## {stats.get('name', book_key)}")
        if stats.get("insufficient_data"):
            lines.append("- INSUFFICIENT DATA")
        else:
            lines.append("- " + _diff_line("Sharpe (train)", "sharpe_train", stats, prior_stats))
            lines.append("- " + _diff_line("Sharpe (validation)", "sharpe_validation", stats, prior_stats))
            lines.append("- " + _diff_line("Sharpe (test)", "sharpe_test", stats, prior_stats))
            lines.append("- " + _diff_line("Turnover", "turnover", stats, prior_stats))
            lines.append("- " + _diff_line("Max drawdown", "max_dd", stats, prior_stats))
            lines.append("- " + _diff_line("95% VaR (weekly)", "var95", stats, prior_stats))
            lines.append("- " + _diff_line("95% ES (weekly)", "es95", stats, prior_stats))
            lines.append(f"- Observations: {stats.get('n_obs')}")
        lines.append("")

    weights = results["combination_weights"]
    lines.append("## Combination weights (EWMA risk-parity, fixed, fit on TRAIN)")
    lines.append(f"- Trend: {weights['trend']:.3f}  Relative Value: {weights['relative_value']:.3f}")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {md_path.name} and {json_path.name}")
    return md_path, json_path


def run(week_id: str = None) -> tuple:
    week_id = week_id or dt.date.today().isoformat()
    results = run_backtests()
    return write_report(week_id, results)


if __name__ == "__main__":
    run()
