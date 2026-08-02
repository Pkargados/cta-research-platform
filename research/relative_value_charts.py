"""
research/relative_value_charts.py — Static chart exports for the Relative
Value sleeve (WORKFLOW.md §11d, extended by the 2026-08-01 follow-on session
documented in that day's handoff, not yet folded into WORKFLOW.md itself).

Writes one PNG tearsheet per pair (11) plus one pooled-Book tearsheet and one
pooled-Book attribution/correlation panel to `charts/relative_value/`. Reuses
`relative_value.py`'s and `relative_value_book.py`'s already-computed
bake-off/Book outputs directly — no new backtest logic here (CLAUDE.md Rule 6:
this is a display layer over already-validated pipeline outputs, the same
relationship `dashboard/pages/26_relative_value_performance.py` has to the
same two modules).

Per-pair construction: reuses `relative_value.py`'s own hedge-ratio-method
winners (the `spreads` dict from its bake-off) and generalizes
`relative_value_book.pick_entry_exit_winners`'s selection rule to also pick
FREQUENCY per pair (not just entry/exit spec), by validation NET Sharpe over
the already-computed 28-row bake-off table — same metric, same discipline,
one more already-tabulated dimension, not a new bake-off. The pooled Book
itself stays pinned to weekly (that's a separate, already-settled decision,
not relitigated here).

Styling: `dashboard/lib.py`'s CATEGORICAL palette and `apply_chart_theme`, not
a new palette — that module imports `streamlit` at module level but none of
the functions reused here call any `st.*` API, so importing it outside a
`streamlit run` context is safe (verified directly: only a harmless "no
runtime found" cache warning is printed).

Run: `python research/relative_value_charts.py` from the repo root.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "research"))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

import relative_value as rv
import relative_value_book as rvb
from lib import CATEGORICAL, apply_chart_theme

from signals.relative_value import (
    ALL_PAIR_NAMES, build_pair_signal, spread_return, zscore_spread_signal,
    DEFAULT_ENTRY_Z, DEFAULT_EXIT_Z,
)
from backtest.engine import normalized_positions
from backtest.costs import turnover as turnover_fn
from backtest.performance import simple_sharpe
from backtest.splits import TRAIN_END, VALIDATION_END, train_validation_test_split
from portfolio.risk_metrics import expanding_var_and_es

OUT_DIR = REPO_ROOT / "charts" / "relative_value"
PAIRS_DIR = OUT_DIR / "pairs"
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

GROSS_COLOR = CATEGORICAL[0]
NET_COLOR = CATEGORICAL[5]
POS_COLOR = CATEGORICAL[1]
NEG_COLOR = CATEGORICAL[5]
ZSCORE_COLOR = CATEGORICAL[4]
TURNOVER_COLOR = CATEGORICAL[7]
ROLLING_SHARPE_COLOR = CATEGORICAL[3]

SPLIT_LINE_COLOR = "#898781"


def _one_col(series: pd.Series, col: str = "pair") -> pd.DataFrame:
    return series.to_frame(col)


def add_split_lines(fig, row, col):
    for x in (TRAIN_END, VALIDATION_END):
        fig.add_vline(x=x, row=row, col=col, line_dash="dash", line_color=SPLIT_LINE_COLOR, line_width=1)


def equity_curves(gross: pd.Series, net: pd.Series):
    return (1 + gross.dropna()).cumprod(), (1 + net.dropna()).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.dropna()).cumprod()
    return wealth / wealth.cummax() - 1


def rolling_sharpe_series(returns: pd.Series, window: int, periods_per_year: int) -> pd.Series:
    r = returns.dropna()
    return (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(periods_per_year)


def sharpe_by_period(returns: pd.Series, periods_per_year: int = 252):
    tr, va, te = train_validation_test_split(returns)
    return [
        simple_sharpe(tr, periods_per_year=periods_per_year),
        simple_sharpe(va, periods_per_year=periods_per_year),
        simple_sharpe(te, periods_per_year=periods_per_year),
    ]


def monthly_returns_matrix(returns: pd.Series) -> pd.DataFrame:
    r = returns.dropna()
    if len(r) == 0:
        return pd.DataFrame()
    monthly = (1 + r).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    return pivot.reindex(columns=range(1, 13))


def add_heatmap(fig, pivot: pd.DataFrame, row: int, col: int):
    if pivot.empty:
        return
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z = (pivot.values * 100)
    fig.add_trace(
        go.Heatmap(
            z=z, x=month_labels, y=[str(y) for y in pivot.index],
            colorscale="RdBu", zmid=0, showscale=False,
            hovertemplate="Year %{y}, %{x}: %{z:.2f}%<extra></extra>",
        ),
        row=row, col=col,
    )


# ---------------------------------------------------------------------------
# Per-pair winning construction: generalizes pick_entry_exit_winners to also
# pick frequency, from the already-computed bake-off table.
# ---------------------------------------------------------------------------

def pick_pair_construction(result: pd.DataFrame, pair_name: str):
    sub = result.xs(pair_name, level="pair")["validation_net"].dropna()
    if len(sub) == 0:
        return "continuous", "weekly"
    return sub.idxmax()


def make_pair_tearsheet(pair_name: str, spread: pd.Series, spec: str, frequency: str, cost_value: float):
    signal = build_pair_signal(spread, entry_exit=spec, target_vol=rv.TARGET_VOL_SIGNAL)
    ret = spread_return(spread)
    gross, net = rv.backtest_pair(signal, ret, cost_value, frequency)
    positions = normalized_positions(_one_col(signal), frequency)
    to = turnover_fn(positions).sum(axis=1)
    z = zscore_spread_signal(spread)

    ppy = 252  # both frequencies trade daily-return series (only rebalance cadence differs) — see module docstring reasoning in relative_value.py
    roll_window = 252

    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=[
            "Equity Curve (Gross vs Net)",
            "Sharpe by Period", "Drawdown (Net)",
            "Rolling 1Y Sharpe (Gross)", "Monthly Returns, % (Net)",
            "Spread Z-Score & Entry/Exit Bands", "Annualized Turnover (Rolling)",
        ],
        vertical_spacing=0.07, horizontal_spacing=0.10,
        row_heights=[0.28, 0.24, 0.24, 0.24],
    )

    eq_g, eq_n = equity_curves(gross, net)
    fig.add_trace(go.Scatter(x=eq_g.index, y=eq_g.values, name="Gross", line=dict(color=GROSS_COLOR, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_n.index, y=eq_n.values, name="Net", line=dict(color=NET_COLOR, width=1.6)), row=1, col=1)
    add_split_lines(fig, row=1, col=1)

    periods = ["Train", "Validation", "Test"]
    fig.add_trace(go.Bar(x=periods, y=sharpe_by_period(gross, ppy), name="Gross", marker_color=GROSS_COLOR, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=periods, y=sharpe_by_period(net, ppy), name="Net", marker_color=NET_COLOR, showlegend=False), row=2, col=1)

    dd = drawdown_series(net)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy", line=dict(color=NEG_COLOR, width=1), showlegend=False), row=2, col=2)

    rs = rolling_sharpe_series(gross, roll_window, ppy)
    fig.add_trace(go.Scatter(x=rs.index, y=rs.values, line=dict(color=ROLLING_SHARPE_COLOR, width=1.2), showlegend=False), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1, line_color=SPLIT_LINE_COLOR, line_width=1)

    add_heatmap(fig, monthly_returns_matrix(net), row=3, col=2)

    fig.add_trace(go.Scatter(x=z.index, y=z.values, line=dict(color=ZSCORE_COLOR, width=1), showlegend=False), row=4, col=1)
    for level, color in ((DEFAULT_ENTRY_Z, NEG_COLOR), (-DEFAULT_ENTRY_Z, NEG_COLOR), (DEFAULT_EXIT_Z, POS_COLOR), (-DEFAULT_EXIT_Z, POS_COLOR)):
        fig.add_hline(y=level, row=4, col=1, line_dash="dot", line_color=color, line_width=1)

    to_roll_window = 63 if frequency == "daily" else 13
    to_roll = to.rolling(to_roll_window).mean() * 252
    fig.add_trace(go.Scatter(x=to_roll.index, y=to_roll.values, line=dict(color=TURNOVER_COLOR, width=1.2), showlegend=False), row=4, col=2)

    fig.update_layout(
        title=f"Relative Value — {pair_name}  (winning spec: {spec}, {frequency})",
        height=1500, width=1250,
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.04),
        margin=dict(t=100),
    )
    fig = apply_chart_theme(fig)
    fig.write_image(str(PAIRS_DIR / f"{pair_name}.png"), scale=2)
    print(f"  wrote {pair_name}.png (spec={spec}, frequency={frequency})")


def make_pooled_book_tearsheet(alpha_df: pd.DataFrame, returns_df: pd.DataFrame, cost_bps: pd.Series):
    book_gross = rvb.ssp.build_book("relative_value", alpha_df, returns_df, cost_bps=None)
    result_gross = book_gross.run(returns_df)
    book_net = rvb.ssp.build_book("relative_value", alpha_df, returns_df, cost_bps=cost_bps)
    result_net = book_net.run(returns_df)

    pnl_g, pnl_n = result_gross["pnl"], result_net["pnl"]
    ppy = rvb.ssp.PERIODS_PER_YEAR  # 52 — Book runs at weekly cadence

    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=[
            "Equity Curve (Gross vs Net)",
            "Sharpe by Period", "Drawdown (Net)",
            "Rolling 1Y Sharpe (Gross)", "Monthly Returns, % (Net)",
            "Annualized Turnover (Rolling)", "Expanding 95% VaR / ES (Net)",
        ],
        vertical_spacing=0.07, horizontal_spacing=0.10,
        row_heights=[0.28, 0.24, 0.24, 0.24],
    )

    eq_g, eq_n = equity_curves(pnl_g, pnl_n)
    fig.add_trace(go.Scatter(x=eq_g.index, y=eq_g.values, name="Gross", line=dict(color=GROSS_COLOR, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_n.index, y=eq_n.values, name="Net", line=dict(color=NET_COLOR, width=1.8)), row=1, col=1)
    add_split_lines(fig, row=1, col=1)

    periods = ["Train", "Validation", "Test"]
    fig.add_trace(go.Bar(x=periods, y=sharpe_by_period(pnl_g, ppy), name="Gross", marker_color=GROSS_COLOR, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=periods, y=sharpe_by_period(pnl_n, ppy), name="Net", marker_color=NET_COLOR, showlegend=False), row=2, col=1)

    dd = drawdown_series(pnl_n)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy", line=dict(color=NEG_COLOR, width=1), showlegend=False), row=2, col=2)

    rs = rolling_sharpe_series(pnl_g, 52, ppy)
    fig.add_trace(go.Scatter(x=rs.index, y=rs.values, line=dict(color=ROLLING_SHARPE_COLOR, width=1.2), showlegend=False), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1, line_color=SPLIT_LINE_COLOR, line_width=1)

    add_heatmap(fig, monthly_returns_matrix(pnl_n), row=3, col=2)

    to_series = result_net.get("turnover_series")
    if to_series is not None and len(to_series):
        to_roll = to_series.rolling(13).mean() * ppy
        fig.add_trace(go.Scatter(x=to_roll.index, y=to_roll.values, line=dict(color=TURNOVER_COLOR, width=1.2), showlegend=False), row=4, col=1)

    var_es = expanding_var_and_es(pnl_n.dropna(), confidence=0.95)
    if len(var_es):
        fig.add_trace(go.Scatter(x=var_es.index, y=var_es["var"] * 100, name="VaR 95%", line=dict(color=CATEGORICAL[2], width=1.2)), row=4, col=2)
        fig.add_trace(go.Scatter(x=var_es.index, y=var_es["es"] * 100, name="ES 95%", line=dict(color=NEG_COLOR, width=1.2)), row=4, col=2)

    fig.update_layout(
        title="Relative Value — Pooled Book (Ledoit-Wolf, weekly)",
        height=1500, width=1250,
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.04),
        margin=dict(t=100),
    )
    fig = apply_chart_theme(fig)
    fig.write_image(str(OUT_DIR / "pooled_book_tearsheet.png"), scale=2)
    print("  wrote pooled_book_tearsheet.png")
    return result_net


def make_pooled_attribution_chart(result_net: dict, active_pairs: list):
    contributions = result_net["asset_contributions"][active_pairs]
    cum_contrib = contributions.cumsum()
    corr = contributions.corr()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Cumulative Gross Contribution by Pair", "Active-Pair Correlation Matrix (weekly contribution)"],
        row_heights=[0.55, 0.45], vertical_spacing=0.12,
    )
    for i, col in enumerate(cum_contrib.columns):
        fig.add_trace(go.Scatter(x=cum_contrib.index, y=cum_contrib[col], name=col, line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=1.6)), row=1, col=1)
    add_split_lines(fig, row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdBu", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            showscale=True,
        ),
        row=2, col=1,
    )
    fig.update_yaxes(automargin=True, row=2, col=1)
    fig.update_xaxes(automargin=True, row=2, col=1)

    fig.update_layout(
        title=dict(text="Relative Value — Pooled Book Attribution & Active-Pair Correlation", y=0.995),
        height=1250, width=1200,
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5),
        margin=dict(t=220, l=130, r=130),
    )
    fig = apply_chart_theme(fig)
    fig.write_image(str(OUT_DIR / "pooled_book_attribution.png"), scale=2)
    print("  wrote pooled_book_attribution.png")


def main():
    print("=== Relative Value chart export ===")
    print("Running standalone bake-off (reused for both per-pair and pooled-Book charts)...")
    standalone = rv.main()
    result, hedge_winners, spreads = standalone

    _, volume, included = rv.load_and_prepare_data()
    from backtest.costs import liquidity_tiered_cost_bps
    cost_by_leg = liquidity_tiered_cost_bps(volume[included], window_start=rv.ADV_WINDOW_START)

    print("\n=== Per-pair tearsheets (11 pairs) ===")
    for pair_name in ALL_PAIR_NAMES:
        spec, frequency = pick_pair_construction(result, pair_name)
        cost_value = rv.pair_cost_bps(cost_by_leg, pair_name)
        make_pair_tearsheet(pair_name, spreads[pair_name], spec, frequency, cost_value)

    print("\n=== Pooled Relative Value Book tearsheet ===")
    alpha_df, returns_df, active, cost_bps = rvb.prepare_rv_book_inputs(standalone=standalone)
    result_net = make_pooled_book_tearsheet(alpha_df, returns_df, cost_bps)

    print("\n=== Pooled Book attribution & correlation ===")
    make_pooled_attribution_chart(result_net, active)

    print(f"\nDone. Charts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
