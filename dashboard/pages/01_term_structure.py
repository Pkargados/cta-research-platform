"""Page 01 — Term Structure / Forward Curves.

Reads: Data/dashboard_summary/term_structure_curve.parquet,
       Data/dashboard_summary/term_structure_summary.csv,
       Data/dashboard_summary/term_structure_curve_shape_history.parquet
       Data/term_structure_manifest.csv (raw, for contract-count-over-time)
       Data/term_structure.parquet, term_structure_spreads.parquet,
       term_structure_butterflies.parquet (raw, for the per-instrument drill-down and
       richness badge below — small, already-filtered reads at render time, same
       pattern as the manifest-based coverage chart already on this page; not the kind
       of heavy signal-generation computation that belongs in the precompute job).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import (
    apply_chart_theme,
    DATA_DIR, page_header, render_key_takeaways, require_summary_files,
    load_csv, load_parquet, CATEGORICAL, DIVERGING, colored_badge, price_richness,
)

page_header("Term Structure / Forward Curves", "What does each asset's forward curve look like right now?")

require_summary_files("term_structure_curve.parquet", "term_structure_summary.csv")
curve = load_parquet("term_structure_curve.parquet")
summary = load_csv("term_structure_summary.csv")

n_contango = (summary["curve_shape"] == "Contango").sum()
n_backwardation = (summary["curve_shape"] == "Backwardation").sum()
render_key_takeaways([
    f"**{n_contango} assets in contango**, **{n_backwardation} in backwardation** "
    f"across {len(summary)} assets with a captured curve.",
    "Curves are built from the forward-capture archive (started 2026-07-14) — depth "
    "is currently limited to whatever's been captured since then, not full history. "
    "See the Databento historical backfill status in WORKFLOW.md Phase 4.",
])

st.divider()

instrument_type = st.radio(
    "Instrument type", ["Outrights", "Calendar Spreads", "Inter-Commodity Spreads", "Butterflies"],
    horizontal=True,
)


# ------------------------------------------------------------------
# Outrights — today's curve (existing view) + historical curve shape +
# single-contract drill-down (new, only meaningful where deep history exists)
# ------------------------------------------------------------------
if instrument_type == "Outrights":
    asset = st.selectbox("Select asset", sorted(curve["asset"].unique()))
    asset_curve = curve[curve["asset"] == asset].copy()
    asset_summary = summary[summary["asset"] == asset].iloc[0] if asset in summary["asset"].values else None

    if asset_summary is not None:
        shape = asset_summary["curve_shape"]
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Curve shape**")
        c1.markdown(colored_badge(shape, DIVERGING.get(shape, "#898781")))
        c2.metric("Contracts captured", int(asset_summary["n_contracts"]))
        c3.metric("Front contract close", f"{asset_summary['front_close']:,.3f}")

    st.subheader(f"Forward Curve — {asset}")

    if len(asset_curve) < 2:
        st.info("Only one contract captured for this asset so far — a curve needs at least two.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=asset_curve["expiry_label"], y=asset_curve["close"],
            mode="lines+markers",
            line=dict(color=CATEGORICAL[0], width=2),
            marker=dict(size=9),
        ))
        fig.update_layout(
            xaxis_title="Contract month", yaxis_title="Price",
            height=420, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    with st.expander("Contract detail"):
        st.dataframe(
            asset_curve[["contract_symbol", "expiry_label", "close", "last_price_date"]]
            .rename(columns={"contract_symbol": "Contract", "expiry_label": "Expiry", "close": "Close", "last_price_date": "Last Price Date"}),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # --- historical curve shape + single-contract drill-down --------
    shape_history_path = Path(DATA_DIR) / "dashboard_summary" / "term_structure_curve_shape_history.parquet"
    shape_history = pd.read_parquet(shape_history_path) if shape_history_path.exists() else pd.DataFrame()
    asset_shape_history = shape_history[shape_history["asset"] == asset] if not shape_history.empty else pd.DataFrame()

    if asset_shape_history.empty:
        st.info(
            f"No historical curve-shape depth for {asset} yet — that needs the Databento "
            "historical backfill (WORKFLOW.md Phase 4), not just forward-capture. "
            "LiveCattle has it; more assets will gain it as their pulls complete."
        )
    else:
        st.subheader(f"Curve Shape History — {asset}")
        fig_shape = go.Figure()
        fig_shape.add_trace(go.Scatter(
            x=asset_shape_history["date"], y=asset_shape_history["steepness_pct"],
            mode="lines",
            line=dict(color=CATEGORICAL[0], width=1),
            fill="tozeroy",
            name="(Back − Front) / Front",
        ))
        fig_shape.add_hline(y=0, line_dash="dash", line_color="#898781")
        fig_shape.update_layout(
            xaxis_title="Date", yaxis_title="Curve steepness, % (>0 = Contango, <0 = Backwardation)",
            height=320, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_shape = apply_chart_theme(fig_shape)
        st.plotly_chart(fig_shape, use_container_width=True, theme="streamlit")
        st.caption(
            "Normalized (% of front close), not raw $ — LiveCattle's price level moved "
            "roughly $90 → $250 over this history, so a fixed dollar gap means something "
            "different depending on when it occurs."
        )

        pct_contango = (asset_shape_history["curve_shape"] == "Contango").mean() * 100
        st.caption(
            f"Contango {pct_contango:.0f}% of {len(asset_shape_history)} captured days "
            f"({asset_shape_history['date'].min().date()} → {asset_shape_history['date'].max().date()})."
        )

        st.subheader(f"Term Structure Heatmap — {asset}")
        rank_history_path = Path(DATA_DIR) / "dashboard_summary" / "term_structure_curve_rank_history.parquet"
        rank_history = pd.read_parquet(rank_history_path) if rank_history_path.exists() else pd.DataFrame()
        asset_rank_history = rank_history[rank_history["asset"] == asset] if not rank_history.empty else pd.DataFrame()

        if asset_rank_history.empty:
            st.info("No rank-history depth for this asset yet.")
        else:
            pivot = asset_rank_history.pivot(index="rank", columns="date", values="close_minus_front")
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index,
                colorscale=[[0, "#e34948"], [0.5, "#f5f0e8"], [1, "#2a78d6"]],
                zmid=0, colorbar=dict(title="Close − Front"),
            ))
            fig_heat.update_layout(
                xaxis_title="Date", yaxis_title="Rank from front (0 = nearest)",
                height=380, margin=dict(t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_heat = apply_chart_theme(fig_heat)
            st.plotly_chart(fig_heat, use_container_width=True, theme="streamlit")
            st.caption(
                "Ranked by position along the curve (0 = nearest contract), not absolute "
                "contract month — absolute months roll off every few weeks, which would "
                "make a heatmap indexed by them mostly empty. Blue = that rung trading "
                "above the front contract (contango at that point on the curve), red = below."
            )

        st.subheader(f"Single-Contract History — {asset}")
        ts_raw = pd.read_parquet(DATA_DIR / "term_structure.parquet")
        ts_raw = ts_raw[(ts_raw["asset"] == asset) & (ts_raw["volume"] > 0)]
        contract_volume = ts_raw.groupby("contract_symbol")["volume"].sum().sort_values(ascending=False)
        contract_symbol = st.selectbox(
            "Select a specific contract", contract_volume.index.tolist(),
            index=0, help="Sorted by total observed volume (most liquid first).",
        )
        contract_series = ts_raw[ts_raw["contract_symbol"] == contract_symbol].sort_values("date")
        fig_contract = go.Figure()
        fig_contract.add_trace(go.Scatter(
            x=contract_series["date"], y=contract_series["close"],
            mode="lines", line=dict(color=CATEGORICAL[1], width=1.5),
        ))
        fig_contract.update_layout(
            xaxis_title="Date", yaxis_title="Close",
            height=320, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_contract = apply_chart_theme(fig_contract)
        st.plotly_chart(fig_contract, use_container_width=True, theme="streamlit")

        with st.expander("Volume"):
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=contract_series["date"], y=contract_series["volume"],
                marker=dict(color=CATEGORICAL[1]),
            ))
            fig_vol.update_layout(
                xaxis_title="Date", yaxis_title="Volume",
                height=220, margin=dict(t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_vol = apply_chart_theme(fig_vol)
            st.plotly_chart(fig_vol, use_container_width=True, theme="streamlit")

    st.divider()

    st.subheader(f"Capture Coverage Over Time — {asset}")
    manifest = pd.read_csv(DATA_DIR / "term_structure_manifest.csv", parse_dates=["run_date"])
    asset_hist = manifest[manifest["asset"] == asset].sort_values("run_date")

    if asset_hist.empty:
        st.info("No capture history found for this asset in term_structure_manifest.csv.")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=asset_hist["run_date"], y=asset_hist["contracts_pulled"],
            mode="lines+markers",
            line=dict(color=CATEGORICAL[0], width=2),
        ))
        fig2.update_layout(
            xaxis_title="Run date", yaxis_title="Contracts captured",
            height=320, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig2 = apply_chart_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
        st.caption(
            "This will become more meaningful once the Databento historical backfill's "
            "transform/join stage lands (WORKFLOW.md Phase 4) — right now it only shows "
            "the short forward-capture window since 2026-07-14."
        )

    st.divider()

    st.subheader("All Assets — Curve Shape")
    disp = summary[["asset", "n_contracts", "curve_shape", "front_close", "back_close", "status"]].sort_values("asset")
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# Calendar Spreads / Inter-Commodity Spreads / Butterflies — per-instrument
# drill-down + richness badge. New (2026-07-16), Databento-sourced only — no
# yfinance forward-capture equivalent exists for these instrument types.
# ------------------------------------------------------------------
else:
    spreads_path = DATA_DIR / "term_structure_spreads.parquet"
    bf_path = DATA_DIR / "term_structure_butterflies.parquet"

    if instrument_type in ("Calendar Spreads", "Inter-Commodity Spreads"):
        if not spreads_path.exists():
            st.info("No spread data transformed yet. See WORKFLOW.md Phase 4.")
            st.stop()
        spreads = pd.read_parquet(spreads_path)
        is_calendar = spreads["near_root"] == spreads["far_root"]
        data = spreads[is_calendar] if instrument_type == "Calendar Spreads" else spreads[~is_calendar]
        leg_cols = [
            "near_root", "near_contract_symbol", "near_expiry_year", "near_expiry_code",
            "far_root", "far_contract_symbol", "far_expiry_year", "far_expiry_code",
        ]
        key_cols = ["near_contract_symbol", "far_contract_symbol"]
    else:
        if not bf_path.exists():
            st.info("No butterfly data transformed yet. See WORKFLOW.md Phase 4.")
            st.stop()
        data = pd.read_parquet(bf_path)
        leg_cols = [
            "near_root", "near_contract_symbol", "near_expiry_year", "near_expiry_code",
            "mid_root", "mid_contract_symbol", "mid_expiry_year", "mid_expiry_code",
            "far_root", "far_contract_symbol", "far_expiry_year", "far_expiry_code",
        ]
        key_cols = ["near_contract_symbol", "mid_contract_symbol", "far_contract_symbol"]

    if data.empty:
        st.info(f"No {instrument_type.lower()} found in the transformed data yet.")
        st.stop()

    # Key on the disambiguated leg-contract-symbol tuple, never the raw spread_symbol/
    # butterfly_symbol string - that string is decade-ambiguous (DATA_QUALITY_REPORT.md
    # Section 1, the same collision documented for outrights: e.g. "LEM6-LEQ6" matches
    # both a real Jun/Aug-2016 spread and an unrelated real Jun/Aug-2026 spread). Found
    # live on 2026-07-16: grouping by the raw string silently stitched two unrelated
    # real instruments into one series, and the multi-year gap between them rendered as
    # a straight diagonal line connecting 2016's last point to 2026's first - not a data
    # gap, a wrong grouping key. Affects effectively every spread/butterfly symbol, not
    # a rare edge case, since the 16-year pull window means most month-pairs occurring
    # early recur with the same digit a decade later.
    data = data.copy()
    data["_display_symbol"] = data[key_cols].agg(" / ".join, axis=1)

    assets_available = sorted(data["asset"].unique())
    asset = st.selectbox("Select asset", assets_available)
    asset_data = data[data["asset"] == asset]

    symbol_volume = asset_data.groupby("_display_symbol")["volume"].sum().sort_values(ascending=False)
    symbol = st.selectbox(
        f"Select a specific {'spread' if 'Spread' in instrument_type else 'butterfly'}",
        symbol_volume.index.tolist(), index=0,
        help="Sorted by total observed volume (most liquid first). Labeled by disambiguated "
             "contract symbols, not the raw (decade-ambiguous) spread/butterfly symbol.",
    )
    instrument_series = asset_data[asset_data["_display_symbol"] == symbol].sort_values("date")

    richness = price_richness(instrument_series["close"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest close", f"{instrument_series['close'].iloc[-1]:,.3f}")
    if richness["percentile"] is not None:
        c2.metric("Percentile in own history", f"{richness['percentile']:.0f}%")
        c3.metric("Z-score vs. own history", f"{richness['zscore']:.2f}" if richness["zscore"] is not None else "N/A")
    else:
        c2.markdown("*Not enough history yet for a richness read*")
    st.caption(
        "Percentile/z-score are descriptive only — where today's price sits relative to "
        f"this instrument's own {richness['n_obs']} observed closes, not a trade signal."
    )

    st.subheader(f"Price History — {symbol}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=instrument_series["date"], y=instrument_series["close"],
        mode="lines", line=dict(color=CATEGORICAL[2], width=1.5),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#898781")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Close",
        height=380, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig = apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    with st.expander("Volume"):
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=instrument_series["date"], y=instrument_series["volume"],
            marker=dict(color=CATEGORICAL[2]),
        ))
        fig_vol.update_layout(
            xaxis_title="Date", yaxis_title="Volume",
            height=220, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_vol = apply_chart_theme(fig_vol)
        st.plotly_chart(fig_vol, use_container_width=True, theme="streamlit")

    # --- carry + synthetic-vs-real validation (calendar spreads only — needs a
    # same-underlying near/far pair, and both legs' own outright history to join
    # against; inter-commodity pairs and butterflies don't fit this shape) --------
    if instrument_type == "Calendar Spreads":
        MONTH_CODES = "FGHJKMNQUVXZ"  # matches capture_term_structure.py
        near_symbol = instrument_series["near_contract_symbol"].iloc[0]
        far_symbol = instrument_series["far_contract_symbol"].iloc[0]
        near_year, near_code = instrument_series["near_expiry_year"].iloc[0], instrument_series["near_expiry_code"].iloc[0]
        far_year, far_code = instrument_series["far_expiry_year"].iloc[0], instrument_series["far_expiry_code"].iloc[0]
        months_apart = (far_year * 12 + MONTH_CODES.index(far_code)) - (near_year * 12 + MONTH_CODES.index(near_code))
        days_between = months_apart * 30.44  # expiry is only known to month granularity in this schema

        ts_raw_all = pd.read_parquet(DATA_DIR / "term_structure.parquet")
        near_leg = ts_raw_all[ts_raw_all["contract_symbol"] == near_symbol][["date", "close"]].rename(columns={"close": "near_close"})
        far_leg = ts_raw_all[ts_raw_all["contract_symbol"] == far_symbol][["date", "close"]].rename(columns={"close": "far_close"})

        carry_df = instrument_series[["date", "close"]].rename(columns={"close": "spread_close"})
        carry_df = carry_df.merge(near_leg, on="date", how="inner").merge(far_leg, on="date", how="inner")

        if days_between > 0 and not carry_df.empty:
            carry_df["carry_pct"] = carry_df["spread_close"] / carry_df["near_close"] * 365 / days_between * 100
            carry_df["synthetic_spread"] = carry_df["near_close"] - carry_df["far_close"]

            st.subheader(f"Carry — {symbol}")
            c1, c2 = st.columns(2)
            c1.metric("Latest annualized carry", f"{carry_df['carry_pct'].iloc[-1]:.2f}%")
            c2.metric("Days between expiries (approx.)", f"{days_between:.0f}")

            fig_carry = go.Figure()
            fig_carry.add_trace(go.Scatter(
                x=carry_df["date"], y=carry_df["carry_pct"], mode="lines",
                line=dict(color=CATEGORICAL[3], width=1.5),
            ))
            fig_carry.add_hline(y=0, line_dash="dash", line_color="#898781")
            fig_carry.update_layout(
                xaxis_title="Date", yaxis_title="Annualized carry, %",
                height=300, margin=dict(t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_carry = apply_chart_theme(fig_carry)
            st.plotly_chart(fig_carry, use_container_width=True, theme="streamlit")
            st.caption(
                "Carry = (F_near − F_far) / F_near × 365 / days between expiries "
                "(WORKFLOW.md Phase 4), using this spread's own quoted close for "
                "F_near − F_far — not back-differenced — and the near leg's outright "
                "close for F_near."
            )

            st.subheader("Real Quote vs. Back-Differenced Synthetic")
            fig_synth = go.Figure()
            fig_synth.add_trace(go.Scatter(
                x=carry_df["date"], y=carry_df["spread_close"], mode="lines",
                name="Real quoted spread", line=dict(color=CATEGORICAL[0], width=1.5),
            ))
            fig_synth.add_trace(go.Scatter(
                x=carry_df["date"], y=carry_df["synthetic_spread"], mode="lines",
                name="Synthetic (near close − far close)", line=dict(color=CATEGORICAL[5], width=1.5, dash="dot"),
            ))
            fig_synth.update_layout(
                xaxis_title="Date", yaxis_title="Spread",
                height=320, margin=dict(t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.15),
            )
            fig_synth = apply_chart_theme(fig_synth)
            st.plotly_chart(fig_synth, use_container_width=True, theme="streamlit")

            residual = (carry_df["spread_close"] - carry_df["synthetic_spread"]).abs()
            st.caption(
                f"Mean absolute difference between the real quote and the back-differenced "
                f"synthetic: {residual.mean():.3f} (median {residual.median():.3f}) over "
                f"{len(carry_df)} overlapping days — this is the leg-alignment risk "
                "DATA_QUALITY_REPORT.md flagged as a reason to prefer the real quoted spread "
                "over back-differencing two separate outright closes."
            )
        else:
            st.info("Not enough overlapping outright data for both legs to compute carry / the synthetic comparison.")

    with st.expander("Leg detail"):
        leg_detail = asset_data[asset_data["_display_symbol"] == symbol][leg_cols].drop_duplicates()
        st.dataframe(leg_detail, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"All {instrument_type} — {asset}")
    counts_disp = symbol_volume.reset_index()
    counts_disp.columns = ["contract_symbols", "total_volume"]
    st.dataframe(counts_disp, use_container_width=True, hide_index=True)
