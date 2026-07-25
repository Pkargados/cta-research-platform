"""
Shared, display-only helpers for the CTA data QA dashboard.

Pattern borrowed from an unrelated AQR case-study Streamlit submission (thin
ui_components module imported by every page, page files do no analytical work of
their own) — adapted here for this project's data, not copied. This module reads
nothing analytical itself. `jobs/update_dashboard_summary.py` is the source of
precomputed figures for the QA pages (00-04) that read Data/dashboard_summary/*;
pages 05-16 compute live at render time instead (see their own module docstrings)
and don't depend on this pipeline.

Color roles below come from the project's validated dataviz palette
(categorical order, status colors, sequential blue ramp) rather than Plotly's
default qualitative cycle.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
SUMMARY_DIR = DATA_DIR / "dashboard_summary"

# Categorical slots, fixed order (never cycled/reassigned per filter).
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

# Status colors — reserved, never reused as a categorical series color.
STATUS_COLORS = {
    "OK": "#0ca30c",
    "PARTIAL": "#fab219",
    "STALE": "#fab219",
    "FAILED": "#d03b3b",
    "NO_DATA_AVAILABLE": "#d03b3b",
    "NOT_YET_PUBLISHED": "#fab219",
    "UNKNOWN": "#898781",
}

# Diverging pair (polarity: contango vs. backwardation) — blue/red poles.
DIVERGING = {"Contango": "#2a78d6", "Backwardation": "#e34948", "N/A (single contract)": "#898781"}

# Sequential blue ramp, light->dark (magnitude — coverage gaps, vol heatmaps).
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"*{subtitle}*")


def render_key_takeaways(bullets: list[str]) -> None:
    with st.container(border=True):
        st.markdown("**Key Takeaways**")
        for bullet in bullets:
            st.markdown(f"- {bullet}")


# Maps our hex palette to Streamlit's fixed `:color[text]` markdown keywords
# (blue/green/orange/red/violet/gray/rainbow — not arbitrary hex).
_STREAMLIT_COLOR_KEYWORDS = {
    "#0ca30c": "green", "#fab219": "orange", "#d03b3b": "red", "#898781": "gray",
    "#2a78d6": "blue", "#e34948": "red",
}


def colored_badge(text: str, hex_color: str) -> str:
    keyword = _STREAMLIT_COLOR_KEYWORDS.get(hex_color, "gray")
    return f":{keyword}[**{text}**]"


def status_badge(status: str) -> str:
    return colored_badge(status, STATUS_COLORS.get(status, "#898781"))


def require_summary_files(*names: str) -> bool:
    """Guard pattern: check required dashboard_summary/ files exist before
    rendering. Returns True if all present; otherwise shows a friendly message
    (not a stack trace) and stops the page.
    """
    missing = [n for n in names if not (SUMMARY_DIR / n).exists()]
    if missing:
        st.info(
            f"Required summary file(s) not found: {missing}. "
            "Run `python jobs/update_dashboard_summary.py` to generate them "
            "(or wait for the next CTA_DashboardSummary scheduled run, 6:25PM)."
        )
        st.stop()
    return True


@st.cache_data(ttl=1200)
def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(SUMMARY_DIR / name, **kwargs)


@st.cache_data(ttl=1200)
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(SUMMARY_DIR / name)


def price_richness(series: pd.Series, min_obs: int = 20) -> dict:
    """Where the most recent value sits in its own trailing history - percentile rank
    and z-score. Purely descriptive (this dashboard is a QA/monitoring tool, not a
    signal-analysis one, per CLAUDE.md) - not a trade signal, just "is today high or
    low relative to what this instrument has itself printed." Returns None values if
    there isn't enough history to make that a meaningful statement.
    """
    series = series.dropna()
    if len(series) < min_obs:
        return {"percentile": None, "zscore": None, "n_obs": len(series)}
    latest = series.iloc[-1]
    percentile = (series <= latest).mean() * 100
    std = series.std()
    zscore = (latest - series.mean()) / std if std > 0 else None
    return {"percentile": percentile, "zscore": zscore, "n_obs": len(series)}


def blue_gradient_style(df: pd.DataFrame, subset=None):
    """A pandas Styler background gradient built from the project's sequential
    blue ramp, not matplotlib's default 'Blues' colormap, so table heatmaps
    stay on-palette.

    `highlight_null` runs after `background_gradient` because pandas colors NaN
    cells solid black by default (verified directly against this pandas version,
    not assumed) - a real missing-data case (e.g. an asset with insufficient
    history in a given period) would otherwise render as a jarring black cell
    rather than reading as "no data."
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    return (
        df.style
        .background_gradient(cmap=cmap, subset=subset)
        .highlight_null(props="background-color: transparent; color: inherit;")
    )
