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


_SERIF_TITLE_CSS = """
<style>
h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 500;
}
</style>
"""


def page_header(title: str, subtitle: str) -> None:
    st.markdown(_SERIF_TITLE_CSS, unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"*{subtitle}*")


def apply_chart_theme(fig, y_unit: str = None):
    """Shared Plotly aesthetic applied on top of each page's own
    `update_layout` call (Plotly's update methods merge, not replace, so
    this only adds these specific properties without disturbing a chart's
    own title/height/margin choices already set elsewhere):

    - No vertical gridlines; a single faint horizontal zero-line instead of
      a full grid -- a cleaner look than Plotly's default grid.
    - Generous margins so charts aren't cramped against the page edge.
    - If `y_unit` is given, the y-axis unit label is mirrored on the right
      edge too, so a reader scanning either side of a wide chart can read
      the scale without eye travel back to one corner.
    """
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(137,135,129,0.15)", gridwidth=1,
        zeroline=True, zerolinecolor="rgba(137,135,129,0.35)", zerolinewidth=1,
    )
    fig.update_layout(margin=dict(t=40, b=40, l=60, r=60 if y_unit else 40))
    if y_unit:
        fig.update_layout(
            yaxis=dict(title=y_unit),
            yaxis2=dict(title=y_unit, overlaying="y", side="right", matches="y", showgrid=False),
        )
    return fig


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
