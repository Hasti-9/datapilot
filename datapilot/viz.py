"""Picks reasonable Plotly charts for a dataframe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .eda import column_kind

SURFACE = "#FCFCFB"
INK = "#0B0B0B"
INK_2 = "#52514E"
INK_3 = "#8A8984"
GRID = "#ECECE9"

# categorical slots in fixed order; slot 1 doubles as the single-series color
SERIES = ["#2A78D6", "#EB6834", "#1BAF7A"]
ACCENT = SERIES[0]
SEQUENTIAL = ["#CDE2FB", "#9EC5F4", "#6DA7EC", "#3987E5", "#256ABF", "#184F95", "#0D366B"]
DIVERGING = [[0.0, "#B83232"], [0.25, "#E38A84"], [0.5, "#F0EFEC"], [0.75, "#6DA7EC"], [1.0, "#1C5CAB"]]


def style(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    is_hist = any(t.type == "histogram" for t in fig.data)
    fig.update_layout(
        template="none",
        height=height,
        margin=dict(l=8, r=8, t=44 if title else 12, b=8),
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=13, color=INK_2, family="Inter, sans-serif", weight=500)),
        font=dict(family="Inter, sans-serif", size=12, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.06 if is_hist else 0.35,
        barcornerradius=2 if is_hist else 4,
        showlegend=len(fig.data) > 1,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right",
                    title_text="", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E4E4E0", font=dict(color=INK, size=12,
                        family="Inter, sans-serif")),
        coloraxis_colorbar=dict(thickness=8, outlinewidth=0, tickfont=dict(size=10)),
    )
    axis = dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, ticks="",
                tickfont=dict(size=11, color=INK_3), title_font=dict(size=11, color=INK_3),
                automargin=True)
    fig.update_xaxes(**axis, showgrid=False)
    fig.update_yaxes(**axis, showgrid=True)
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line=dict(width=2))
    fig.update_traces(selector=dict(type="scatter", mode="markers"),
                      marker=dict(size=8, line=dict(width=1.5, color=SURFACE)))
    return fig


def hbar(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    fig = px.bar(df, x=x, y=y, orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_yaxes(categoryorder="total ascending", showgrid=False)
    fig.update_xaxes(showgrid=True)
    fig = style(fig, title)
    # keep bars around 20px thick instead of stretching to fill the band
    plot_h = 340 - 60
    fig.update_layout(bargap=min(0.8, max(0.35, 1 - 20 * len(df) / plot_h)))
    return fig


def heatmap(z, x, y, title: str, diverging: bool = False, **kw) -> go.Figure:
    scale = DIVERGING if diverging else SEQUENTIAL
    fig = px.imshow(z, x=x, y=y, color_continuous_scale=scale, aspect="auto", **kw)
    fig.update_traces(xgap=2, ygap=2)
    fig = style(fig, title, height=380)
    fig.update_xaxes(showgrid=False, linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(showgrid=False, linecolor="rgba(0,0,0,0)")

    # label each cell, switching to white text only on the dark end of the scale
    arr = np.asarray(z, dtype=float)
    lo, hi = kw.get("zmin", np.nanmin(arr)), kw.get("zmax", np.nanmax(arr))
    xs, ys = list(fig.data[0].x), list(fig.data[0].y)
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            v = arr[i, j]
            t = (v - lo) / (hi - lo) if hi > lo else 0
            dark = abs(t - 0.5) > 0.3 if diverging else t > 0.55
            fig.add_annotation(x=xv, y=yv, text=f"{v:g}", showarrow=False,
                               font=dict(size=11, color="#FFFFFF" if dark else INK))
    return fig


def auto_charts(df: pd.DataFrame, eda: dict, target: str | None = None, limit: int = 8) -> list[tuple[str, go.Figure]]:
    charts: list[tuple[str, go.Figure]] = []
    kinds = {c: column_kind(df[c]) for c in df.columns}
    numeric = [c for c, k in kinds.items() if k == "numeric" and df[c].nunique() > 1]
    categorical = [c for c, k in kinds.items() if k in ("categorical", "boolean") and 1 < df[c].nunique() <= 30]
    dates = [c for c, k in kinds.items() if k == "datetime"]

    for c in numeric[:3]:
        fig = px.histogram(df, x=c, nbins=36, color_discrete_sequence=[ACCENT])
        fig.update_yaxes(title_text="")
        charts.append((f"Distribution of {c}", style(fig, f"Distribution · {c}")))

    for c in categorical[:2]:
        vc = df[c].astype("string").value_counts().head(12).reset_index()
        vc.columns = [c, "count"]
        fig = hbar(vc, "count", c, f"Top values · {c}")
        fig.update_yaxes(title_text="")
        charts.append((f"Top values of {c}", fig))

    corr = eda.get("correlation_matrix")
    if corr is not None and not corr.empty:
        fig = heatmap(corr.round(2), corr.columns, corr.index, "Correlation", diverging=True,
                      zmin=-1, zmax=1)
        charts.append(("Correlation matrix", fig))

    for p in eda.get("correlated_pairs", [])[:1]:
        sample = df.sample(min(len(df), 1500), random_state=0)
        fig = px.scatter(sample, x=p["a"], y=p["b"], opacity=0.55, color_discrete_sequence=[ACCENT])
        fig = style(fig, f"{p['a']} vs {p['b']} · r = {p['r']}")
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        charts.append((f"{p['a']} vs {p['b']}", fig))

    if dates and numeric:
        d, v = dates[0], (target if target in numeric else numeric[0])
        ts = df[[d, v]].dropna().set_index(d).sort_index()
        freq = "D" if (ts.index.max() - ts.index.min()).days < 120 else "MS"
        ts = ts.resample(freq).mean().reset_index()
        fig = px.line(ts, x=d, y=v, color_discrete_sequence=[ACCENT])
        fig.update_xaxes(title_text="")
        charts.append((f"{v} over time", style(fig, f"Average {v} over time")))

    if target and target in df.columns:
        if kinds[target] == "numeric" and categorical:
            cat = next((c for c in categorical if c != target), None)
            if cat:
                fig = px.box(df, x=cat, y=target, color_discrete_sequence=[ACCENT])
                charts.append((f"{target} by {cat}", style(fig, f"{target} by {cat}")))
        elif kinds[target] in ("categorical", "boolean") and numeric:
            num = next((c for c in numeric if c != target), None)
            if num:
                fig = px.box(df, x=target, y=num, color=target, color_discrete_sequence=SERIES)
                fig.update_traces(marker=dict(size=4))
                charts.append((f"{num} by {target}", style(fig, f"{num} by {target}")))

    return charts[:limit]


def chart_for_result(df: pd.DataFrame) -> go.Figure | None:
    """Pick a sensible chart for a SQL result set."""
    if df is None or df.empty or df.shape[1] < 2 or len(df) > 500:
        return None
    x, y = df.columns[0], df.columns[1]
    if not pd.api.types.is_numeric_dtype(df[y]):
        return None
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        return style(px.line(df, x=x, y=y, color_discrete_sequence=[ACCENT]))
    if len(df) <= 30:
        return hbar(df.astype({x: "string"}), y, x)
    if pd.api.types.is_numeric_dtype(df[x]):
        return style(px.scatter(df, x=x, y=y, color_discrete_sequence=[ACCENT]))
    return None
