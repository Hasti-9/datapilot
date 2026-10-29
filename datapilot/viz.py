"""Picks reasonable Plotly charts for a dataframe."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .eda import column_kind

TEMPLATE = "simple_white"
ACCENT = "#0F766E"


def style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=360, margin=dict(l=40, r=20, t=50, b=40),
                      title=dict(text=title, font=dict(size=14)),
                      font=dict(family="IBM Plex Sans, sans-serif", color="#1F2421"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def auto_charts(df: pd.DataFrame, eda: dict, target: str | None = None, limit: int = 10) -> list[tuple[str, go.Figure]]:
    charts: list[tuple[str, go.Figure]] = []
    kinds = {c: column_kind(df[c]) for c in df.columns}
    numeric = [c for c, k in kinds.items() if k == "numeric" and df[c].nunique() > 1]
    categorical = [c for c, k in kinds.items() if k in ("categorical", "boolean") and 1 < df[c].nunique() <= 30]
    dates = [c for c, k in kinds.items() if k == "datetime"]

    for c in numeric[:4]:
        fig = px.histogram(df, x=c, nbins=40, color_discrete_sequence=[ACCENT], marginal="box")
        charts.append((f"Distribution of {c}", style(fig, f"Distribution of {c}")))

    for c in categorical[:3]:
        vc = df[c].astype("string").value_counts().head(15).reset_index()
        vc.columns = [c, "count"]
        fig = px.bar(vc, x="count", y=c, orientation="h", color_discrete_sequence=[ACCENT])
        fig.update_yaxes(categoryorder="total ascending")
        charts.append((f"Top values of {c}", style(fig, f"Top values of {c}")))

    corr = eda.get("correlation_matrix")
    if corr is not None and not corr.empty:
        fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="BrBG", zmin=-1, zmax=1)
        charts.append(("Correlation matrix", style(fig, "Correlation matrix")))

    for p in eda.get("correlated_pairs", [])[:2]:
        sample = df.sample(min(len(df), 3000), random_state=0)
        fig = px.scatter(sample, x=p["a"], y=p["b"], opacity=0.6, trendline=None,
                         color_discrete_sequence=[ACCENT])
        charts.append((f"{p['a']} vs {p['b']}", style(fig, f"{p['a']} vs {p['b']} (r={p['r']})")))

    if dates and numeric:
        d, v = dates[0], (target if target in numeric else numeric[0])
        ts = df[[d, v]].dropna().set_index(d).sort_index()
        freq = "D" if (ts.index.max() - ts.index.min()).days < 120 else "MS"
        ts = ts.resample(freq).mean().reset_index()
        fig = px.line(ts, x=d, y=v, color_discrete_sequence=[ACCENT])
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
                fig = px.box(df, x=target, y=num, color=target,
                             color_discrete_sequence=[ACCENT, "#C2410C", "#6B7280"])
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
        return style(px.line(df, x=x, y=y, color_discrete_sequence=[ACCENT]), f"{y} by {x}")
    if len(df) <= 50:
        return style(px.bar(df, x=x, y=y, color_discrete_sequence=[ACCENT]), f"{y} by {x}")
    if pd.api.types.is_numeric_dtype(df[x]):
        return style(px.scatter(df, x=x, y=y, color_discrete_sequence=[ACCENT]), f"{y} vs {x}")
    return None
