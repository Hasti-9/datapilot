"""Exploratory data analysis: profiles, correlations, outliers and findings."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nun = s.nunique(dropna=True)
    if isinstance(s.dtype, pd.CategoricalDtype) or nun <= max(20, int(0.05 * len(s))):
        return "categorical"
    return "text"


def looks_like_id(s: pd.Series) -> bool:
    """Unique-per-row keys: text codes, or integers named like an id / forming a sequence."""
    n = len(s)
    if n <= 20 or s.nunique() < n:
        return False
    if re.search(r"(^|_)(id|key|uuid|code)$", str(s.name).lower()):
        return True
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = s.dropna()
        integral = bool((vals == vals.round()).all())
        return integral and (vals.max() - vals.min() + 1) == len(vals)
    return column_kind(s) == "text"


def profile(df: pd.DataFrame) -> dict[str, Any]:
    cols = []
    for c in df.columns:
        s = df[c]
        info: dict[str, Any] = {
            "name": c,
            "dtype": str(s.dtype),
            "kind": column_kind(s),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        if info["kind"] == "numeric":
            d = s.describe()
            info.update(
                mean=float(d["mean"]), std=float(d["std"]) if not np.isnan(d["std"]) else 0.0,
                min=float(d["min"]), median=float(s.median()), max=float(d["max"]),
                skew=float(s.skew()) if len(s) > 2 else 0.0,
            )
        elif info["kind"] in ("categorical", "boolean", "text"):
            top = s.astype("string").value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        elif info["kind"] == "datetime":
            info.update(min=str(s.min()), max=str(s.max()))
        cols.append(info)
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "columns": cols,
    }


def correlations(df: pd.DataFrame, threshold: float = 0.5) -> tuple[pd.DataFrame, list[dict]]:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return pd.DataFrame(), []
    corr = num.corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and abs(v) >= threshold:
                pairs.append({"a": cols[i], "b": cols[j], "r": round(float(v), 3)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return corr, pairs


def outliers(df: pd.DataFrame) -> dict[str, int]:
    out = {}
    for c in df.select_dtypes(include="number").columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        k = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        if k:
            out[c] = k
    return out


def key_findings(df: pd.DataFrame, prof: dict, corr_pairs: list[dict], outl: dict) -> list[str]:
    """Rule-based plain-language findings; the LLM later enriches these."""
    f = [f"The dataset has {prof['n_rows']:,} rows and {prof['n_cols']} columns."]
    kinds: dict[str, list[str]] = {}
    for c in prof["columns"]:
        kinds.setdefault(c["kind"], []).append(c["name"])
    f.append("Column types: " + "; ".join(f"{k}: {len(v)}" for k, v in kinds.items()) + ".")
    for c in prof["columns"]:
        if c["kind"] == "numeric" and abs(c.get("skew", 0)) > 1:
            side = "right" if c["skew"] > 0 else "left"
            f.append(f"`{c['name']}` is strongly {side}-skewed (skew {c['skew']:.2f}); "
                     f"median {c['median']:.4g} vs mean {c['mean']:.4g}.")
        if c["kind"] == "categorical" and c.get("top_values"):
            top, cnt = next(iter(c["top_values"].items()))
            share = cnt / max(prof["n_rows"], 1)
            if share > 0.5:
                f.append(f"`{c['name']}` is dominated by '{top}' ({share:.0%} of rows).")
        if looks_like_id(df[c["name"]]):
            f.append(f"`{c['name']}` is unique per row and is likely an identifier.")
    for p in corr_pairs[:5]:
        strength = "very strong" if abs(p["r"]) > 0.8 else "strong"
        direction = "positive" if p["r"] > 0 else "negative"
        f.append(f"{strength.capitalize()} {direction} correlation between `{p['a']}` and `{p['b']}` (r={p['r']}).")
    for c, k in sorted(outl.items(), key=lambda kv: -kv[1])[:3]:
        f.append(f"`{c}` has {k} outliers by the 1.5×IQR rule ({k / prof['n_rows']:.1%} of rows).")
    return f


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    prof = profile(df)
    corr, pairs = correlations(df)
    outl = outliers(df)
    return {
        "profile": prof,
        "correlation_matrix": corr,
        "correlated_pairs": pairs,
        "outliers": outl,
        "findings": key_findings(df, prof, pairs, outl),
    }
