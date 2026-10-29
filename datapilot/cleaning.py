"""Automated data cleaning with an auditable log of every change."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    steps: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0

    def add(self, msg: str) -> None:
        self.steps.append(msg)

    def summary(self) -> str:
        head = (f"Rows {self.rows_before} -> {self.rows_after}, "
                f"columns {self.cols_before} -> {self.cols_after}.")
        return head + "\n" + "\n".join(f"- {s}" for s in self.steps)


def _snake(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return name or "col"


def _is_texty(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def clean_dataframe(
    df: pd.DataFrame,
    drop_missing_threshold: float = 0.6,
    clip_outliers: bool = False,
) -> tuple[pd.DataFrame, CleaningReport]:
    rep = CleaningReport(rows_before=len(df), cols_before=df.shape[1])
    df = df.copy()

    # Normalise column names (also required for clean SQL identifiers).
    new_cols, seen = [], {}
    for c in df.columns:
        base = _snake(c)
        n = seen.get(base, 0)
        seen[base] = n + 1
        new_cols.append(base if n == 0 else f"{base}_{n}")
    if list(df.columns) != new_cols:
        rep.add("Standardised column names to snake_case.")
    df.columns = new_cols

    # Trim whitespace and turn empty / placeholder strings into NaN.
    placeholders = {"", "na", "n/a", "null", "none", "nan", "-", "?"}
    for c in df.columns:
        if _is_texty(df[c]):
            s = df[c].astype("string").str.strip()
            mask = s.str.lower().isin(placeholders)
            if mask.any():
                rep.add(f"`{c}`: converted {int(mask.sum())} placeholder values to missing.")
            df[c] = s.mask(mask)

    # Drop fully empty rows/columns and exact duplicates.
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        rep.add(f"Dropped empty columns: {', '.join(empty_cols)}.")
    n = len(df)
    df = df.dropna(how="all")
    if len(df) < n:
        rep.add(f"Dropped {n - len(df)} fully empty rows.")
    n = len(df)
    df = df.drop_duplicates()
    if len(df) < n:
        rep.add(f"Removed {n - len(df)} duplicate rows.")

    # Infer types: numeric-looking text, booleans, dates.
    for c in df.columns:
        s = df[c]
        if not _is_texty(s):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        stripped = non_null.str.replace(r"[,$%\s]", "", regex=True)
        as_num = pd.to_numeric(stripped, errors="coerce")
        if as_num.notna().mean() >= 0.95:
            df[c] = pd.to_numeric(s.str.replace(r"[,$%\s]", "", regex=True), errors="coerce")
            rep.add(f"`{c}`: converted text to numeric.")
            continue
        lowered = set(non_null.str.lower().unique())
        if lowered <= {"true", "false", "yes", "no", "y", "n", "0", "1"} and len(lowered) <= 2:
            truthy = {"true", "yes", "y", "1"}
            df[c] = s.str.lower().map(lambda v: (v in truthy) if isinstance(v, str) else np.nan)
            df[c] = df[c].astype("boolean")
            rep.add(f"`{c}`: converted to boolean.")
            continue
        if re.search(r"date|time|_at$|_on$|day", c) or non_null.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.9 * s.notna().mean():
                df[c] = parsed
                rep.add(f"`{c}`: parsed as datetime.")
                continue
        # Low-cardinality text -> category for memory and modelling.
        if non_null.nunique() <= max(20, int(0.05 * len(non_null))):
            df[c] = s.astype("category")

    # Drop columns that are mostly missing.
    miss = df.isna().mean()
    sparse = miss[miss > drop_missing_threshold].index.tolist()
    if sparse:
        df = df.drop(columns=sparse)
        rep.add(f"Dropped columns with >{int(drop_missing_threshold * 100)}% missing: {', '.join(sparse)}.")

    # Impute remaining missing values.
    for c in df.columns:
        k = int(df[c].isna().sum())
        if k == 0:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            mode = s.mode(dropna=True)
            df[c] = s.fillna(mode.iloc[0] if not mode.empty else False)
            rep.add(f"`{c}`: filled {k} missing with mode.")
        elif pd.api.types.is_numeric_dtype(s):
            med = s.median()
            df[c] = s.fillna(med)
            rep.add(f"`{c}`: filled {k} missing with median ({med:.4g}).")
        elif pd.api.types.is_datetime64_any_dtype(s):
            df[c] = s.ffill().bfill()
            rep.add(f"`{c}`: forward/back-filled {k} missing dates.")
        else:
            if isinstance(s.dtype, pd.CategoricalDtype):
                if "Unknown" not in s.cat.categories:
                    s = s.cat.add_categories(["Unknown"])
            df[c] = s.fillna("Unknown")
            rep.add(f"`{c}`: filled {k} missing with 'Unknown'.")

    # Optional IQR outlier clipping on numeric columns.
    if clip_outliers:
        for c in df.select_dtypes(include="number").columns:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            k = int(((df[c] < lo) | (df[c] > hi)).sum())
            if k:
                df[c] = df[c].clip(lo, hi)
                rep.add(f"`{c}`: clipped {k} outliers to IQR bounds.")

    df = df.reset_index(drop=True)
    rep.rows_after, rep.cols_after = df.shape
    if not rep.steps:
        rep.add("No issues found; data was already clean.")
    return df, rep
