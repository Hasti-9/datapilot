"""Train, compare and evaluate several models on a chosen target column."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .eda import column_kind, looks_like_id


@dataclass
class MLResult:
    task: str
    target: str
    features: list[str]
    leaderboard: pd.DataFrame
    best_model_name: str
    best_pipeline: Any
    feature_importance: pd.DataFrame
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lb = self.leaderboard.to_string(index=False)
        top = ", ".join(self.feature_importance.head(5)["feature"].tolist())
        return (f"Task: {self.task} on `{self.target}` using {len(self.features)} features.\n"
                f"Best model: {self.best_model_name}.\nTop features: {top}.\n\nLeaderboard:\n{lb}")


def detect_task(y: pd.Series) -> str:
    kind = column_kind(y)
    if kind == "numeric" and y.nunique() > 15:
        return "regression"
    return "classification"


def _select_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    num, cat = [], []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        kind = column_kind(s)
        # Skip identifiers, which only add noise / leakage.
        if looks_like_id(s):
            continue
        if kind == "numeric":
            num.append(c)
        elif kind in ("categorical", "boolean") and s.nunique() <= 50:
            cat.append(c)
        elif kind == "datetime":
            continue
    return num, cat


def _prepare(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype("string").astype(object).where(X[c].notna(), "missing")
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def _preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), cat),
    ])


def candidate_models(task: str) -> dict[str, Any]:
    if task == "classification":
        return {
            "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
    return {
        "Baseline (mean)": DummyRegressor(),
        "Ridge Regression": Ridge(alpha=1.0),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=15),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def train_and_compare(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    models: list[str] | None = None,
    max_rows: int = 50_000,
) -> MLResult:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    data = df.dropna(subset=[target])
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
    y = data[target]
    task = detect_task(y)
    if task == "classification":
        y = y.astype("string").astype(object)
        if y.nunique() < 2:
            raise ValueError("Target has a single class; nothing to learn.")
    else:
        y = y.astype(float)

    num, cat = _select_features(data, target)
    if not num and not cat:
        raise ValueError("No usable feature columns were found.")
    X = _prepare(data, num, cat)

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    scoring = "f1_weighted" if task == "classification" else "r2"
    folds = min(cv_folds, int(y_tr.value_counts().min())) if task == "classification" else cv_folds
    folds = max(folds, 2)

    rows, fitted = [], {}
    for name, est in candidate_models(task).items():
        if models and name not in models:
            continue
        pipe = Pipeline([("prep", _preprocessor(num, cat)), ("model", est)])
        t0 = time.perf_counter()
        cv = cross_val_score(pipe, X_tr, y_tr, cv=folds, scoring=scoring, n_jobs=None)
        pipe.fit(X_tr, y_tr)
        secs = time.perf_counter() - t0
        pred = pipe.predict(X_te)
        row: dict[str, Any] = {"model": name, f"cv_{scoring}": cv.mean(), "cv_std": cv.std()}
        if task == "classification":
            row["test_accuracy"] = accuracy_score(y_te, pred)
            row["test_f1"] = f1_score(y_te, pred, average="weighted", zero_division=0)
            if hasattr(pipe, "predict_proba") and y.nunique() == 2:
                try:
                    proba = pipe.predict_proba(X_te)[:, 1]
                    pos = pipe.classes_[1]
                    row["test_roc_auc"] = roc_auc_score((y_te == pos).astype(int), proba)
                except Exception:
                    row["test_roc_auc"] = np.nan
        else:
            row["test_r2"] = r2_score(y_te, pred)
            row["test_rmse"] = root_mean_squared_error(y_te, pred)
            row["test_mae"] = mean_absolute_error(y_te, pred)
        row["train_seconds"] = secs
        rows.append(row)
        fitted[name] = pipe

    sort_key = "test_f1" if task == "classification" else "test_r2"
    lb = pd.DataFrame(rows).sort_values(sort_key, ascending=False).reset_index(drop=True)
    lb = lb.round(4)
    best_name = lb.iloc[0]["model"]
    best = fitted[best_name]

    imp = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=42,
                                 scoring="f1_weighted" if task == "classification" else "r2")
    fi = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
          .sort_values("importance", ascending=False).reset_index(drop=True))

    details: dict[str, Any] = {"n_train": len(X_tr), "n_test": len(X_te), "cv_folds": folds,
                               "numeric_features": num, "categorical_features": cat}
    pred = best.predict(X_te)
    if task == "classification":
        labels = sorted(pd.unique(y))
        details["labels"] = labels
        details["confusion_matrix"] = confusion_matrix(y_te, pred, labels=labels).tolist()
        details["class_balance"] = y.value_counts(normalize=True).round(3).to_dict()
    else:
        details["y_test"] = np.asarray(y_te).tolist()
        details["y_pred"] = np.asarray(pred).tolist()

    return MLResult(task, target, list(X.columns), lb, best_name, best, fi, details)


def suggest_target(df: pd.DataFrame) -> str | None:
    """Guess a sensible prediction target by name, then fall back to the last column."""
    hints = ("target", "label", "churn", "outcome", "class", "price", "sales", "revenue",
             "survived", "default", "fraud", "y")
    for c in df.columns:
        if any(c == h or c.endswith("_" + h) or c.startswith(h + "_") for h in hints):
            return c
    return df.columns[-1] if len(df.columns) else None
