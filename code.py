"""
Portfolio version of an Account Prioritization / BI scoring pipeline.

- No raw data shipped in this repo.
- File and column names are intentionally generic.
- This script is a template: it documents the end-to-end workflow (IO -> cleaning -> features -> model).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")          # repo contains no real files; user supplies locally
    random_state: int = 42
    test_size: float = 0.2

    # Generic filenames (user provides locally)
    firmographics_file: str = "account_firmographics.xlsx"
    intent_file: str = "intent_signals.xlsx"
    engagement_file: str = "engagement_dates.xlsx"
    target_file: str = "conversion_labels.xlsx"


CFG = Config()


# -----------------------------
# IO
# -----------------------------
def load_inputs(cfg: Config) -> dict[str, pd.DataFrame]:
    def _read_xlsx(name: str) -> pd.DataFrame:
        path = cfg.data_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input: {path}. This repo does not ship data. "
                f"Place your local files under /data with the expected names."
            )
        return pd.read_excel(path)

    return {
        "firmo": _read_xlsx(cfg.firmographics_file),
        "intent": _read_xlsx(cfg.intent_file),
        "engagement": _read_xlsx(cfg.engagement_file),
        "target": _read_xlsx(cfg.target_file),
    }


# -----------------------------
# Cleaning / standardization
# -----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase + snake_case for consistent public-facing code
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def prepare_master_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    firmo = standardize_columns(inputs["firmo"])
    intent = standardize_columns(inputs["intent"])
    engagement = standardize_columns(inputs["engagement"])
    target = standardize_columns(inputs["target"])

    # Required minimal keys (generic)
    # IMPORTANT: use a stable, non-sensitive surrogate key in public examples
    key = "account_key"

    for name, df in [("firmo", firmo), ("intent", intent), ("engagement", engagement), ("target", target)]:
        if key not in df.columns:
            raise ValueError(f"Missing required key '{key}' in {name} input.")

    # Example aggregations (generic, safe)
    # Intent: aggregate to account level
    if "signal_score" in intent.columns:
        intent_agg = intent.groupby(key, as_index=False).agg(
            intent_signal_mean=("signal_score", "mean"),
            intent_signal_max=("signal_score", "max"),
            intent_topic_count=("topic", "nunique") if "topic" in intent.columns else ("signal_score", "size"),
        )
    else:
        intent_agg = intent.groupby(key, as_index=False).size().rename(columns={"size": "intent_row_count"})

    # Engagement: compute simple recency features (requires date columns)
    eng = engagement.copy()
    date_cols = [c for c in eng.columns if c.endswith("_date")]
    for c in date_cols:
        eng[c] = pd.to_datetime(eng[c], errors="coerce")

    # Choose a fixed reference to avoid “now()” instability in portfolio outputs
    reference_date = pd.Timestamp("2025-01-01")
    for c in date_cols:
        eng[f"days_since_{c}"] = (reference_date - eng[c]).dt.days

    engagement_agg = eng.groupby(key, as_index=False).agg({c: "min" for c in eng.columns if c.startswith("days_since_")})

    # Target label
    if "converted" not in target.columns:
        raise ValueError("Target file must contain 'converted' column (0/1).")
    y = target[[key, "converted"]].drop_duplicates(subset=[key])

    # Merge master
    master = firmo.merge(intent_agg, on=key, how="left")
    master = master.merge(engagement_agg, on=key, how="left")
    master = master.merge(y, on=key, how="inner")

    return master


# -----------------------------
# Modeling
# -----------------------------
def train_model(master: pd.DataFrame, cfg: Config) -> None:
    # Drop identifiers
    y = master["converted"].astype(int)
    X = master.drop(columns=["converted"])

    # Separate feature types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=cfg.random_state,
        n_estimators=300,
    )

    pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])

    # Lightweight search (public-safe)
    param_dist = {
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 2, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=cfg.random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("Best params:", search.best_params_)
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("PR AUC:", average_precision_score(y_test, proba))
    print(classification_report(y_test, pred, digits=4))


def main() -> None:
    inputs = load_inputs(CFG)
    master = prepare_master_table(inputs)
    train_model(master, CFG)


if __name__ == "__main__":
    main()
