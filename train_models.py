"""
Train and evaluate five classifiers on the UCI "Online Shoppers Purchasing
Intention" dataset and persist everything the Streamlit app needs.

Run:  python train_models.py

Outputs
-------
model/<model_key>.joblib   fitted sklearn Pipeline (preprocessing + estimator)
model/metrics.json         hold-out metrics + best hyper-parameters per model
model/feature_schema.json  expected input columns / dtypes for the app
test_data.csv              the untouched 20% hold-out split (features + target)
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "Revenue"
SEED = 2026
TEST_SIZE = 0.20

# Continuous browsing-behaviour features (heavily right-skewed -> log1p + scale)
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
# Integer-coded or string categories -> one-hot (codes carry no order)
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
BINARY_FEATURES = ["Weekend"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def to_bool_int(series: pd.Series) -> pd.Series:
    """Map TRUE/FALSE, True/False, 1/0, yes/no to 0/1 integers."""
    if series.dtype == bool:
        return series.astype(int)
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
    return series.astype(str).str.strip().str.lower().map(mapping).astype(int)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Weekend"] = to_bool_int(df["Weekend"])
    df[TARGET] = to_bool_int(df[TARGET])
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


# Each entry: display name, estimator, small hyper-parameter grid
MODEL_SPECS = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(max_iter=2000, solver="lbfgs"),
        {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0], "clf__class_weight": [None, "balanced"]},
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(random_state=SEED),
        {
            "clf__max_depth": [4, 6, 8, 12, None],
            "clf__min_samples_leaf": [1, 10, 30],
            "clf__criterion": ["gini", "entropy"],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "knn": (
        "KNN",
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [5, 11, 21, 31],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    ),
    "naive_bayes": (
        "Naive Bayes",
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]},
    ),
    "svm": (
        "Support Vector Machine",
        SVC(kernel="rbf", probability=True, random_state=SEED),
        {
            "clf__C": [0.5, 1.0, 5.0],
            "clf__gamma": ["scale", 0.01],
            "clf__class_weight": [None, "balanced"],
        },
    ),
}


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    # Labels come from P(purchase) >= threshold for every model, so the numbers match the
    # Streamlit app exactly. (SVC.predict() uses the raw decision function instead of the
    # Platt-scaled probability, which would give slightly different labels.)
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main() -> None:
    df = load_dataset()
    X, y = df[ALL_FEATURES], df[TARGET]
    print(f"Dataset after de-duplication: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Positive class (Revenue=1) share: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for key, (display_name, estimator, grid) in MODEL_SPECS.items():
        start = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        # F1 is used for model selection because the classes are imbalanced (~15% buyers)
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        metrics = evaluate(best, X_test, y_test)
        results[key] = {
            "display_name": display_name,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1": search.best_score_,
            "test_metrics": metrics,
        }
        joblib.dump(best, MODEL_DIR / f"{key}.joblib", compress=3)
        print(
            f"{display_name:<24} "
            + " ".join(f"{m}={v:.4f}" for m, v in metrics.items())
            + f"  ({time.time() - start:.1f}s) best={results[key]['best_params']}"
        )

    meta = {
        "dataset": "UCI Online Shoppers Purchasing Intention (id 468)",
        "n_rows_after_dedup": int(X.shape[0]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_positive_rate": float(y_test.mean()),
        "random_state": SEED,
        "models": results,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(meta, indent=2, default=str))

    schema = {
        "target": TARGET,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "binary": BINARY_FEATURES,
        "model_files": {key: f"{key}.joblib" for key in MODEL_SPECS},
        "display_names": {key: spec[0] for key, spec in MODEL_SPECS.items()},
    }
    (MODEL_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_CSV, index=False)
    print(f"\nSaved {len(MODEL_SPECS)} models to {MODEL_DIR} and hold-out set to {TEST_CSV}")

    table = pd.DataFrame(
        {v["display_name"]: v["test_metrics"] for v in results.values()}
    ).T.round(4)
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
