import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_FEATURES = ["year", "is_english"]
HISTORY_FEATURES = ["prev_books", "prev_avg_rating", "prev_best_rate", "prev_log_ratings"]
ALL_FEATURES = BASE_FEATURES + HISTORY_FEATURES

TRAIN_START, TRAIN_END = 1990, 2010  # train on 1990-2010, test on 2011 onward


def add_model_features(df):
    """Add the label and pre-publication features (author history uses only EARLIER years)."""
    df = df.copy()

    # Label: bestseller = top 20% by popularity within its decade (a proxy, not real sales)
    df["is_bestseller"] = (df["visibility_pct"] >= 80).astype(int)
    df["is_english"] = df["language_code"].fillna("").str.startswith("en").astype(int)
    df["log_ratings"] = np.log1p(df["ratings_count"])

    # One row per (author, year), then running totals over earlier years only
    per_year = (
        df.groupby(["main_author", "year"])
        .agg(
            n=("book_id", "size"),
            rating_sum=("average_rating", "sum"),
            best_sum=("is_bestseller", "sum"),
            log_sum=("log_ratings", "sum"),
        )
        .reset_index()
        .sort_values(["main_author", "year"])
    )
    for col in ["n", "rating_sum", "best_sum", "log_sum"]:
        # cumulative total minus this year's own value = total of strictly earlier years
        per_year["prev_" + col] = per_year.groupby("main_author")[col].cumsum() - per_year[col]

    n = per_year["prev_n"]
    has_history = n > 0
    safe_n = n.where(has_history, 1)  # avoids dividing by zero for debut authors
    per_year["prev_books"] = n
    per_year["prev_avg_rating"] = np.where(
        has_history, per_year["prev_rating_sum"] / safe_n, df["average_rating"].mean()
    )
    per_year["prev_best_rate"] = np.where(has_history, per_year["prev_best_sum"] / safe_n, 0.0)
    per_year["prev_log_ratings"] = np.where(has_history, per_year["prev_log_sum"] / safe_n, 0.0)

    keep = ["main_author", "year"] + HISTORY_FEATURES
    return df.merge(per_year[keep], on=["main_author", "year"], how="left")


def make_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, eval_metric="logloss", random_state=42,
        ),
    }


def train_and_evaluate(df):
    """Train 3 models on the earlier period, evaluate on the later period."""
    train = df[(df["year"] >= TRAIN_START) & (df["year"] <= TRAIN_END)]
    test = df[df["year"] > TRAIN_END]
    y_train, y_test = train["is_bestseller"], test["is_bestseller"]

    rows, xgb_model = [], None
    for name, model in make_models().items():
        model.fit(train[ALL_FEATURES], y_train)
        probs = model.predict_proba(test[ALL_FEATURES])[:, 1]
        rows.append({
            "Model": name,
            "AUC": roc_auc_score(y_test, probs),
            "Avg precision": average_precision_score(y_test, probs),
        })
        if name == "XGBoost":
            xgb_model = model

    # Ablation: does the author's earlier record help? (XGBoost with and without it)
    ablation = []
    for label, feats in [("Metadata only", BASE_FEATURES), ("Metadata + author history", ALL_FEATURES)]:
        m = make_models()["XGBoost"]
        m.fit(train[feats], y_train)
        ablation.append({
            "Feature set": label,
            "AUC": roc_auc_score(y_test, m.predict_proba(test[feats])[:, 1]),
        })

    return {
        "models": pd.DataFrame(rows),
        "ablation": pd.DataFrame(ablation),
        "xgb": xgb_model,
        "n_train": len(train),
        "n_test": len(test),
        "base_rate": float(y_test.mean()),
    }


def explain_row(model, row):
    """Contribution of each feature to one prediction (SHAP values, in log-odds)."""
    try:
        import shap

        values = np.asarray(shap.TreeExplainer(model).shap_values(row)).reshape(-1)
        if len(values) != row.shape[1]:
            raise ValueError("unexpected SHAP output shape")
    except Exception:
        # Fallback: XGBoost's built-in SHAP-style contributions
        from xgboost import DMatrix

        contribs = model.get_booster().predict(DMatrix(row), pred_contribs=True)
        values = contribs[0][:-1]  # last value is the baseline, not a feature
    return pd.Series(values, index=row.columns)