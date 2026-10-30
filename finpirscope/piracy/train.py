"""Train every model from the notebook and save artifacts + metrics for the web app.

    python -m piracy.train            # full grids (as in the notebook)
    python -m piracy.train --fast     # small grids, a few minutes faster
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .data import build_dataset, make_features

try:  # optional dependency
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_spec(raw_train, numeric, logc):
    spec = {"numeric": numeric, "log": logc}
    tmp = make_features(raw_train, spec)
    spec["medians"] = {c: float(tmp[c].median()) for c in numeric}
    spec["categories"] = {c: sorted(raw_train[c].dropna().unique().tolist()) for c in C.CATEGORICAL}
    spec["columns"] = make_features(raw_train, spec).columns.tolist()
    return spec


def evaluate(name, y_true, y_pred, extra=None):
    out = {"model": name,
           "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
           "f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
           "confusion": confusion_matrix(y_true, y_pred).tolist()}
    out.update(extra or {})
    log(f"  {name:<14} acc={out['accuracy']:.3f}  f1={out['f1']:.3f}")
    return out


def best_k(Xs, y, ks=(3, 5, 7, 11, 15, 21, 31)):
    scores = {k: cross_val_score(KNeighborsClassifier(n_neighbors=k), Xs, y, cv=5).mean() for k in ks}
    return max(scores, key=scores.get), {str(k): round(float(v), 4) for k, v in scores.items()}


def importance_dict(model, cols, top=15):
    s = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).head(top)
    return [{"feature": k, "importance": round(float(v), 4)} for k, v in s.items()]


def train_question(raw, y, numeric, logc, fast, labels, kind):
    Xr_tr, Xr_te, y_tr, y_te = train_test_split(raw, y, test_size=0.2, random_state=C.RANDOM_STATE, stratify=y)
    spec = make_spec(Xr_tr, numeric, logc)
    X_tr, X_te = make_features(Xr_tr, spec), make_features(Xr_te, spec)
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    log(f"  train {X_tr.shape}, test {X_te.shape}")

    models, results = {}, []

    # ---- Decision tree
    grid = {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [1, 5, 10]}
    if fast: grid = {"max_depth": [6, 10], "min_samples_leaf": [5, 10]}
    g = GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=3, scoring="accuracy").fit(X_tr, y_tr)
    models["Decision Tree"] = (g.best_estimator_, False)
    results.append(evaluate("Decision Tree", y_te, g.predict(X_te), {"params": g.best_params_}))
    imp_model = g.best_estimator_

    # ---- KNN
    k, cv = best_k(Xs_tr, y_tr)
    knn = KNeighborsClassifier(n_neighbors=k).fit(Xs_tr, y_tr)
    models["KNN"] = (knn, True)
    results.append(evaluate("KNN", y_te, knn.predict(Xs_te), {"params": {"k": k}, "k_curve": cv}))

    if kind == "q1":
        # ---- ANN (scikit-learn MLP; same 128-64-32 architecture as the Keras notebook)
        ann = MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3, learning_rate_init=5e-4,
                            batch_size=64, early_stopping=True, validation_fraction=0.2,
                            n_iter_no_change=10, max_iter=100, random_state=42).fit(Xs_tr, y_tr)
        models["ANN"] = (ann, True)
        results.append(evaluate("ANN", y_te, ann.predict(Xs_te), {
            "loss_curve": [round(float(v), 4) for v in ann.loss_curve_],
            "val_acc_curve": [round(float(v), 4) for v in ann.validation_scores_]}))

        # ---- SVM
        grid = {"C": [1, 10, 50], "gamma": ["scale", 0.01, 0.001]}
        if fast: grid = {"C": [1, 10], "gamma": ["scale", 0.01]}
        gs = GridSearchCV(SVC(kernel="rbf"), grid, cv=3, scoring="accuracy").fit(Xs_tr[:5000], y_tr[:5000])
        svm = SVC(kernel="rbf", **gs.best_params_).fit(Xs_tr, y_tr)
        models["SVM"] = (svm, True)
        results.append(evaluate("SVM", y_te, svm.predict(Xs_te), {"params": gs.best_params_}))

        # ---- XGBoost (or HistGradientBoosting if xgboost is not installed)
        if HAVE_XGB:
            grid = {"n_estimators": [300, 500], "max_depth": [4, 6, 8], "learning_rate": [0.03, 0.1]}
            if fast: grid = {"n_estimators": [300], "max_depth": [4, 6], "learning_rate": [0.1]}
            gx = GridSearchCV(XGBClassifier(random_state=42), grid, cv=3, scoring="accuracy").fit(X_tr, y_tr)
            name = "XGBoost"
        else:
            grid = {"max_iter": [300, 500], "max_depth": [4, 6, 8], "learning_rate": [0.03, 0.1]}
            if fast: grid = {"max_iter": [300], "max_depth": [4, 6], "learning_rate": [0.1]}
            gx = GridSearchCV(HistGradientBoostingClassifier(random_state=42), grid, cv=3, scoring="accuracy").fit(X_tr, y_tr)
            name = "Gradient Boosting"  # stand-in used when xgboost is not installed
        models[name] = (gx.best_estimator_, False)
        results.append(evaluate(name, y_te, gx.predict(X_te),
                                {"params": {k: v for k, v in gx.best_params_.items()},
                                 "engine": "xgboost" if HAVE_XGB else "sklearn HistGradientBoosting"}))
        if HAVE_XGB:
            imp_model = gx.best_estimator_
        else:  # HGB has no feature_importances_: use permutation importance on a test sample
            from sklearn.inspection import permutation_importance
            idx = np.random.RandomState(0).choice(len(X_te), min(1500, len(X_te)), replace=False)
            pi = permutation_importance(gx.best_estimator_, X_te.iloc[idx], y_te.iloc[idx], n_repeats=3, random_state=0)
            class _M: feature_importances_ = np.clip(pi.importances_mean, 0, None)
            imp_model = _M()
    else:
        # ---- Random forest
        grid = {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5]}
        if fast: grid = {"n_estimators": [200], "max_depth": [10, 20], "min_samples_leaf": [3, 5]}
        gr = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=3, scoring="accuracy").fit(X_tr, y_tr)
        models["Random Forest"] = (gr.best_estimator_, False)
        results.append(evaluate("Random Forest", y_te, gr.predict(X_te), {"params": gr.best_params_}))
        imp_model = gr.best_estimator_

    cols = spec["columns"]
    bundle = {"spec": spec, "scaler": scaler, "models": models, "labels": labels}
    return bundle, results, importance_dict(imp_model, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    fast = ap.parse_args().fast

    log("Building dataset ...")
    df = build_dataset()
    C.MODEL_DIR.mkdir(exist_ok=True)
    df.to_csv(C.CLEAN_CSV, index=False)
    log("dataset", df.shape, "| matched with TMDB:", f"{df['has_tmdb'].mean():.1%}")

    # ---------------- Question 1: High vs Low piracy
    log("Question 1: High vs Low piracy")
    df["piracy_tier"] = pd.qcut(df["downloads"].rank(method="first"), q=3, labels=[0, 1, 2]).astype(int)
    tiers = df.groupby("piracy_tier")["downloads"].agg(["count", "min", "max"]).to_dict("index")
    df1 = df[df["piracy_tier"] != 1].copy()
    y1 = (df1["piracy_tier"] == 2).astype(int)
    b1, r1, imp1 = train_question(df1, y1, C.NUMERIC_Q1, C.LOG_Q1, fast, ["Low", "High"], "q1")
    joblib.dump(b1, C.MODEL_DIR / "q1.joblib")

    # ---------------- Question 2: Fast vs Slow piracy
    log("Question 2: Fast vs Slow piracy")
    site_start = df["posted_date"].quantile(0.05)
    df2 = df[df["release_date"] >= site_start].copy()
    removed_neg = int((df2["leak_days"] < 0).sum())
    removed_old = int((df2["leak_days"] > 365).sum())
    df2 = df2[(df2["leak_days"] >= 0) & (df2["leak_days"] <= 365)]
    median_leak = float(df2["leak_days"].median())
    y2 = (df2["leak_days"] < median_leak).astype(int)
    b2, r2, imp2 = train_question(df2, y2, C.NUMERIC_Q2, C.LOG_Q2, fast, ["Slow", "Fast"], "q2")
    b2["median_leak"] = median_leak
    joblib.dump(b2, C.MODEL_DIR / "q2.joblib")

    # ---------------- EDA payload for the dashboard
    def hist(values, bins):
        h, e = np.histogram(values, bins=bins)
        return {"counts": h.tolist(), "edges": [round(float(x), 2) for x in e]}

    ind = df.groupby("industry")["downloads"].median().sort_values()
    ind2 = df2.groupby("industry")["leak_days"].median().sort_values()
    yr = df[df["release_year"].between(2000, 2023)].groupby("release_year")["downloads"].median()
    corr_cols = ["downloads", "views", "imdb_rating", "runtime", "days_since_posted", "release_year"]
    corr = df[corr_cols].corr(method="spearman").round(2)

    metrics = {
        "dataset": {"rows": int(len(df)), "tmdb_matched_pct": round(float(df["has_tmdb"].mean() * 100), 1),
                    "q1_rows": int(len(df1)), "q2_rows": int(len(df2)),
                    "tmdb_v11_used": C.TMDB_V11_CSV.exists(),
                    "date_min": str(df["posted_date"].quantile(0.05).date()),
                    "date_max": str(df["posted_date"].max().date())},
        "q1": {"results": r1, "importance": imp1,
               "tiers": {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in tiers.items()},
               "labels": ["Low", "High"]},
        "q2": {"results": r2, "importance": imp2, "median_leak": median_leak,
               "removed_before_release": removed_neg, "removed_over_year": removed_old,
               "labels": ["Slow", "Fast"]},
        "eda": {
            "downloads_log_hist": hist(np.log10(df["downloads"].clip(lower=1)), 30),
            "leak_hist": hist(df2["leak_days"], 36),
            "median_downloads_by_industry": {"labels": ind.index.tolist(), "values": ind.round(0).tolist()},
            "median_leak_by_industry": {"labels": ind2.index.tolist(), "values": ind2.round(0).tolist()},
            "median_downloads_by_year": {"labels": yr.index.astype(int).tolist(), "values": yr.round(0).tolist()},
            "correlation": {"labels": corr_cols, "matrix": corr.values.tolist()},
        },
        "options": {c: sorted(df[c].dropna().unique().tolist()) for c in C.CATEGORICAL},
    }
    (C.MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log("Done. Artifacts saved in", C.MODEL_DIR)


if __name__ == "__main__":
    main()
