"""FinPirScope - Flask web app.

    python -m piracy.train     # once, to train and save models
    python app.py              # then open http://127.0.0.1:5000
"""
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from piracy import config as C
from piracy.data import make_features

app = Flask(__name__)

_cache = {}


def load():
    if not _cache:
        if not (C.MODEL_DIR / "q1.joblib").exists():
            raise RuntimeError("Models not found. Run:  python -m piracy.train")
        _cache["q1"] = joblib.load(C.MODEL_DIR / "q1.joblib")
        _cache["q2"] = joblib.load(C.MODEL_DIR / "q2.joblib")
        _cache["metrics"] = json.loads((C.MODEL_DIR / "metrics.json").read_text())
        df = pd.read_csv(C.CLEAN_CSV, usecols=[
            "title", "release_year", "release_month", "imdb_rating", "runtime", "industry", "language",
            "age_rating", "genre", "downloads", "leak_days", "days_since_posted", "has_tmdb",
            "tmdb_vote_average", "tmdb_vote_count", "tmdb_popularity", "tmdb_budget"])
        _cache["movies"] = df
        _cache["titles_lower"] = df["title"].astype(str).str.lower()
    return _cache


def nan_to_none(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def num(d, key, default=np.nan):
    v = d.get(key)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")


def run_models(bundle, raw):
    X = make_features(raw, bundle["spec"])
    Xs = bundle["scaler"].transform(X)
    out = []
    for name, (model, scaled) in bundle["models"].items():
        data = Xs if scaled else X
        pred = int(model.predict(data)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(data)[0][1])  # P(class 1)
        out.append({"model": name, "label": bundle["labels"][pred], "class": pred, "prob_positive": prob})
    votes = sum(m["class"] for m in out)
    probs = [m["prob_positive"] for m in out if m["prob_positive"] is not None]
    return {"models": out,
            "consensus": bundle["labels"][1 if votes * 2 > len(out) else 0],
            "agree": f"{max(votes, len(out) - votes)}/{len(out)}",
            "avg_prob_positive": float(np.mean(probs)) if probs else None}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/metrics")
def api_metrics():
    return jsonify(load()["metrics"])


@app.get("/api/search")
def api_search():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])
    c = load()
    hits = c["movies"][c["titles_lower"].str.contains(q, regex=False)].copy()
    hits["_starts"] = ~c["titles_lower"][hits.index].str.startswith(q)
    hits = hits.sort_values(["_starts", "downloads"], ascending=[True, False]).head(8).drop(columns="_starts")
    return jsonify([{k: nan_to_none(v) for k, v in r.items()} for r in hits.to_dict("records")])


@app.post("/api/predict")
def api_predict():
    c = load()
    d = request.get_json(force=True, silent=True) or {}
    try:
        tmdb_fields = ["tmdb_vote_average", "tmdb_vote_count", "tmdb_popularity", "tmdb_budget"]
        has_tmdb = int(any(d.get(f) not in (None, "") for f in tmdb_fields))
        raw = pd.DataFrame([{
            "imdb_rating": num(d, "imdb_rating"),
            "runtime": num(d, "runtime"),
            "release_year": num(d, "release_year"),
            "release_month": num(d, "release_month"),
            "days_since_posted": num(d, "days_since_posted"),
            "has_tmdb": has_tmdb,
            **{f: num(d, f) for f in tmdb_fields},
            **{col: d.get(col) or "Other" for col in C.CATEGORICAL},
        }])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    q1 = run_models(c["q1"], raw)
    q2 = run_models(c["q2"], raw)
    q2["median_leak"] = c["q2"]["median_leak"]
    return jsonify({"q1": q1, "q2": q2})


if __name__ == "__main__":
    load()
    app.run(debug=False, port=5000)
