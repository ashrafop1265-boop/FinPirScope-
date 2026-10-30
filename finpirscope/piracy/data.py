"""Data cleaning + merging. A faithful port of sections 3-5 of the notebook."""
import numpy as np
import pandas as pd

from . import config as C


def runtime_to_minutes(x):
    if pd.isnull(x):
        return np.nan
    x = str(x).replace("min", "").strip()
    try:
        if "h" in x:
            hours, minutes = x.split("h")
            minutes = minutes.strip() or 0
            return int(hours.strip()) * 60 + int(minutes)
        return float(x)
    except Exception:
        return np.nan


def _clean_title(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace("[^a-z0-9 ]", "", regex=True).str.strip()


def clean_pirated(path=C.PIRATED_CSV) -> pd.DataFrame:
    p = pd.read_csv(path)
    p = p.drop(columns=["Unnamed: 0", "storyline"], errors="ignore")
    p = p.rename(columns={"IMDb-rating": "imdb_rating", "appropriate_for": "age_rating"})

    for col in ("downloads", "views"):
        p[col] = pd.to_numeric(p[col].astype(str).str.replace(",", ""), errors="coerce")
    p["imdb_rating"] = pd.to_numeric(p["imdb_rating"], errors="coerce")

    p["runtime"] = p["run_time"].apply(runtime_to_minutes)
    p = p.drop(columns=["run_time"])

    p["posted_date"] = pd.to_datetime(p["posted_date"], errors="coerce", format="mixed")
    p["release_date"] = pd.to_datetime(p["release_date"], errors="coerce", format="mixed")

    p = p.dropna(subset=["title", "downloads", "posted_date"])
    p = p.drop_duplicates(subset=["id"])

    for col in ("industry", "language", "age_rating"):
        p[col] = p[col].fillna("Unknown")
    p["language"] = p["language"].astype(str).str.split(",").str[0].str.strip()

    p.loc[p["runtime"] > 400, "runtime"] = np.nan
    p.loc[p["runtime"] < 20, "runtime"] = np.nan

    last_date = p["posted_date"].max()
    p["days_since_posted"] = (last_date - p["posted_date"]).dt.days
    p["leak_days"] = (p["posted_date"] - p["release_date"]).dt.days
    p["release_year"] = p["release_date"].dt.year
    p["release_month"] = p["release_date"].dt.month

    for col in ("industry", "language", "age_rating"):
        counts = p[col].value_counts()
        p.loc[p[col].isin(counts[counts < 50].index), col] = "Other"
    return p


def clean_tmdb(pirated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if C.TMDB_V11_CSV.exists():
        t2 = pd.read_csv(C.TMDB_V11_CSV, low_memory=False)
        t2 = t2[["id", "title", "release_date", "vote_average", "vote_count", "popularity",
                 "budget", "revenue", "genres", "status", "adult"]].copy()
        t2["release_date"] = pd.to_datetime(t2["release_date"], errors="coerce")
        t2 = t2.dropna(subset=["release_date", "title"])
        t2 = t2[t2["release_date"] >= pirated["release_date"].min()]
        t2 = t2[(t2["status"] == "Released") & (t2["adult"].astype(str) == "False")]
        t2 = t2[t2["vote_count"] >= 10]
        frames.append(t2.drop(columns=["status", "adult"]))

    t1 = pd.read_csv(C.ETHICAL_CSV)[["id", "title", "release_date", "vote_average", "vote_count",
                                      "popularity", "budget", "revenue", "genres"]].copy()
    t1["release_date"] = pd.to_datetime(t1["release_date"], errors="coerce")
    t1 = t1.dropna(subset=["release_date", "title"])
    frames.append(t1)

    tmdb = pd.concat(frames).drop_duplicates(subset=["id"])
    tmdb["budget"] = tmdb["budget"].replace(0, np.nan)
    tmdb["revenue"] = tmdb["revenue"].replace(0, np.nan)
    g = tmdb["genres"].astype(str).str.replace("Science Fiction", "SciFi").str.replace("TV Movie", "TVMovie")
    tmdb["genre"] = g.str.split(",").str[0].str.split(" ").str[0]
    tmdb.loc[tmdb["genre"] == "nan", "genre"] = "Unknown"
    return tmdb


def build_dataset() -> pd.DataFrame:
    pirated = clean_pirated()
    tmdb = clean_tmdb(pirated)

    pirated["title_clean"] = _clean_title(pirated["title"])
    tmdb["title_clean"] = _clean_title(tmdb["title"])
    tmdb["release_year"] = tmdb["release_date"].dt.year
    tmdb = tmdb[tmdb["title_clean"].isin(pirated["title_clean"])]
    tmdb = (tmdb.sort_values("vote_count", ascending=False)
                .drop_duplicates(subset=["title_clean", "release_year"]))
    tmdb = tmdb[["title_clean", "release_year", "vote_average", "vote_count", "popularity",
                 "budget", "revenue", "genre"]]
    tmdb.columns = ["title_clean", "release_year", "tmdb_vote_average", "tmdb_vote_count",
                    "tmdb_popularity", "tmdb_budget", "tmdb_revenue", "genre"]

    df = pd.merge(pirated, tmdb, on=["title_clean", "release_year"], how="left")
    df["has_tmdb"] = df["tmdb_popularity"].notnull().astype(int)
    df["genre"] = df["genre"].fillna("Unknown")
    counts = df["genre"].value_counts()
    df.loc[df["genre"].isin(counts[counts < 30].index), "genre"] = "Other"
    return df


def make_features(raw: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Raw columns -> model matrix. Used for BOTH training and live prediction so the two
    can never drift apart. `spec` holds numeric/log cols and (after fitting) medians,
    category lists and final column order."""
    X = raw[spec["numeric"] + C.CATEGORICAL].copy()
    for col in spec["numeric"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in spec["log"]:
        X[col] = np.log1p(X[col].clip(lower=0))
    if "medians" in spec:
        for col in spec["numeric"]:
            X[col] = X[col].fillna(spec["medians"][col])
    if "categories" in spec:
        for col in C.CATEGORICAL:
            known = spec["categories"][col]
            fallback = "Other" if "Other" in known else known[0]
            X[col] = X[col].where(X[col].isin(known), fallback)
    X = pd.get_dummies(X, columns=C.CATEGORICAL, dtype=int)
    if "columns" in spec:
        X = X.reindex(columns=spec["columns"], fill_value=0)
    return X
