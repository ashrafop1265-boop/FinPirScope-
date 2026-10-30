# FinPirScope: Movie Piracy Prediction (ML project + web app)

Turns the `finpirscope.ipynb` notebook into a complete project: a reusable data/ML pipeline,
trained models, and a Flask web front end.

**Questions answered**
1. **Will a movie be pirated heavily?** (top third vs bottom third of downloads): Decision Tree, ANN, SVM, XGBoost, KNN
2. **Will it be pirated quickly?** (uploaded faster vs slower than the median days after release): Decision Tree, Random Forest, KNN

## Quick start

```bash
./run.sh            # macOS / Linux   (run.bat on Windows)
```
Then open <http://127.0.0.1:5000>.

Manual steps:
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m piracy.train          # optional: models are already included; add --fast for smaller grids
python app.py
```

## Front end
| Tab | What it does |
|---|---|
| **Predict** | Search a movie to auto-fill (or type values) and get Q1 + Q2 predictions from every model, with a consensus and probabilities |
| **Model results** | Accuracy / F1 tables, bar charts, confusion matrices, feature importance, ANN loss curve, KNN K-curve |
| **Data insights** | Download distribution, leak-time distribution, industry and year trends, correlation heatmap |
| **About** | Method and caveats |

Charts are plain SVG, so the UI works offline with no CDN.

## Project layout
```
app.py                  Flask app + JSON API (/api/predict, /api/search, /api/metrics)
piracy/config.py        paths and feature lists
piracy/data.py          cleaning, TMDB merge, shared feature builder (used in training AND prediction)
piracy/train.py         trains all models, saves models/*.joblib and models/metrics.json
templates/, static/     front end (HTML, CSS, vanilla JS)
data/                   movies_pirated.csv, movies_ethical.csv (+ generated cleaned_piracy_data.csv)
notebooks/              original notebook
```

## Notes and differences from the notebook
* **TMDB_movie_dataset_v11.csv was not provided**, so the project is trained with `movies_pirated.csv` + `movies_ethical.csv` only
  (~10% of titles match TMDB, hence TMDB fields are optional in the form). To reproduce the notebook fully, drop
  `TMDB_movie_dataset_v11.csv` into `data/` and re-run `python -m piracy.train`; it is picked up automatically.
* **ANN** uses scikit-learn's `MLPClassifier` (same 128-64-32 layout, early stopping) instead of Keras, so no TensorFlow install is needed.
* **XGBoost**: if `xgboost` is installed it is used; otherwise scikit-learn's `HistGradientBoostingClassifier`
  is used and shown as "Gradient Boosting" in the UI. Uncomment `xgboost` in `requirements.txt` to switch.
* Genre parsing keeps "Science Fiction" / "TV Movie" together instead of cutting at the first word.
* Q1 depends strongly on *days on the site* (older uploads have had longer to accumulate downloads); Q2 is much harder (~60% vs 50% chance).
* Intended as an academic prediction exercise on historical data.
