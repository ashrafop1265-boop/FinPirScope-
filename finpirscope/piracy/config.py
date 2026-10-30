from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

PIRATED_CSV = DATA_DIR / "movies_pirated.csv"
ETHICAL_CSV = DATA_DIR / "movies_ethical.csv"
# Optional large TMDB dump (~1M movies). If present it is merged in, exactly like the notebook.
TMDB_V11_CSV = DATA_DIR / "TMDB_movie_dataset_v11.csv"
CLEAN_CSV = DATA_DIR / "cleaned_piracy_data.csv"

NUMERIC_Q1 = ["imdb_rating", "runtime", "release_year", "release_month", "days_since_posted",
              "has_tmdb", "tmdb_vote_average", "tmdb_vote_count", "tmdb_popularity", "tmdb_budget"]
LOG_Q1 = ["days_since_posted", "tmdb_vote_count", "tmdb_popularity", "tmdb_budget"]

NUMERIC_Q2 = ["imdb_rating", "runtime", "release_month", "has_tmdb", "tmdb_vote_average",
              "tmdb_vote_count", "tmdb_popularity", "tmdb_budget"]
LOG_Q2 = ["tmdb_vote_count", "tmdb_popularity", "tmdb_budget"]

CATEGORICAL = ["industry", "language", "age_rating", "genre"]
RANDOM_STATE = 42
