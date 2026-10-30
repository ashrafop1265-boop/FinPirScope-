#!/usr/bin/env bash
# One-command start: creates a venv, installs deps, trains (first run only), launches the web app.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f models/q1.joblib ] || python -m piracy.train
python app.py
