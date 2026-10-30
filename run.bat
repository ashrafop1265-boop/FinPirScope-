@echo off
cd /d "%~dp0finpirscope"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
if not exist models\q1.joblib python -m piracy.train
python app.py
pause
