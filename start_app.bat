@echo off
set PYTHONPATH=c:/Users/coura/Desktop/L400/final year project/run/Automated Fake News Detection;c:/Users/coura/Desktop/L400/final year project/run
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
