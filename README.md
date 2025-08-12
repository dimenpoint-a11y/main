# AI Business MVP — Stocks, Real Estate & Consulting (Zero Cost)

Two Streamlit apps to deploy for free on **Streamlit Cloud** or **Hugging Face Spaces**.

## Apps
1. **app_stocks.py** — Price charts, moving averages, basic momentum signals, and manual news sentiment (FinBERT).
2. **app_realestate.py** — RandomForest estimator on synthetic or uploaded data.

## Quick Start
- Upload these files to a GitHub repo.
- On Streamlit Cloud, create a new app from the repo, pick the desired `.py` file, and deploy.

## Local Run
```bash
pip install -r requirements.txt
streamlit run app_stocks.py
# new terminal
streamlit run app_realestate.py
```