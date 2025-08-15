# Trader Dashboard — Pro v6.6.2
Critical fix: charts no longer drop all rows on short timeframes (no more empty charts). Added hover crosshair, clearer captions ("higher/lower is better"), improved watchlist & heatmap, and a nicer EPS chart (with 4Q MA).

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_trader_pro_v662.py
```