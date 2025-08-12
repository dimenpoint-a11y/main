# Personal Trader Dashboard — PLUS v2

Adds:
- **CSV export** of Watchlist Scanner results
- **Sector heatmap** that shows signal intensity by sector
- **Webhook hardening** (HTTPS-only validation, no redirects, JSON content-type, explicit timeout)
- **Diagnostics tab** and **scheduling tips** for daily checks

## Deploy
1) Create GitHub repo and upload `app_trader_plus_v2.py` + `requirements.txt`  
2) Streamlit Cloud → New app → `app_trader_plus_v2.py` → Deploy

## Local
```bash
pip install -r requirements.txt
streamlit run app_trader_plus_v2.py
```