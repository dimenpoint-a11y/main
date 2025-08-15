# Trader Dashboard — Pro v6.6.7
Fixes:
- Removed duplicate sparkline on Price & Indicators (now a single, primary chart).
- Volume bars now color **green/red vs previous close**.
- Heatmap facet sorting fixed (explicit sector order by average returns).

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_trader_pro_v667.py
```