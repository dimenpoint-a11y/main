# Trader Dashboard — Pro v6.4.4
- Fundamentals: **table only** (no charts)
- Insider: **no chart**, table + metrics (90d)
- Heatmap: **sector breadth** table (avg score/RSI, % above 200D, % new highs/lows, % MACD bull, % squeeze) + ranking
- Removed email field; simplified Legal & Disclosures

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_trader_pro_v644.py
```