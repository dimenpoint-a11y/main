# Trader Dashboard — Pro v4 (Personal Use)

Adds clear **value‑meaning captions** under all charts (what's "higher better" vs "lower better").

## Files
- `app_trader_pro.py` — Streamlit app with explanatory captions under Price, RSI, MACD, ATR, Backtest, Heatmap, etc.
- `tests_indicators.py` — local unit-style tests
- `requirements.txt` — pinned versions for reproducible deploys

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_trader_pro.py
```

## Tests
```bash
python tests_indicators.py
```

Notes: Heuristics are for **personal, educational use only**. Not investment advice.