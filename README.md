# Trader Dashboard — Pro v6.4.3
Fixes the FPDF download crash (uses `output(dest='S').encode('latin-1')`), removes duplicate helper, adds small UX clarifications.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_trader_pro_v643.py
```

## Streamlit Cloud
- Ensure `requirements.txt` matches this repo (no `streamlit-autorefresh`).
- Set entrypoint to `app_trader_pro_v643.py`.
- App menu → Clear cache if CSS looks stale.