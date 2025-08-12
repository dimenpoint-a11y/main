# Trader Dashboard — Pro v6.4.2 (bundle)
- Aligned icon headers, JS auto-refresh (no extra deps), legal/policy exports, charts-first UX.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_trader_pro_v642.py
```
## Streamlit Cloud
- Ensure `requirements.txt` matches this repo (no `streamlit-autorefresh`).
- App entrypoint: `app_trader_pro_v642.py`.
- App menu → Clear cache if CSS looks stale.