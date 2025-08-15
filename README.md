# Trader Dashboard — Pro v6.6.6
- **True dark/light/auto** theming for all **Altair charts** (transparent backgrounds, themed axes/legends).
- **Price chart** now colored for dark mode; SMAs have distinct colors.
- **Enhanced Heatmap**: multi-timeframe (1D/1W/1M/3M/YTD), optional sector grouping, optional volume bubbles, symmetric color domain.
- PDF export remains fixed.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_trader_pro_v666.py
```