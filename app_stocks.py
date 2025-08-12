import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Analyzer (Starter)", layout="wide")
st.title("📈 AI Stock Analyzer (Starter)")

@st.cache_data(show_spinner=False)
def load_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    # Flatten MultiIndex columns if present (common on newer pandas/yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Standardize column names just in case ('close' -> 'Close')
    data.columns = [str(c).title() for c in data.columns]
    return data

def compute_indicators(df, windows=(20, 50, 200)):
    out = df.copy()
    if "Close" not in out.columns:
        return out
    for w in windows:
        out[f"Sma_{w}"] = out["Close"].rolling(w).mean()
    out["Returns"] = out["Close"].pct_change()
    out["Volatility_20d"] = out["Returns"].rolling(20).std() * np.sqrt(252)
    return out

with st.sidebar:
    st.markdown("**Controls**")
    ticker = st.text_input("Ticker", value="AAPL")
    start = st.date_input("Start date", pd.to_datetime("2023-01-01"))
    end = st.date_input("End date", pd.Timestamp.today().date())
    show_20 = st.checkbox("20-day SMA", True)
    show_50 = st.checkbox("50-day SMA", True)
    show_200 = st.checkbox("200-day SMA", True)

tab1, tab2, tab3 = st.tabs(["Price & Indicators", "Basic Signals", "News Sentiment (Manual)"])

with tab1:
    df = load_prices(ticker, start, end)
    if df.empty:
        st.warning("No price data. Try a different ticker or date range.")
    else:
        df = compute_indicators(df)
        st.subheader(f"Price & Moving Averages — {ticker}")
        if "Close" not in df:
            st.error("Downloaded data doesn't include a 'Close' column.")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
            if show_20 and "Sma_20" in df: ax.plot(df.index, df["Sma_20"], label="SMA 20")
            if show_50 and "Sma_50" in df: ax.plot(df.index, df["Sma_50"], label="SMA 50")
            if show_200 and "Sma_200" in df: ax.plot(df.index, df["Sma_200"], label="SMA 200")
            ax.set_ylabel("Price")
            ax.legend(loc="upper left")
            st.pyplot(fig)

            st.markdown("**Volatility (20d annualized) & Returns**")
            cols_to_plot = [c for c in ["Returns", "Volatility_20d"] if c in df.columns]
            if cols_to_plot:
                st.line_chart(df[cols_to_plot].dropna(how="all"))
            else:
                st.info("Insufficient data to compute returns/volatility for this date range.")

with tab2:
    st.subheader("Simple Momentum/Crossover Signals")
    if not df.empty and "Close" in df.columns:
        latest = df.dropna().iloc[-1]
        signals = []
        s20 = latest.get("Sma_20", np.nan)
        s50 = latest.get("Sma_50", np.nan)
        s200 = latest.get("Sma_200", np.nan)
        if s20 > s50 > s200:
            signals.append("Bullish MA alignment (20 > 50 > 200).")
        if s20 < s50 < s200:
            signals.append("Bearish MA alignment (20 < 50 < 200).")
        if latest.get("Returns", 0) > 0:
            signals.append("Positive latest daily return.")
        if latest.get("Volatility_20d", 0) < 0.25:
            signals.append("Lower recent volatility (20d) relative to 25% threshold.")
        if not signals:
            signals.append("No strong basic signals detected.")
        st.write("\n".join([f"- {s}" for s in signals]))
    else:
        st.info("Load price data first.")

with tab3:
    st.subheader("News Sentiment (Manual)")
    st.write("Paste a few recent headlines and press **Analyze**. (No API key required.)")
    txt = st.text_area("One headline per line", value="Apple tops earnings; services revenue hits record\niPhone sales soften as macro headwinds persist")
    if st.button("Analyze"):
        try:
            from transformers import pipeline
            clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if not lines:
                st.warning("Please paste at least one headline.")
            else:
                preds = clf(lines)
                out = pd.DataFrame({"headline": lines, "label": [p["label"] for p in preds], "score": [round(float(p["score"]),3) for p in preds]})
                st.dataframe(out)
                label_map = {"positive": 1, "negative": -1, "neutral": 0}
                out["sentiment_score"] = out["label"].map(lambda x: label_map.get(x.lower(), 0))
                st.metric("Avg Sentiment", f"{out['sentiment_score'].mean():.2f}")
        except Exception as e:
            st.error("Could not run sentiment model. Try again, or use a smaller model. Error: " + str(e))

st.caption("Educational starter app. Not financial advice.")