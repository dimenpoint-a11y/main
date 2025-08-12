import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from urllib.parse import urlparse
from datetime import datetime

st.set_page_config(page_title="Personal Trader Dashboard PLUS v2", layout="wide")
st.title("📊 Personal Trader Dashboard — PLUS v2 (For Your Use Only)")

# ------------------ Utilities ------------------
@st.cache_data(show_spinner=False)
def load_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).title() for c in data.columns]
    return data

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gains = (delta.clip(lower=0)).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gains / losses.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series, window=20, num_std=2.0):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return ma, upper, lower

def atr(df, period=14):
    h_l = df["High"] - df["Low"]
    h_pc = (df["High"] - df["Close"].shift()).abs()
    l_pc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_indicators(df):
    out = df.copy()
    if "Close" not in out.columns:
        return out
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["SMA_200"] = out["Close"].rolling(200).mean()
    out["Returns"] = out["Close"].pct_change()
    out["Volatility_20d"] = out["Returns"].rolling(20).std() * np.sqrt(252)
    out["RSI_14"] = rsi(out["Close"], 14)
    macd_line, signal_line, hist = macd(out["Close"], 12, 26, 9)
    out["MACD"] = macd_line
    out["MACD_signal"] = signal_line
    out["MACD_hist"] = hist
    bb_ma, bb_up, bb_lo = bollinger(out["Close"], 20, 2.0)
    out["BB_MA"] = bb_ma
    out["BB_UP"] = bb_up
    out["BB_LO"] = bb_lo
    out["ATR_14"] = atr(out, 14)
    out["Hi_20"] = out["Close"].rolling(20).max()
    out["Lo_20"] = out["Close"].rolling(20).min()
    out["Hi_252"] = out["Close"].rolling(252).max()
    out["Lo_252"] = out["Close"].rolling(252).min()
    out["BandWidth"] = (out["BB_UP"] - out["BB_LO"]) / out["BB_MA"]
    return out

def trade_score(row):
    score = 50.0
    if not np.isnan(row.get("RSI_14", np.nan)):
        if row["RSI_14"] < 30: score += 10
        elif row["RSI_14"] > 70: score -= 10
    if not np.isnan(row.get("MACD", np.nan)) and not np.isnan(row.get("MACD_signal", np.nan)):
        if row["MACD"] > row["MACD_signal"]: score += 10
        else: score -= 5
    if not np.isnan(row.get("SMA_200", np.nan)) and not np.isnan(row.get("Close", np.nan)):
        if row["Close"] > row["SMA_200"]: score += 10
        else: score -= 10
    if not np.isnan(row.get("BandWidth", np.nan)) and row["BandWidth"] is not None:
        if row["BandWidth"] < 0.05:  # squeeze
            score += 5
    return float(np.clip(score, 0, 100))

@st.cache_data(show_spinner=False)
def get_basic_info(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = {}
        try: info = t.info or {}
        except Exception: info = {}
        return {
            "sector": info.get("sector"),
            "industry": info.get("industry")
        }
    except Exception:
        return {"sector": None, "industry": None}

def is_https_url(url: str) -> bool:
    try:
        p = urlparse(url.strip())
        return p.scheme == "https" and bool(p.netloc)
    except Exception:
        return False

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    start = st.date_input("Start date", pd.to_datetime("2022-01-01"))
    end = st.date_input("End date", pd.Timestamp.today().date())
    st.markdown("---")
    st.subheader("Webhook (Optional)")
    st.caption("HTTPS only. Use IFTTT Webhooks, Slack, or any URL that accepts POST JSON.")
    webhook_url = st.text_input("Webhook URL (optional)", value="", type="password", help="We'll refuse non-HTTPS URLs.")
    st.markdown("---")
    st.subheader("Watchlist for Scanner")
    wl_default = "AAPL, MSFT, NVDA, AMZN, META, GOOGL"
    watchlist = st.text_area("Symbols (comma or newline separated)", value=wl_default, height=80)

tabs = st.tabs([
    "Price & Indicators", "Signals", "Backtest", "Fundamentals",
    "Earnings & Insider", "Watchlist Scanner", "Alerts", "Diagnostics"
])

# ------------------ Tab 1: Price & Indicators ------------------
with tabs[0]:
    df_raw = load_prices(ticker, start, end)
    if df_raw.empty:
        st.warning("No price data. Try a different ticker or date range.")
    else:
        df = compute_indicators(df_raw)
        st.subheader(f"Price & TA — {ticker}")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
        ax.plot(df.index, df["SMA_20"], label="SMA 20")
        ax.plot(df.index, df["SMA_50"], label="SMA 50")
        ax.plot(df.index, df["SMA_200"], label="SMA 200")
        ax.fill_between(df.index, df["BB_LO"], df["BB_UP"], alpha=0.15, label="Bollinger (20,2)")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        st.pyplot(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**RSI (14)**")
            st.line_chart(df[["RSI_14"]])
        with c2:
            st.markdown("**MACD (12/26/9)**")
            st.line_chart(df[["MACD","MACD_signal"]])

        st.markdown("**ATR (14)**")
        st.line_chart(df[["ATR_14"]])

# ------------------ Tab 2: Signals ------------------
with tabs[1]:
    if df_raw.empty:
        st.info("Load price data first.")
    else:
        latest = compute_indicators(df_raw).dropna().iloc[-1]
        sigs = []
        if latest["SMA_20"] > latest["SMA_50"] > latest["SMA_200"]:
            sigs.append("Uptrend: SMA20>SMA50>SMA200")
        elif latest["SMA_20"] < latest["SMA_50"] < latest["SMA_200"]:
            sigs.append("Downtrend: SMA20<SMA50<SMA200")
        if latest["RSI_14"] < 30: sigs.append("RSI oversold (<30)")
        elif latest["RSI_14"] > 70: sigs.append("RSI overbought (>70)")
        if latest["MACD"] > latest["MACD_signal"]: sigs.append("MACD bullish (line>signal)")
        else: sigs.append("MACD bearish (line<=signal)")
        if latest["Close"] >= latest["BB_UP"]: sigs.append("Near/above upper Bollinger band")
        if latest["Close"] <= latest["BB_LO"]: sigs.append("Near/below lower Bollinger band")
        if latest["Close"] >= latest["Hi_20"]: sigs.append("New 20‑day high")
        if latest["Close"] <= latest["Lo_20"]: sigs.append("New 20‑day low")

        score = trade_score(latest)
        st.metric("Trade Score (0–100, heuristic)", f"{score:.0f}")
        st.write("\n".join([f"- {s}" for s in sigs]))
        st.caption("Heuristic score is for personal, educational use only. Not financial advice.")

# ------------------ Tab 3: Backtest ------------------
with tabs[2]:
    if df_raw.empty:
        st.info("Load price data first.")
    else:
        st.subheader("Quick Strategy Backtests (toy models)")
        df = compute_indicators(df_raw).dropna()

        df["pos_sma"] = np.where(df["SMA_20"] > df["SMA_50"], 1, 0)
        df["ret_sma"] = df["pos_sma"].shift() * df["Returns"]
        cum_sma = (1 + df["ret_sma"].fillna(0)).cumprod()

        df["pos_rsi"] = np.where(df["RSI_14"] < 30, 1, np.where(df["RSI_14"] > 70, 0, np.nan))
        df["pos_rsi"] = df["pos_rsi"].ffill().fillna(0)
        df["ret_rsi"] = df["pos_rsi"].shift() * df["Returns"]
        cum_rsi = (1 + df["ret_rsi"].fillna(0)).cumprod()

        cum_bh = (1 + df["Returns"].fillna(0)).cumprod()

        perf = pd.DataFrame({
            "Buy&Hold": cum_bh,
            "SMA(20>50)": cum_sma,
            "RSI<30 long": cum_rsi
        })
        st.line_chart(perf)

        def sharpe(x):
            r = x.pct_change().dropna()
            if r.std() == 0 or len(r) < 2: return 0.0
            return (r.mean() / r.std()) * np.sqrt(252)

        metrics = {
            "CAGR (Buy&Hold)": (perf["Buy&Hold"].iloc[-1])**(252/len(perf)) - 1 if len(perf) > 0 else 0,
            "Sharpe (Buy&Hold)": sharpe(perf["Buy&Hold"]),
            "Sharpe (SMA)": sharpe(perf["SMA(20>50)"]),
            "Sharpe (RSI)": sharpe(perf["RSI<30 long"]),
            "Max Drawdown (Buy&Hold)": (perf["Buy&Hold"]/perf["Buy&Hold"].cummax()-1).min()
        }
        st.write(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]))
        st.caption("Backtests are simplistic and ignore slippage/fees/taxes. Educational only.")

# ------------------ Tab 4: Fundamentals (snapshot) ------------------
with tabs[3]:
    st.subheader("Fundamentals Snapshot")
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        fast = getattr(t, "fast_info", {}) or {}

        price = fast.get("last_price") or info.get("currentPrice")
        pe = info.get("trailingPE") or info.get("forwardPE")
        peg = info.get("pegRatio")
        bvps = info.get("bookValue")
        rec_key = info.get("recommendationKey")
        rec_mean = info.get("recommendationMean")
        roe = info.get("returnOnEquity")
        profit_m = info.get("profitMargins")
        debt_to_eq = info.get("debtToEquity")
        revenue = info.get("totalRevenue")
        gross_m = info.get("grossMargins")

        rows = [
            ("Price", price),
            ("P/E", pe),
            ("PEG", peg),
            ("Book Value / Share", bvps),
            ("Analyst Rating (key)", rec_key),
            ("Analyst Rating (mean)", rec_mean),
            ("ROE", roe),
            ("Profit Margin", profit_m),
            ("Debt to Equity", debt_to_eq),
            ("Revenue", revenue),
            ("Gross Margin", gross_m),
        ]
        df_f = pd.DataFrame(rows, columns=["Metric", "Value"])
        st.dataframe(df_f, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load fundamentals: {e}")

# ------------------ Tab 5: Earnings & Insider ------------------
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Earnings Calendar (next/last)")
        try:
            t = yf.Ticker(ticker)
            ed = None
            try:
                ed = t.earnings_dates
            except Exception:
                pass
            if ed is None or (hasattr(ed, "empty") and ed.empty):
                try:
                    ed = t.get_earnings_dates(limit=8)
                except Exception:
                    ed = None
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                ed = ed.reset_index().rename(columns={"index":"Earnings Date"})
                st.dataframe(ed, use_container_width=True)
            else:
                st.info("No earnings dates available from data source.")
        except Exception as e:
            st.error(f"Earnings lookup failed: {e}")
    with c2:
        st.subheader("Insider Transactions (recent)")
        try:
            t = yf.Ticker(ticker)
            ins = getattr(t, "insider_transactions", None)
            if isinstance(ins, pd.DataFrame) and not ins.empty:
                show_cols = [c for c in ["Date","Insider","Transaction","Value","Shares","Control"] if c in ins.columns]
                st.dataframe(ins[show_cols] if show_cols else ins, use_container_width=True)
            else:
                st.info("No insider transaction data available.")
        except Exception as e:
            st.error(f"Insider data lookup failed: {e}")

# ------------------ Tab 6: Watchlist Scanner + CSV Export + Sector Heatmap ------------------
with tabs[5]:
    st.subheader("Screen your watchlist")
    syms = [s.strip().upper() for s in watchlist.replace("\n", ",").split(",") if s.strip()]
    rows = []
    for sym in syms:
        try:
            dfw = load_prices(sym, start, end)
            if dfw.empty: 
                rows.append({"ticker": sym, "note": "no data"})
                continue
            ind = compute_indicators(dfw).dropna()
            if ind.empty:
                rows.append({"ticker": sym, "note": "insufficient data"})
                continue
            last = ind.iloc[-1]
            info = get_basic_info(sym)
            rows.append({
                "ticker": sym,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "price": round(float(last["Close"]), 2),
                "RSI_14": round(float(last["RSI_14"]), 2),
                "MACD>Signal": bool(last["MACD"] > last["MACD_signal"]),
                "Boll_Squeeze": bool(last.get("BandWidth", np.nan) < 0.05),
                "NewHi20": bool(last["Close"] >= last["Hi_20"]),
                "NewLo20": bool(last["Close"] <= last["Lo_20"]),
                "Near52W_Hi(<=3%)": bool((last["Hi_252"] - last["Close"]) / last["Hi_252"] <= 0.03 if last["Hi_252"] else False),
                "Near52W_Lo(<=3%)": bool((last["Close"] - last["Lo_252"]) / last["Lo_252"] <= 0.03 if last["Lo_252"] else False),
                "Score": round(trade_score(last), 0)
            })
        except Exception as e:
            rows.append({"ticker": sym, "note": f"err: {e}"})
    if rows:
        table = pd.DataFrame(rows)
        order_cols = [c for c in ["ticker","sector","industry","price","RSI_14","MACD>Signal","Boll_Squeeze","NewHi20","NewLo20","Near52W_Hi(<=3%)","Near52W_Lo(<=3%)","Score","note"] if c in table.columns]
        table = table[order_cols]
        st.dataframe(table, use_container_width=True)

        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", data=csv_bytes, file_name="watchlist_scan.csv", mime="text/csv")

        st.subheader("Sector Heatmap")
        if "sector" in table.columns and table["sector"].notna().any():
            signals = {
                "MACD Bull": "MACD>Signal",
                "Squeeze": "Boll_Squeeze",
                "NewHi20": "NewHi20",
                "NewLo20": "NewLo20",
                "Near52W_Hi": "Near52W_Hi(<=3%)",
                "Near52W_Lo": "Near52W_Lo(<=3%)"
            }
            heat = []
            sectors = sorted(table["sector"].dropna().unique())
            for sec in sectors:
                subset = table[table["sector"] == sec]
                row = {"sector": sec}
                n = len(subset)
                for col_lbl, col in signals.items():
                    if col in subset.columns and n > 0:
                        row[col_lbl] = float(subset[col].mean())
                    else:
                        row[col_lbl] = np.nan
                row["Avg Score"] = float(subset["Score"].mean()) if "Score" in subset.columns and n > 0 else np.nan
                heat.append(row)
            heat_df = pd.DataFrame(heat).set_index("sector")
            fig, ax = plt.subplots(figsize=(8, max(2, 0.4*len(heat_df))))
            im = ax.imshow(heat_df.values, aspect="auto")
            ax.set_yticks(range(len(heat_df.index)))
            ax.set_yticklabels(heat_df.index)
            ax.set_xticks(range(len(heat_df.columns)))
            ax.set_xticklabels(heat_df.columns, rotation=45, ha="right")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Sector Signal Intensity (proportion or avg score)")
            st.pyplot(fig)
        else:
            st.info("Sector metadata not available for these symbols.")
    st.caption("Heuristics are for personal, educational use only. Not financial advice.")

# ------------------ Tab 7: Alerts (HTTPS‑only, safe POST) ------------------
with tabs[6]:
    st.subheader("Send an alert to your webhook")
    st.write("Choose a simple condition to check on your Ticker or Watchlist and send a POST to your webhook URL (optional).")
    mode = st.selectbox("Alert source", ["Single Ticker", "Watchlist"])
    condition = st.selectbox("Condition", [
        "RSI<30 (oversold)",
        "RSI>70 (overbought)",
        "Price crosses above SMA_50",
        "Price crosses below SMA_50",
        "New 20-day high",
        "New 20-day low",
        "Bollinger squeeze (bandwidth<0.05)",
        "Trade Score >= 70"
    ])
    if st.button("Check & Send Alert"):
        if not webhook_url:
            st.error("Please add a webhook URL in the sidebar first.")
        elif not is_https_url(webhook_url):
            st.error("Webhook must be HTTPS and include a host.")
        else:
            triggered = []
            def _check(sym):
                df0 = load_prices(sym, start, end)
                ind = compute_indicators(df0).dropna()
                if ind.empty: 
                    return False, None
                last = ind.iloc[-1]
                prev = ind.iloc[-2] if len(ind) > 1 else last
                cond = False
                if condition == "RSI<30 (oversold)":
                    cond = last["RSI_14"] < 30
                elif condition == "RSI>70 (overbought)":
                    cond = last["RSI_14"] > 70
                elif condition == "Price crosses above SMA_50":
                    cond = (prev["Close"] <= prev["SMA_50"]) and (last["Close"] > last["SMA_50"])
                elif condition == "Price crosses below SMA_50":
                    cond = (prev["Close"] >= prev["SMA_50"]) and (last["Close"] < last["SMA_50"])
                elif condition == "New 20-day high":
                    cond = last["Close"] >= last["Hi_20"]
                elif condition == "New 20-day low":
                    cond = last["Close"] <= last["Lo_20"]
                elif condition == "Bollinger squeeze (bandwidth<0.05)":
                    cond = last.get("BandWidth", np.nan) < 0.05
                elif condition == "Trade Score >= 70":
                    cond = trade_score(last) >= 70
                return cond, last

            headers = {"Content-Type": "application/json"}
            if mode == "Single Ticker":
                ok, last = _check(ticker)
                if ok:
                    payload = {
                        "event": "alert",
                        "ticker": ticker,
                        "condition": condition,
                        "price": float(last["Close"]),
                        "time": datetime.utcnow().isoformat() + "Z"
                    }
                    try:
                        r = requests.post(webhook_url, json=payload, headers=headers, timeout=10, allow_redirects=False)
                        st.success(f"Alert sent for {ticker}! HTTP {r.status_code}")
                    except Exception as e:
                        st.error(f"Failed to send webhook: {e}")
                else:
                    st.info("Condition not met right now.")
            else:
                syms = [s.strip().upper() for s in watchlist.replace("\n", ",").split(",") if s.strip()]
                for sym in syms:
                    try:
                        ok, last = _check(sym)
                        if ok:
                            payload = {
                                "event": "alert",
                                "ticker": sym,
                                "condition": condition,
                                "price": float(last["Close"]),
                                "time": datetime.utcnow().isoformat() + "Z"
                            }
                            r = requests.post(webhook_url, json=payload, headers=headers, timeout=10, allow_redirects=False)
                            triggered.append(sym)
                    except Exception:
                        continue
                if triggered:
                    st.success(f"Alerts sent for: {', '.join(triggered)}")
                else:
                    st.info("No symbols met the condition right now.")

# ------------------ Tab 8: Diagnostics & Scheduling Tips ------------------
with tabs[7]:
    st.subheader("Diagnostics")
    try:
        df_raw = load_prices(ticker, start, end)
        ok_price = not df_raw.empty and "Close" in df_raw.columns
        st.write(f"Price data: {'✅' if ok_price else '❌'}")
        df_ind = compute_indicators(df_raw).dropna()
        ok_ind = not df_ind.empty and all(c in df_ind.columns for c in ["SMA_20","RSI_14","MACD","BB_UP","ATR_14"])
        st.write(f"Indicators: {'✅' if ok_ind else '❌'}")
        info = get_basic_info(ticker)
        st.write(f"Sector/Industry: {'✅' if (info.get('sector') or info.get('industry')) else '⚠️ missing'}")
    except Exception as e:
        st.error(f"Diagnostics error: {e}")

    st.subheader("How to schedule daily checks")
    tips = (
        "- Streamlit Cloud does **not** run on a schedule by itself.\n"
        "- Option 1: Create a **GitHub Action (cron)** that runs a small Python script daily which computes your conditions and **POSTs to your webhook**.\n"
        "- Option 2: Use a **serverless function** (Cloud Functions / Lambda free tier) on a schedule to run the same checks.\n"
        "- Keep API usage light to stay within free tiers.\n"
        "- Always review alerts manually before trading. This is for personal, educational use only."
    )
    st.markdown(tips)

st.caption("For personal educational use only. Not financial advice. No automated execution.")