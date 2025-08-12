
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests, re, io, json
from datetime import datetime

st.set_page_config(page_title="Trader Dashboard — Pro (Personal Use)", layout="wide")
st.title("📊 Trader Dashboard — Pro (Personal Use Only)")

# ---------------- Security & Validation ----------------
TICKER_RE = re.compile(r"^[A-Z0-9\.\-\_]{1,10}$")
def clean_symbol(s: str) -> str:
    s = s.strip().upper()
    return s if TICKER_RE.match(s) else ""

def safe_webhook(url: str) -> bool:
    # HTTPS only, no redirects, minimal length
    return bool(url) and url.startswith("https://") and len(url) <= 2048

# -------------- Data helpers (cached) ------------------
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
    if not np.isnan(row.get("BandWidth", np.nan)) and row["BandWidth"] is not None and row["BandWidth"] < 0.05:
        score += 5
    return float(np.clip(score, 0, 100))

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    ticker = clean_symbol(st.text_input("Ticker", value="AAPL"))
    start = st.date_input("Start date", pd.to_datetime("2021-01-01"))
    end = st.date_input("End date", pd.Timestamp.today().date())

    st.subheader("Webhook (optional)")
    st.caption("HTTPS only. Use IFTTT/Slack/Make.com or your own endpoint.")
    webhook_url = st.text_input("Webhook URL", value="", type="password")

    st.subheader("Watchlist")
    wl_default = "AAPL, MSFT, NVDA, AMZN, META, GOOGL"
    watchlist_raw = st.text_area("Symbols", value=wl_default, height=80)

tabs = st.tabs([
    "Price & Indicators", "Signals", "Backtest", "Fundamentals",
    "Earnings & Insider", "Watchlist Scanner", "Alerts", "Heatmap", "Diagnostics"
])

# ------------- Tab 1: Price & Indicators -------------
with tabs[0]:
    if not ticker:
        st.warning("Enter a valid ticker (A–Z, digits, ., -, _).")
    else:
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
            st.caption("**How to read:** Rising price and price above longer SMAs (e.g., SMA200) generally = uptrend (bullish context). "
                       "Touches of upper Bollinger band often indicate strength; lower band can indicate weakness. "
                       "_“Higher” price or slope is not always “better”; it indicates trend strength, not value._")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**RSI (14)**")
                st.line_chart(df[["RSI_14"]])
                st.caption("**RSI:** 70+ = overbought risk (not necessarily good); 30− = oversold risk (potential mean‑reversion). "
                           "Middle (40‑60) = neutral. _Extremes rise/fall quickly; “higher” isn’t inherently better._")
            with c2:
                st.markdown("**MACD (12/26/9)**")
                st.line_chart(df[["MACD","MACD_signal"]])
                st.caption("**MACD:** When MACD line is **above** signal line → bullish momentum; **below** → bearish. "
                           "A **rising histogram** suggests strengthening momentum. _More positive isn’t always safer; it’s just stronger trend._")

            st.markdown("**ATR (14)**")
            st.line_chart(df[["ATR_14"]])
            st.caption("**ATR (volatility):** **Higher** = bigger daily moves (more risk + potential reward). "
                       "**Lower** = calmer price action. _Neither is “better”; ATR helps size positions & stops._")

# ------------- Tab 2: Signals -------------
with tabs[1]:
    if not ticker:
        st.info("Load price data first.")
    else:
        dfr = load_prices(ticker, start, end)
        if dfr.empty:
            st.info("No data.")
        else:
            latest = compute_indicators(dfr).dropna().iloc[-1]
            sigs = []
            if latest["SMA_20"] > latest["SMA_50"] > latest["SMA_200"]:
                sigs.append("Uptrend: SMA20>SMA50>SMA200 (bullish context)")
            elif latest["SMA_20"] < latest["SMA_50"] < latest["SMA_200"]:
                sigs.append("Downtrend: SMA20<SMA50<SMA200 (bearish context)")
            if latest["RSI_14"] < 30: sigs.append("RSI oversold (<30) — potential bounce risk/reward")
            elif latest["RSI_14"] > 70: sigs.append("RSI overbought (>70) — potential pullback risk")
            if latest["MACD"] > latest["MACD_signal"]: sigs.append("MACD bullish (line>signal)")
            else: sigs.append("MACD bearish (line<=signal)")
            if latest["Close"] >= latest["BB_UP"]: sigs.append("Near/above upper band (strong or stretched)")
            if latest["Close"] <= latest["BB_LO"]: sigs.append("Near/below lower band (weak or stretched)")

            score = trade_score(latest)
            st.metric("Trade Score (0–100, heuristic)", f"{score:.0f}")
            st.write("\n".join([f"- {s}" for s in sigs]))
            st.caption("**Legend:** Higher score = more bullish signals aligned; lower = more bearish/neutral. "
                       "_Score is heuristic for personal use only — not advice._")

# ------------- Tab 3: Backtest (with CSV export) -------------
with tabs[2]:
    if not ticker:
        st.info("Load price data first.")
    else:
        dfr = load_prices(ticker, start, end)
        if dfr.empty:
            st.info("No data.")
        else:
            st.subheader("Quick Strategy Backtests (toy models)")
            dfb = compute_indicators(dfr).dropna()

            dfb["pos_sma"] = np.where(dfb["SMA_20"] > dfb["SMA_50"], 1, 0)
            dfb["ret_sma"] = dfb["pos_sma"].shift() * dfb["Returns"]
            cum_sma = (1 + dfb["ret_sma"].fillna(0)).cumprod()

            dfb["pos_rsi"] = np.where(dfb["RSI_14"] < 30, 1, np.where(dfb["RSI_14"] > 70, 0, np.nan))
            dfb["pos_rsi"] = dfb["pos_rsi"].ffill().fillna(0)
            dfb["ret_rsi"] = dfb["pos_rsi"].shift() * dfb["Returns"]
            cum_rsi = (1 + dfb["ret_rsi"].fillna(0)).cumprod()

            cum_bh = (1 + dfb["Returns"].fillna(0)).cumprod()

            perf = pd.DataFrame({
                "Buy&Hold": cum_bh,
                "SMA(20>50)": cum_sma,
                "RSI<30 long": cum_rsi
            })
            st.line_chart(perf)
            st.caption("**Equity curves:** **Higher** line = better cumulative performance. "
                       "Compare shapes (smoother is often preferable) and drawdowns (deep dips = higher pain).")

            def sharpe(x):
                r = x.pct_change().dropna()
                if r.std() == 0 or len(r) < 2: return 0.0
                return (r.mean() / r.std()) * np.sqrt(252)

            metrics = {
                "CAGR (Buy&Hold)": float((perf["Buy&Hold"].iloc[-1])**(252/len(perf)) - 1) if len(perf) > 0 else 0.0,
                "Sharpe (Buy&Hold)": float(sharpe(perf["Buy&Hold"])),
                "Sharpe (SMA)": float(sharpe(perf["SMA(20>50)"])),
                "Sharpe (RSI)": float(sharpe(perf["RSI<30 long"])),
                "Max Drawdown (Buy&Hold)": float((perf["Buy&Hold"]/perf["Buy&Hold"].cummax()-1).min())
            }
            st.write(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]))
            st.caption("**Metrics meaning:** Higher **CAGR**/**Sharpe** = better; **Max Drawdown** more negative = worse. "
                       "Toy backtests ignore slippage/fees/taxes.")

            # CSV export
            csv_buf = io.StringIO()
            perf.to_csv(csv_buf)
            st.download_button("⬇️ Download equity curves (CSV)", data=csv_buf.getvalue(), file_name=f"{ticker}_backtest_equity.csv", mime="text/csv")

# ------------- Tab 4: Fundamentals -------------
with tabs[3]:
    st.subheader("Fundamentals Snapshot")
    try:
        t = yf.Ticker(ticker) if ticker else None
        info = t.info if t else {}
        fast = getattr(t, "fast_info", {}) if t else {}

        rows = [
            ("Price", fast.get("last_price") if fast else None),
            ("P/E", info.get("trailingPE") or info.get("forwardPE") if info else None),
            ("PEG", info.get("pegRatio") if info else None),
            ("Book Value / Share", info.get("bookValue") if info else None),
            ("Analyst Rating (key)", info.get("recommendationKey") if info else None),
            ("Analyst Rating (mean)", info.get("recommendationMean") if info else None),
            ("ROE", info.get("returnOnEquity") if info else None),
            ("Profit Margin", info.get("profitMargins") if info else None),
            ("Debt to Equity", info.get("debtToEquity") if info else None),
            ("Revenue", info.get("totalRevenue") if info else None),
            ("Gross Margin", info.get("grossMargins") if info else None),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Metric","Value"]), use_container_width=True)
        st.caption("**Meaning:** Lower **P/E**/**PEG** can imply cheaper valuation (context matters). "
                   "Higher **ROE**/**Gross/Profit Margins** generally better. **Debt/Equity** lower usually safer.")
    except Exception as e:
        st.error(f"Could not load fundamentals: {e}")

# ------------- Tab 5: Earnings & Insider -------------
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Earnings Calendar")
        try:
            if ticker:
                t = yf.Ticker(ticker)
                try:
                    ed = t.earnings_dates
                except Exception:
                    ed = None
                if ed is None or (hasattr(ed, "empty") and ed.empty):
                    try:
                        ed = t.get_earnings_dates(limit=8)
                    except Exception:
                        ed = None
                if isinstance(ed, pd.DataFrame) and not ed.empty:
                    st.dataframe(ed.reset_index(), use_container_width=True)
                else:
                    st.info("No earnings dates available.")
        except Exception as e:
            st.error(f"Earnings lookup failed: {e}")
        st.caption("**Meaning:** Proximity to earnings can increase volatility; consider sizing risk accordingly.")
    with c2:
        st.subheader("Insider Transactions")
        try:
            if ticker:
                t = yf.Ticker(ticker)
                ins = getattr(t, "insider_transactions", None)
                if isinstance(ins, pd.DataFrame) and not ins.empty:
                    show_cols = [c for c in ["Date","Insider","Transaction","Value","Shares","Control"] if c in ins.columns]
                    st.dataframe(ins[show_cols] if show_cols else ins, use_container_width=True)
                else:
                    st.info("No insider data available.")
        except Exception as e:
            st.error(f"Insider data lookup failed: {e}")
        st.caption("**Meaning:** Insider **buys** can be constructive; **sales** can be neutral (liquidity/tax) or negative depending on context.")

# ------------- Tab 6: Watchlist Scanner (CSV export) -------------
with tabs[5]:
    st.subheader("Watchlist Scanner")
    syms = [clean_symbol(s) for s in watchlist_raw.replace("\\n", ",").split(",")]
    syms = [s for s in syms if s]
    rows = []
    for sym in syms:
        try:
            dfw = load_prices(sym, start, end)
            if dfw.empty:
                rows.append({"ticker": sym, "note": "no data"}); continue
            ind = compute_indicators(dfw).dropna()
            if ind.empty:
                rows.append({"ticker": sym, "note": "insufficient data"}); continue
            last = ind.iloc[-1]
            rows.append({
                "ticker": sym,
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
    table = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not table.empty:
        st.dataframe(table, use_container_width=True)
        csv_buf = io.StringIO(); table.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download scanner results (CSV)", data=csv_buf.getvalue(), file_name="watchlist_scan.csv", mime="text/csv")
        st.caption("**Meaning of columns:** "
                   "**RSI_14** lower (<30) = oversold risk; higher (>70) = overbought risk. "
                   "**MACD>Signal** True = bullish momentum. **Boll_Squeeze** True = low volatility (potential breakout). "
                   "**NewHi20/Lo20** True = fresh momentum extremes. **Score** higher = more bullish factors aligned.")
    else:
        st.info("No results.")

# ------------- Tab 7: Alerts (hardened webhook) -------------
with tabs[6]:
    st.subheader("Webhook Alerts")
    st.write("Checks a simple condition and sends a JSON payload to your webhook if met.")
    if webhook_url and not safe_webhook(webhook_url):
        st.error("Webhook must be HTTPS and reasonable length.")
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

    def check_symbol(sym):
        df0 = load_prices(sym, start, end)
        ind = compute_indicators(df0).dropna()
        if ind.empty: return False, None
        last, prev = ind.iloc[-1], ind.iloc[-2] if len(ind)>1 else ind.iloc[-1]
        cond = False
        if condition == "RSI<30 (oversold)": cond = last["RSI_14"] < 30
        elif condition == "RSI>70 (overbought)": cond = last["RSI_14"] > 70
        elif condition == "Price crosses above SMA_50": cond = (prev["Close"] <= prev["SMA_50"]) and (last["Close"] > last["SMA_50"])
        elif condition == "Price crosses below SMA_50": cond = (prev["Close"] >= prev["SMA_50"]) and (last["Close"] < last["SMA_50"])
        elif condition == "New 20-day high": cond = last["Close"] >= last["Hi_20"]
        elif condition == "New 20-day low": cond = last["Close"] <= last["Lo_20"]
        elif condition == "Bollinger squeeze (bandwidth<0.05)": cond = last.get("BandWidth", np.nan) < 0.05
        elif condition == "Trade Score >= 70": cond = trade_score(last) >= 70
        return cond, last

    if st.button("Check & Send"):
        if not (webhook_url and safe_webhook(webhook_url)):
            st.error("Add a valid HTTPS webhook URL in the sidebar first.")
        else:
            headers = {"Content-Type": "application/json"}
            if mode == "Single Ticker":
                sym = ticker
                ok, last = check_symbol(sym)
                if ok:
                    payload = {"event":"alert","ticker":sym,"condition":condition,"price":float(last["Close"]),"time":datetime.utcnow().isoformat()+"Z"}
                    try:
                        r = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=10, allow_redirects=False)
                        st.success(f"Alert sent for {sym} — HTTP {r.status_code}")
                    except Exception as e:
                        st.error(f"Webhook error: {e}")
                else:
                    st.info("Condition not met.")
            else:
                syms = [clean_symbol(s) for s in watchlist_raw.replace("\\n", ",").split(",")]
                syms = [s for s in syms if s]
                triggered = []
                for sym in syms:
                    try:
                        ok, last = check_symbol(sym)
                        if ok:
                            payload = {"event":"alert","ticker":sym,"condition":condition,"price":float(last["Close"]),"time":datetime.utcnow().isoformat()+"Z"}
                            r = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=10, allow_redirects=False)
                            if 200 <= r.status_code < 300:
                                triggered.append(sym)
                    except Exception:
                        continue
                if triggered:
                    st.success("Alerts sent for: " + ", ".join(triggered))
                else:
                    st.info("No symbols met the condition.")

# ------------- Tab 8: Heatmap (by sector) -------------
with tabs[7]:
    st.subheader("Sector Heatmap (from info.sector)")
    syms = [clean_symbol(s) for s in watchlist_raw.replace("\\n", ",").split(",")]
    syms = [s for s in syms if s]
    rows = []
    for sym in syms:
        try:
            dfw = load_prices(sym, start, end)
            if dfw.empty: continue
            ind = compute_indicators(dfw).dropna()
            if ind.empty: continue
            last = ind.iloc[-1]
            info = {}
            try:
                info = yf.Ticker(sym).info or {}
            except Exception:
                info = {}
            sector = info.get("sector","Unknown")
            rows.append({"sector": sector, "ticker": sym, "score": trade_score(last),
                         "macd_bull": int(last["MACD"] > last["MACD_signal"]),
                         "squeeze": int(last.get("BandWidth", np.nan) < 0.05),
                         "newhi20": int(last["Close"] >= last["Hi_20"]),
                         "newlo20": int(last["Close"] <= last["Lo_20"])})
        except Exception:
            continue
    if rows:
        dfh = pd.DataFrame(rows)
        agg = dfh.groupby("sector").agg(
            tickers=("ticker","count"),
            avg_score=("score","mean"),
            macd_bull=("macd_bull","sum"),
            squeeze=("squeeze","sum"),
            newhi20=("newhi20","sum"),
            newlo20=("newlo20","sum"),
        ).sort_values("avg_score", ascending=False)
        st.dataframe(agg, use_container_width=True)
        st.caption("**Meaning:** **avg_score** higher = watchlist more bullish in that sector. "
                   "**macd_bull/squeeze/newhi20/newlo20** are **counts** across your symbols.")
    else:
        st.info("No sector data available for this watchlist/date range.")

# ------------- Tab 9: Diagnostics -------------
with tabs[8]:
    st.subheader("Diagnostics")
    try:
        if not ticker:
            st.info("Enter a ticker to test.")
        else:
            df_test = load_prices(ticker, start, end)
            st.write(f"Rows: {len(df_test)}, Columns: {list(df_test.columns)}")
            ind = compute_indicators(df_test)
            na_rate = ind.isna().mean().round(3).to_dict()
            st.write("NA rate by column:", na_rate)
            ok_cols = all(c in ind.columns for c in ["RSI_14","MACD","MACD_signal","ATR_14","BandWidth","SMA_200"])
            st.write("Indicators present:", ok_cols)
            st.success("Diagnostics complete.")
            st.caption("**Meaning:** Low NA rates and presence of all indicator columns indicate healthy data for the date range.")
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")

st.caption("For personal educational use only. Not financial advice. No automated execution.")
