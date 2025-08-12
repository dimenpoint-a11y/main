
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import re, io, pathlib
from datetime import datetime

APP_VERSION = "v6.3.0"

# ---------------- Page & theme ----------------
st.set_page_config(page_title="Trader Dashboard - Pro", layout="wide")

# Sidebar controls (including Dark Mode)
with st.sidebar:
    st.header("Settings")
    dark_mode = st.toggle("🌙 Dark Mode (beta)", value=False, help="Overrides base colors + matplotlib style.")
    auto_refresh = st.toggle("Auto-refresh", value=True, help="Re-run automatically at an interval.")
    interval_s = st.slider("Refresh every (seconds)", 5, 600, 30)
    compact = st.toggle("Compact tables", value=False)
    analytics_toggle = st.toggle("Enable analytics (demo)", value=False, help="Shows a consent banner; no real analytics sent.")
    st.subheader("Watchlist")
    wl_default = "AAPL, MSFT, NVDA, AMZN, META, GOOGL"
    watchlist_raw = st.text_area("Symbols", value=wl_default, height=80)

# CSS theme override (light/dark)
light_css = """
<style>
  .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1200px; }
  .appview-container, .main { background-color: #FFFFFF; color: #111111; }
  .stMetric, .stDataFrame, .stMarkdown { color: #111111; }
  .footer { margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #ddd; font-size: 0.9rem; color: #666; }
  .cookie-banner { padding: 0.75rem; border: 1px solid #e1e1e1; background: #fffbe6; border-radius: 8px; }
</style>
"""
dark_css = """
<style>
  .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1200px; }
  .appview-container, .main { background-color: #0f172a; color: #e5e7eb; }
  .stMetric, .stDataFrame, .stMarkdown, .stCaption, .stText, .stHeader { color: #e5e7eb !important; }
  .stDataFrame { filter: invert(1) hue-rotate(180deg); } /* quick invert for tables */
  .footer { margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #334155; font-size: 0.9rem; color: #cbd5e1; }
  .cookie-banner { padding: 0.75rem; border: 1px solid #334155; background: #111827; border-radius: 8px; }
</style>
"""
st.markdown(dark_css if dark_mode else light_css, unsafe_allow_html=True)

# Matplotlib style align
plt.style.use('dark_background' if dark_mode else 'default')

# Cookie/consent banner
if "consent" not in st.session_state:
    st.session_state.consent = None
if analytics_toggle and st.session_state.consent is None:
    st.markdown("<div class='cookie-banner'>This app may use cookies for preferences/analytics. Do you consent?</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Accept"):
            st.session_state.consent = "accepted"
    with c2:
        if st.button("Decline"):
            st.session_state.consent = "declined"

# Auto-refresh
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        count = st_autorefresh(interval=interval_s * 1000, limit=100000, key="autorefresh_counter")
        st.caption(f"⏱ Auto-refresh ON (every {interval_s}s).")
    except Exception:
        st.caption("Auto-refresh component not installed; running without timed refresh (install 'streamlit-autorefresh').")

# ---------------- Security & validation ----------------
TICKER_RE = re.compile(r"^[A-Z0-9\.\-_]{1,10}$")
def clean_symbol(s: str) -> str:
    s = (s or "").strip().upper()
    return s if TICKER_RE.match(s) else ""

@st.cache_data(show_spinner=False, ttl=300)
def load_prices(ticker: str, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).title() for c in data.columns]
    return data

def ema(series, span): return series.ewm(span=span, adjust=False).mean()

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

# Inputs
left, right = st.columns([1,1])
with left:
    ticker = clean_symbol(st.text_input("Ticker", value="AAPL"))
with right:
    start = st.date_input("Start date", pd.to_datetime("2021-01-01"))
    end = st.date_input("End date", pd.Timestamp.today().date())

# Normalize watchlist
symbols = [clean_symbol(s) for s in watchlist_raw.replace("\n", ",").split(",")]
symbols = [s for s in symbols if s][:100]

tabs = st.tabs([
    "Price & Indicators", "Signals", "Backtest", "Fundamentals",
    "Earnings & Insider", "Watchlist (Charts)", "Heatmap", "Diagnostics",
    "Legal & Disclosures", "Policies (ToS/Privacy/Risk)"
])

# Tab 1: Price & Indicators
with tabs[0]:
    st.title("📊 Trader Dashboard — Pro (Personal Use Only)")
    st.info("Educational tool. Not legal or investment advice. See Legal & Policies tabs.")

    if not ticker:
        st.warning("Enter a valid ticker (A-Z, digits, ., -, _).")
    else:
        df_raw = load_prices(ticker, start, end)
        if df_raw.empty:
            st.warning("No price data. Try a different ticker or date range.")
        else:
            df = compute_indicators(df_raw).dropna()
            st.subheader(f"Price & TA — {ticker}")
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
            ax.plot(df.index, df["SMA_20"], label="SMA 20")
            ax.plot(df.index, df["SMA_50"], label="SMA 50")
            ax.plot(df.index, df["SMA_200"], label="SMA 200")
            ax.fill_between(df.index, df["BB_LO"], df["BB_UP"], alpha=0.15, label="Bollinger (20,2)")
            ax.set_ylabel("Price")
            ax.legend(loc="upper left")
            st.pyplot(fig, use_container_width=True)
            st.caption("How to read: price above longer SMAs (e.g., SMA200) suggests uptrend. "
                       "Upper band touches = strength; lower band touches = weakness. "
                       "Higher price/slope signals trend strength, not value.")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**RSI (14)**")
                st.line_chart(df[["RSI_14"]], use_container_width=True)
                st.caption("RSI: 70+ overbought risk; 30- oversold risk; 40–60 neutral.")
            with c2:
                st.markdown("**MACD (12/26/9)**")
                st.line_chart(df[["MACD","MACD_signal"]], use_container_width=True)
                st.caption("MACD>Signal: bullish momentum; MACD<Signal: bearish. Rising histogram = strengthening momentum.")

            st.markdown("**ATR (14)**")
            st.line_chart(df[["ATR_14"]], use_container_width=True)
            st.caption("ATR (volatility): higher = bigger daily moves (more risk & potential reward); lower = calmer.")

# Tab 2: Signals
with tabs[1]:
    if not ticker:
        st.info("Load price data first.")
    else:
        dfr = load_prices(ticker, start, end)
        if dfr.empty:
            st.info("No data.")
        else:
            latest = compute_indicators(dfr).dropna().iloc[-1]
            parts = {
                "RSI": (10 if latest["RSI_14"] < 30 else (-10 if latest["RSI_14"] > 70 else 0)),
                "MACD": (10 if latest["MACD"] > latest["MACD_signal"] else -5),
                "Trend vs SMA200": (10 if latest["Close"] > latest["SMA_200"] else -10),
                "Squeeze": (5 if latest.get("BandWidth", np.nan) < 0.05 else 0),
            }
            score = trade_score(latest)
            st.metric("Trade Score (0–100, heuristic)", f"{score:.0f}")
            df_parts = pd.DataFrame({"Component": list(parts.keys()), "Contribution": list(parts.values())}).set_index("Component")
            st.bar_chart(df_parts, use_container_width=True)
            st.caption("How to read: positive bars add to the score (bullish factors), negative bars subtract (bearish). "
                       "Higher total score = more bullish alignment. Heuristic only, not advice.")

# Tab 3: Backtest
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
            st.line_chart(perf, use_container_width=True)
            st.caption("Equity curves: higher = better cumulative performance. "
                       "Check smoothness and depth of drawdowns (deeper = worse).")

            def sharpe(x):
                r = x.pct_change().dropna()
                if r.std() == 0 or len(r) < 2: return 0.0
                return (r.mean() / r.std()) * np.sqrt(252)

            met = {
                "CAGR (Buy&Hold)": float((perf["Buy&Hold"].iloc[-1])**(252/len(perf)) - 1) if len(perf) > 0 else 0.0,
                "Sharpe (Buy&Hold)": float(sharpe(perf["Buy&Hold"])),
                "Sharpe (SMA)": float(sharpe(perf["SMA(20>50)"])),
                "Sharpe (RSI)": float(sharpe(perf["RSI<30 long"])),
                "Max DD (Buy&Hold)": float((perf["Buy&Hold"]/perf["Buy&Hold"].cummax()-1).min()),
            }
            met_df = pd.DataFrame({"Metric": list(met.keys()), "Value": list(met.values())}).set_index("Metric")
            st.bar_chart(met_df, use_container_width=True)
            st.caption("Metrics: higher CAGR/Sharpe = better; Max DD more negative = worse. "
                       "Backtests are hypothetical; past performance is not indicative of future results.")

            buf = io.StringIO(); perf.to_csv(buf)
            st.download_button("⬇️ Download equity curves (CSV)", data=buf.getvalue(), file_name=f"{ticker}_backtest_equity.csv", mime="text/csv")

# Tab 4: Fundamentals
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
            ("ROE", info.get("returnOnEquity") if info else None),
            ("Profit Margin", info.get("profitMargins") if info else None),
            ("Debt to Equity", info.get("debtToEquity") if info else None),
            ("Revenue", info.get("totalRevenue") if info else None),
            ("Gross Margin", info.get("grossMargins") if info else None),
        ]
        df_f = pd.DataFrame(rows, columns=["Metric","Value"]).dropna()
        st.dataframe(df_f, use_container_width=True)
        chartable = df_f[df_f["Metric"].isin(["P/E","PEG","ROE","Profit Margin","Gross Margin"])].set_index("Metric")
        if not chartable.empty:
            st.bar_chart(chartable, use_container_width=True)
        st.caption("Meaning: lower P/E/PEG can imply cheaper valuation (context matters). Higher ROE/margins generally better. Lower Debt/Equity usually safer.")
    except Exception as e:
        st.error(f"Could not load fundamentals: {e}")

# Tab 5: Earnings & Insider
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
                    dfe = ed.reset_index()
                    st.dataframe(dfe, use_container_width=True)
                    try:
                        nd = dfe["Earnings Date"].min()
                        if pd.notna(nd):
                            days = (pd.to_datetime(nd).date() - pd.Timestamp.today().date()).days
                            st.metric("Days until next earnings", value=days)
                    except Exception:
                        pass
                else:
                    st.info("No earnings dates available.")
        except Exception as e:
            st.error(f"Earnings lookup failed: {e}")
        st.caption("Meaning: earnings events often increase volatility; adjust risk sizing accordingly.")
    with c2:
        st.subheader("Insider Transactions (last 90d)")
        try:
            if ticker:
                t = yf.Ticker(ticker)
                ins = getattr(t, "insider_transactions", None)
                if isinstance(ins, pd.DataFrame) and not ins.empty:
                    ins2 = ins.copy()
                    if "Date" in ins2.columns:
                        ins2 = ins2[ins2["Date"] >= (pd.Timestamp.today() - pd.Timedelta(days=90))]
                    if not ins2.empty and "Transaction" in ins2.columns and "Value" in ins2.columns:
                        buys = float(ins2[ins2["Transaction"].str.contains("Buy", case=False, na=False)]["Value"].fillna(0).sum())
                        sells = float(ins2[ins2["Transaction"].str.contains("Sell", case=False, na=False)]["Value"].fillna(0).sum())
                        st.bar_chart(pd.DataFrame({"Type":["Buys","Sells"],"Value":[buys,sells]}).set_index("Type"), use_container_width=True)
                    st.dataframe(ins2, use_container_width=True)
                else:
                    st.info("No insider data available.")
        except Exception as e:
            st.error(f"Insider data lookup failed: {e}")
        st.caption("Meaning: net insider buying can be constructive; sales can be neutral or negative depending on context.")

# Tab 6: Watchlist (Charts)
with tabs[5]:
    st.subheader("Watchlist — Chart Views")
    rows = []
    for sym in symbols:
        try:
            dfw = load_prices(sym, start, end)
            if dfw.empty: continue
            ind = compute_indicators(dfw).dropna()
            if ind.empty: continue
            last = ind.iloc[-1]
            rows.append({
                "ticker": sym,
                "price": float(last["Close"]),
                "RSI_14": float(last["RSI_14"]),
                "MACD>Signal": 1 if last["MACD"] > last["MACD_signal"] else 0,
                "Squeeze": 1 if last.get("BandWidth", np.nan) < 0.05 else 0,
                "NewHi20": 1 if last["Close"] >= last["Hi_20"] else 0,
                "NewLo20": 1 if last["Close"] <= last["Lo_20"] else 0,
                "Score": float(trade_score(last)),
            })
        except Exception:
            continue

    if rows:
        wdf = pd.DataFrame(rows)
        st.markdown("**Top by Score**")
        top = wdf.sort_values("Score", ascending=False).head(20).set_index("ticker")[["Score"]]
        st.bar_chart(top, use_container_width=True)
        st.caption("Higher score = more bullish alignment (heuristic).")

        st.markdown("**RSI vs Score**")
        st.scatter_chart(wdf, x="RSI_14", y="Score", use_container_width=True)
        st.caption("Identify stretched (RSI>70, high score) or potential mean-reversion (RSI<30).")

        st.markdown("**Momentum & Breakout States (counts)**")
        counts = pd.DataFrame({
            "MACD Bullish":[int(wdf["MACD>Signal"].sum())],
            "Squeezes":[int(wdf["Squeeze"].sum())],
            "New 20d Highs":[int(wdf["NewHi20"].sum())],
            "New 20d Lows":[int(wdf["NewLo20"].sum())],
        }).T
        counts.columns = ["Count"]
        st.bar_chart(counts, use_container_width=True)
        st.caption("Counts across your watchlist: more 'MACD Bullish' or 'Squeezes' can indicate broad momentum or compression.")
    else:
        st.info("No valid symbols for charts.")

# Tab 7: Heatmap
with tabs[6]:
    st.subheader("Sector Heatmap (from info.sector)")
    rows = []
    for sym in symbols:
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
                         "squeeze": int(last.get("BandWidth", np.nan) < 0.05)})
        except Exception:
            continue
    if rows:
        dfh = pd.DataFrame(rows)
        agg = dfh.groupby("sector").agg(
            tickers=("ticker","count"),
            avg_score=("score","mean"),
            macd_bull=("macd_bull","sum"),
            squeeze=("squeeze","sum"),
        ).sort_values("avg_score", ascending=False)
        st.dataframe(agg, use_container_width=True)
        st.caption("Meaning: higher avg_score = more bullish context in that sector; counts show breadth of momentum/compression.")
    else:
        st.info("No sector data available for this watchlist/date range.")

# Tab 8: Diagnostics
with tabs[7]:
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
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")

# Tab 9: Legal & Disclosures
with tabs[8]:
    st.subheader("Legal & Disclosures (US-centric, not legal advice)")
    st.markdown("""
**Investment adviser risk**  
If you provide personalized investment advice for compensation, you may need state/SEC RIA registration.

**Safer lane**  
Offer tools/education/publishing (bona fide, general, regular-schedule content; no 1:1 tailored advice). Avoid "Buy X now" to a specific user.

**Backtest & performance claims**  
"Past performance is not indicative… Hypothetical results have limitations…" Backtests are hypothetical; don't imply future profits.

**Brokerage/execution**  
Don't execute trades or handle client funds. If you ever do, new regimes apply (BD/RIA/CTA).

**Marketing**  
No "guaranteed" or "beat the market" claims. Keep language factual (what the tool computes/alerts).

**Policies you need on day 1**  
- Terms of Service (no advice, no fiduciary duty, personal use, limitations of liability).  
- Privacy Policy (PII collected, cookies, retention, 3rd-party processors).  
- Risk Disclosure (market risk, data delays, outages, webhooks not guaranteed).  
- Cookie/consent banner if you track analytics.  
- If you serve EU/UK/CA users: GDPR/UK-GDPR/CCPA–CPRA basics (data access/delete, DPA with processors).

**Recordkeeping**  
Keep logs of what was shown/sent (alerts), timestamps, and versions (handy for audits/support).
    """)
    st.caption("For personal educational use only. Not legal or investment advice.")

# Tab 10: Policies (ToS/Privacy/Risk)
with tabs[9]:
    st.subheader("Policies")
    base = pathlib.Path(__file__).parent
    def read_md(name):
        try:
            return (base / name).read_text(encoding="utf-8")
        except Exception:
            return f"# {name}\n(Not found)"
    st.markdown("### Terms of Service", help="Not legal advice. Consult a lawyer for your jurisdiction.")
    st.markdown(read_md("TERMS_OF_SERVICE.md"))
    st.divider()
    st.markdown("### Privacy Policy")
    st.markdown(read_md("PRIVACY_POLICY.md"))
    st.divider()
    st.markdown("### Risk Disclosure")
    st.markdown(read_md("RISK_DISCLOSURE.md"))

# Footer
st.markdown(
    f"""
    <div class="footer">
      <b>Legal links:</b>
      <a href="#policies" title="Scroll to Policies tab">Policies</a> ·
      <a href="#legal--disclosures" title="Scroll to Legal tab">Legal & Disclosures</a>
      <br/>App {APP_VERSION} — Data via Yahoo Finance (yfinance); for personal, educational use.
    </div>
    """,
    unsafe_allow_html=True
)
