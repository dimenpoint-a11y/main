import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import re, io, pytz
from datetime import datetime, date, time as dtime
from fpdf import FPDF

APP_VERSION = "v6.4.4"

st.set_page_config(page_title="Trader Dashboard - Pro", layout="wide")

# ---------- Helpers ----------
def to_pdf_bytes(title: str, text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16); pdf.multi_cell(0, 10, title); pdf.ln(2)
    pdf.set_font("Arial", "", 11)
    clean = re.sub(r"[#*_`>]+", "", text)
    for line in clean.splitlines():
        pdf.multi_cell(0, 6, line)
    return pdf.output(dest='S').encode('latin-1')

def local_now_chicago():
    tz = pytz.timezone("America/Chicago")
    return datetime.now(tz)

def is_night_chicago():
    now = local_now_chicago().time()
    return (now >= dtime(19,0)) or (now <= dtime(7,0))

def header(icon: str, text: str, level: int = 2):
    tag = "h1" if level == 1 else ("h2" if level == 2 else "h3")
    st.markdown(f"""
    <div class="title-row {tag}">
        <span class="emoji">{icon}</span>
        <span class="title-text">{text}</span>
    </div>
    """, unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    dm_mode = st.radio("Dark Mode", ["Auto (by clock)", "Light", "Dark"], index=0)
    auto_refresh = st.toggle("Auto-refresh", value=True, help="JS-based reload; no extra packages.")
    interval_s = st.slider("Refresh every (seconds)", 5, 600, 30)
    st.subheader("Organization (for Policies)")
    org_name = st.text_input("Organization name", value="Your Company LLC")
    jurisdiction = st.text_input("Jurisdiction (state/country)", value="Illinois, USA")
    st.subheader("Watchlist")
    wl_default = "AAPL, MSFT, NVDA, AMZN, META, GOOGL"
    watchlist_raw = st.text_area("Symbols", value=wl_default, height=80)

# ---------- Theme CSS ----------
COMMON_CSS = """
<style>
  .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1200px; }
  .title-row { display: flex; align-items: center; gap: .5rem; margin: .25rem 0 .5rem; }
  .title-row.h1 { margin-top: .25rem; }
  .title-row.h1 .emoji { font-size: 1.6rem; }
  .title-row.h2 .emoji { font-size: 1.3rem; }
  .title-row.h3 .emoji { font-size: 1.15rem; }
  .title-row .emoji { display: inline-flex; line-height: 1; vertical-align: middle; }
  .title-row .title-text { line-height: 1.1; }
  .badge { display:inline-block; padding: .15rem .5rem; border-radius: .5rem; font-size:.85rem; font-weight:600; }
  .badge.neutral { background: var(--badge-bg); color: var(--badge-fg); border:1px solid var(--badge-br); }
  .footer { margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid var(--border); font-size: 0.9rem; color: var(--muted); }
  .stDataFrame tbody tr td, .stDataFrame thead tr th { padding-top: .4rem; padding-bottom: .4rem; }
</style>
"""
LIGHT_VARS = "<style>:root{ --badge-bg:#eef2ff; --badge-fg:#1e293b; --badge-br:#c7d2fe; --border:#e5e7eb; --muted:#6b7280; }</style>"
DARK_VARS  = "<style>:root{ --badge-bg:#1f2937; --badge-fg:#e5e7eb; --badge-br:#334155; --border:#334155; --muted:#cbd5e1; }</style>"

light_css = "<style>.appview-container, .main { background-color: #FFFFFF; color: #111111; }</style>"
dark_css = "<style>.appview-container, .main { background-color: #0f172a; color: #e5e7eb; } .stDataFrame { filter: invert(1) hue-rotate(180deg); }</style>"

auto_dark = (dm_mode == "Auto (by clock)" and is_night_chicago()) or (dm_mode == "Dark")
st.markdown(COMMON_CSS, unsafe_allow_html=True)
st.markdown(DARK_VARS if auto_dark else LIGHT_VARS, unsafe_allow_html=True)
st.markdown(dark_css if auto_dark else light_css, unsafe_allow_html=True)
plt.style.use('dark_background' if auto_dark else 'default')

# ---------- JS auto-refresh ----------
if "refresh_count" not in st.session_state: st.session_state.refresh_count = 0
if auto_refresh:
    st.session_state.refresh_count += 1
    st.caption(f"⏱ Auto-refresh ON — every {interval_s}s · Count: {st.session_state.refresh_count}")
    st.markdown(f"<script>setTimeout(()=>window.location.reload(), {interval_s * 1000});</script>", unsafe_allow_html=True)
else:
    st.caption("⏱ Auto-refresh OFF")

# ---------- Data helpers ----------
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
    out["Above_SMA200"] = out["Close"] > out["SMA_200"]
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

# ---------- Inputs ----------
left, right = st.columns([1,1])
with left:
    ticker = clean_symbol(st.text_input("Ticker", value="AAPL"))
with right:
    start = st.date_input("Start date", pd.to_datetime("2021-01-01"))
    end = st.date_input("End date", pd.Timestamp.today().date())

symbols = [clean_symbol(s) for s in watchlist_raw.replace("\n", ",").split(",")]
symbols = [s for s in symbols if s][:100]

tabs = st.tabs([
    "Price & Indicators", "Signals", "Backtest", "Fundamentals",
    "Earnings & Insider", "Watchlist (Charts)", "Heatmap", "Diagnostics",
    "Legal & Disclosures", "Policies (ToS/Privacy/Risk)"
])

# ---------- Tab 1 ----------
with tabs[0]:
    header("📊", "Trader Dashboard — Pro (Personal Use Only)", level=1)
    st.info("Educational tool. Not legal or investment advice. See Legal & Policies tabs.")
    if not ticker:
        st.warning("Enter a valid ticker (A-Z, digits, ., -, _).")
    else:
        df_raw = load_prices(ticker, start, end)
        if df_raw.empty:
            st.warning("No price data. Try a different ticker or date range.")
        else:
            df = compute_indicators(df_raw).dropna()
            header("📈", f"Price & TA — {ticker}", level=2)
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
            ax.plot(df.index, df["SMA_20"], label="SMA 20")
            ax.plot(df.index, df["SMA_50"], label="SMA 50")
            ax.plot(df.index, df["SMA_200"], label="SMA 200")
            ax.fill_between(df.index, df["BB_LO"], df["BB_UP"], alpha=0.15, label="Bollinger (20,2)")
            ax.set_ylabel("Price"); ax.legend(loc="upper left")
            st.pyplot(fig, use_container_width=True)
            st.caption("Meaning: price above SMA200 suggests uptrend; band touches show potential strength (upper) or weakness (lower).")

            c1, c2 = st.columns(2)
            with c1:
                header("📏", "RSI (14)", level=3)
                st.line_chart(df[["RSI_14"]], use_container_width=True)
                st.caption("Meaning: higher RSI → more overbought (70+), lower RSI → more oversold (≤30).")
            with c2:
                header("📉", "MACD (12/26/9)", level=3)
                st.line_chart(df[["MACD","MACD_signal"]], use_container_width=True)
                st.caption("Meaning: MACD above Signal = bullish momentum; below = bearish; rising histogram = strengthening.")

            header("🌪️", "ATR (14)", level=3)
            st.line_chart(df[["ATR_14"]], use_container_width=True)
            st.caption("Meaning: higher ATR = larger daily ranges (more risk & potential reward); lower ATR = calmer.")

# ---------- Tab 2 ----------
with tabs[1]:
    header("🧭", "Signals", level=2)
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
            st.caption("Meaning: positive bars add to the score (bullish), negative bars subtract (bearish). Heuristic only.")

# ---------- Tab 3 ----------
with tabs[2]:
    header("🧪", "Backtest", level=2)
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

            perf = pd.DataFrame({"Buy&Hold": cum_bh, "SMA(20>50)": cum_sma, "RSI<30 long": cum_rsi})
            st.line_chart(perf, use_container_width=True)
            st.caption("Meaning: higher equity curve → better cumulative performance.")

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
            st.caption("Meaning: higher CAGR/Sharpe better; more negative Max DD worse. Past performance ≠ future results.")

            buf = io.StringIO(); perf.to_csv(buf)
            st.download_button("⬇️ Download equity curves (CSV)", data=buf.getvalue(), file_name=f"{ticker}_backtest_equity.csv", mime="text/csv")

# ---------- Tab 4 (Fundamentals: TABLE ONLY) ----------
with tabs[3]:
    header("📚", "Fundamentals Snapshot (Table)", level=2)
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
        st.caption("Meaning: lower P/E/PEG can imply cheaper valuation (context matters). Higher ROE/margins generally better. Lower Debt/Equity usually safer.")
    except Exception as e:
        st.error(f"Could not load fundamentals: {e}")

# ---------- Tab 5 (Earnings & Insider: TABLES) ----------
with tabs[4]:
    header("📅", "Earnings & Insider", level=2)
    c1, c2 = st.columns(2)
    with c1:
        header("🗓️", "Earnings Calendar", level=3)
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
        st.caption("Earnings events often increase volatility; adjust risk sizing accordingly.")
    with c2:
        header("🧾", "Insider Transactions (last 90d)", level=3)
        try:
            if ticker:
                t = yf.Ticker(ticker)
                ins = getattr(t, "insider_transactions", None)
                if isinstance(ins, pd.DataFrame) and not ins.empty:
                    ins2 = ins.copy()
                    if "Date" in ins2.columns:
                        ins2 = ins2[ins2["Date"] >= (pd.Timestamp.today() - pd.Timedelta(days=90))]
                    # Summary metrics (no charts)
                    if not ins2.empty and "Transaction" in ins2.columns and "Value" in ins2.columns:
                        buys = float(ins2[ins2["Transaction"].str.contains("Buy", case=False, na=False)]["Value"].fillna(0).sum())
                        sells = float(ins2[ins2["Transaction"].str.contains("Sell", case=False, na=False)]["Value"].fillna(0).sum())
                        st.metric("Net $ Bought (90d)", f"{(buys - sells):,.0f}")
                        st.metric("# Transactions (90d)", f"{len(ins2):,}")
                    st.dataframe(ins2, use_container_width=True)
                else:
                    st.info("No insider data available.")
        except Exception as e:
            st.error(f"Insider data lookup failed: {e}")
        st.caption("Net insider buying can be constructive; sales can be neutral or negative depending on context.")

# ---------- Tab 6 (Watchlist Charts) ----------
with tabs[5]:
    header("👀", "Watchlist — Chart Views", level=2)
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
                "Above200": 1 if last["Close"] > last["SMA_200"] else 0,
                "Score": float(trade_score(last)),
            })
        except Exception:
            continue
    if rows:
        wdf = pd.DataFrame(rows)
        st.markdown('<span class="badge neutral">Top by Score</span>', unsafe_allow_html=True)
        top = wdf.sort_values("Score", ascending=False).head(20).set_index("ticker")[["Score"]]
        st.bar_chart(top, use_container_width=True)
        st.caption("Higher score = more bullish alignment (heuristic).")
        st.markdown('<span class="badge neutral">RSI vs Score</span>', unsafe_allow_html=True)
        st.scatter_chart(wdf, x="RSI_14", y="Score", use_container_width=True)
        st.caption("Identify stretched (RSI>70, high score) or potential mean-reversion (RSI<30).")
    else:
        st.info("No valid symbols for charts.")

# ---------- Tab 7 (Meaningful Heatmap) ----------
with tabs[6]:
    header("🗺️", "Sector Heatmap — Breadth & Momentum", level=2)
    rows = []
    for sym in symbols:
        try:
            dfw = load_prices(sym, start, end)
            if dfw.empty: continue
            ind = compute_indicators(dfw).dropna()
            if ind.empty: continue
            last = ind.iloc[-1]
            try:
                info = yf.Ticker(sym).info or {}
            except Exception:
                info = {}
            sector = info.get("sector","Unknown")
            rows.append({
                "sector": sector, "ticker": sym,
                "score": trade_score(last),
                "rsi": float(last["RSI_14"]),
                "above200": int(last["Close"] > last["SMA_200"]),
                "newhi20": int(last["Close"] >= last["Hi_20"]),
                "newlo20": int(last["Close"] <= last["Lo_20"]),
                "macd_bull": int(last["MACD"] > last["MACD_signal"]),
                "squeeze": int(last.get("BandWidth", np.nan) < 0.05),
            })
        except Exception:
            continue
    if rows:
        dfh = pd.DataFrame(rows)
        agg = dfh.groupby("sector").agg(
            tickers=("ticker","count"),
            avg_score=("score","mean"),
            avg_rsi=("rsi","mean"),
            pct_above200=("above200","mean"),
            pct_hi20=("newhi20","mean"),
            pct_lo20=("newlo20","mean"),
            pct_macd_bull=("macd_bull","mean"),
            pct_squeeze=("squeeze","mean"),
        ).reset_index()
        # Ranking score (simple composite)
        agg["sector_rank"] = (
            agg["avg_score"] +
            20*agg["pct_above200"] +
            10*agg["pct_hi20"] -
            10*agg["pct_lo20"]
        )
        agg = agg.sort_values("sector_rank", ascending=False)
        # Pretty formatting
        show = agg.copy()
        for col in ["avg_score","avg_rsi","sector_rank"]:
            show[col] = show[col].round(1)
        for col in ["pct_above200","pct_hi20","pct_lo20","pct_macd_bull","pct_squeeze"]:
            show[col] = (show[col]*100).round(0).astype(int).astype(str) + "%"
        st.dataframe(show.set_index("sector"), use_container_width=True)
        st.caption("Meaning: higher sector_rank & % above 200D indicate stronger breadth; high % new 20d highs suggests breakouts; many 20d lows flags weakness.")
    else:
        st.info("No sector data available for this watchlist/date range.")

# ---------- Tab 8 ----------
with tabs[7]:
    header("🔧", "Diagnostics", level=2)
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

# ---------- Tab 9 (Simplified Legal) ----------
with tabs[8]:
    header("⚖️", "Legal & Disclosures (simplified; not legal advice)", level=2)
    st.markdown("""
- Educational tool for **personal use** only. **Not investment advice**.
- No trade execution; no access to accounts or funds.
- **Backtests are hypothetical** and have limitations. **Past performance is not indicative of future results**.
- No guarantees. Use at your own risk.
    """)

# ---------- Tab 10 (Policies) ----------
with tabs[9]:
    header("📜", "Policies (customized)", level=2)
    today = date.today().isoformat()
    tos = f"""# Terms of Service
_Last updated: {today}_

- No advisory relationship; {org_name} does **not** provide personalized investment advice.
- Personal, non-commercial use only.
- No guarantees; markets involve risk.
- No trade execution or custody.
- Data may be delayed or inaccurate.
- Liability limited to the maximum extent permitted by law.
- Governed by the laws of {jurisdiction}.
"""
    privacy = "# Privacy Policy\n\nMinimal collection; no sale of data; cookies only for preferences; yfinance as data source."
    risk = "# Risk Disclosure\n\nTrading involves risk; backtests are hypothetical; past performance is not indicative of future results."

    st.markdown("### Terms of Service"); st.markdown(tos)
    st.download_button("⬇️ Download ToS (.md)", data=tos.encode("utf-8"), file_name="TERMS_OF_SERVICE.md", mime="text/markdown")
    st.download_button("⬇️ Download ToS (PDF)", data=to_pdf_bytes("Terms of Service", tos), file_name="TERMS_OF_SERVICE.pdf", mime="application/pdf")
    st.divider()
    st.markdown("### Privacy Policy"); st.markdown(privacy)
    st.download_button("⬇️ Download Privacy (.md)", data=privacy.encode("utf-8"), file_name="PRIVACY_POLICY.md", mime="text/markdown")
    st.download_button("⬇️ Download Privacy (PDF)", data=to_pdf_bytes("Privacy Policy", privacy), file_name="PRIVACY_POLICY.pdf", mime="application/pdf")
    st.divider()
    st.markdown("### Risk Disclosure"); st.markdown(risk)
    st.download_button("⬇️ Download Risk (.md)", data=risk.encode("utf-8"), file_name="RISK_DISCLOSURE.md", mime="text/markdown")
    st.download_button("⬇️ Download Risk (PDF)", data=to_pdf_bytes("Risk Disclosure", risk), file_name="RISK_DISCLOSURE.pdf", mime="application/pdf")

# ---------- Footer ----------
st.markdown(
    f"""
    <div class="footer">
      <b>Legal links:</b>
      <a href="#policies-customized" title="Scroll to Policies section">Policies</a> ·
      <a href="#legal--disclosures-simplified-not-legal-advice" title="Scroll to Legal section">Legal & Disclosures</a>
      <br/>App {APP_VERSION} — Data via Yahoo Finance (yfinance); for personal, educational use.
    </div>
    """,
    unsafe_allow_html=True
)