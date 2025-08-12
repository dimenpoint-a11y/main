
# app_trader_pro_v642.py (same content as shared inline)
# [Truncated header comment to keep size modest in this cell]
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import re, io, pytz
from datetime import datetime, date, time as dtime
from fpdf import FPDF

APP_VERSION = "v6.4.2"

st.set_page_config(page_title="Trader Dashboard - Pro", layout="wide")

def to_pdf_bytes(title: str, text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16); pdf.multi_cell(0, 10, title); pdf.ln(2)
    pdf.set_font("Arial", "", 11)
    clean = re.sub(r"[#*_`>]+", "", text)
    for line in clean.splitlines():
        pdf.multi_cell(0, 6, line)
    buf = io.BytesIO(); pdf.output(buf); return buf.getvalue()

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

with st.sidebar:
    st.header("Settings")
    dm_mode = st.radio("Dark Mode", ["Auto (by clock)", "Light", "Dark"], index=0)
    auto_refresh = st.toggle("Auto-refresh", value=True, help="JS-based reload; no extra packages.")
    interval_s = st.slider("Refresh every (seconds)", 5, 600, 30)
    analytics_toggle = st.toggle("Enable analytics (demo)", value=False)
    st.subheader("Organization (for Policies)")
    org_name = st.text_input("Organization name", value="Your Company LLC")
    jurisdiction = st.text_input("Jurisdiction (state/country)", value="Illinois, USA")
    contact_email = st.text_input("Contact email", value="bgajjela@gmail.com")
    st.subheader("Watchlist")
    wl_default = "AAPL, MSFT, NVDA, AMZN, META, GOOGL"
    watchlist_raw = st.text_area("Symbols", value=wl_default, height=80)

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

if "consent" not in st.session_state:
    st.session_state.consent = None
if analytics_toggle and st.session_state.consent is None:
    st.info("This app may use cookies for preferences/analytics. Do you consent?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Accept"): st.session_state.consent = "accepted"
    with c2:
        if st.button("Decline"): st.session_state.consent = "declined"

if "refresh_count" not in st.session_state: st.session_state.refresh_count = 0
if auto_refresh:
    st.session_state.refresh_count += 1
    st.caption(f"⏱ Auto-refresh ON — every {interval_s}s · Count: {st.session_state.refresh_count}")
    st.markdown(f"<script>setTimeout(()=>window.location.reload(), {interval_s * 1000});</script>", unsafe_allow_html=True)
else:
    st.caption("⏱ Auto-refresh OFF")

import re
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

with tabs[0]:
    st.info("Educational tool. Not legal or investment advice. See Legal & Policies tabs.")
    if not ticker:
        st.warning("Enter a valid ticker (A-Z, digits, ., -, _).")
    else:
        df_raw = load_prices(ticker, start, end)
        if df_raw.empty:
            st.warning("No price data. Try a different ticker or date range.")
        else:
            df = compute_indicators(df_raw).dropna()
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
            ax.plot(df.index, df["SMA_20"], label="SMA 20")
            ax.plot(df.index, df["SMA_50"], label="SMA 50")
            ax.plot(df.index, df["SMA_200"], label="SMA 200")
            ax.fill_between(df.index, df["BB_LO"], df["BB_UP"], alpha=0.15, label="Bollinger (20,2)")
            ax.set_ylabel("Price"); ax.legend(loc="upper left")
            st.pyplot(fig, use_container_width=True)
            st.caption("How to read: price above longer SMAs suggests uptrend; band touches indicate strength/weakness.")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**RSI (14)**"); st.line_chart(df[["RSI_14"]], use_container_width=True)
                st.caption("RSI: 70+ overbought; 30- oversold; 40–60 neutral.")
            with c2:
                st.markdown("**MACD (12/26/9)**"); st.line_chart(df[["MACD","MACD_signal"]], use_container_width=True)
                st.caption("MACD>Signal: bullish; MACD<Signal: bearish.")

            st.markdown("**ATR (14)**"); st.line_chart(df[["ATR_14"]], use_container_width=True)
            st.caption("ATR: higher = more volatility; lower = calmer.")

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
            import pandas as _pd
            df_parts = _pd.DataFrame({"Component": list(parts.keys()), "Contribution": list(parts.values())}).set_index("Component")
            st.bar_chart(df_parts, use_container_width=True)
            st.caption("Positive bars add to score (bullish); negative bars subtract (bearish). Heuristic only.")

with tabs[2]:
    if not ticker:
        st.info("Load price data first.")
    else:
        dfr = load_prices(ticker, start, end)
        if dfr.empty: st.info("No data.")
        else:
            st.subheader("Quick Strategy Backtests (toy models)")
            dfb = compute_indicators(dfr).dropna()
            import numpy as _np, pandas as _pd
            dfb["pos_sma"] = _np.where(dfb["SMA_20"] > dfb["SMA_50"], 1, 0)
            dfb["ret_sma"] = dfb["pos_sma"].shift() * dfb["Returns"]
            cum_sma = (1 + dfb["ret_sma"].fillna(0)).cumprod()
            dfb["pos_rsi"] = _np.where(dfb["RSI_14"] < 30, 1, _np.where(dfb["RSI_14"] > 70, 0, _np.nan))
            dfb["pos_rsi"] = dfb["pos_rsi"].ffill().fillna(0)
            dfb["ret_rsi"] = dfb["pos_rsi"].shift() * dfb["Returns"]
            cum_rsi = (1 + dfb["ret_rsi"].fillna(0)).cumprod()
            cum_bh = (1 + dfb["Returns"].fillna(0)).cumprod()
            perf = _pd.DataFrame({"Buy&Hold": cum_bh, "SMA(20>50)": cum_sma, "RSI<30 long": cum_rsi})
            st.line_chart(perf, use_container_width=True)
            st.caption("Equity curves: higher = better cumulative performance.")
            def sharpe(x):
                r = x.pct_change().dropna()
                if r.std() == 0 or len(r) < 2: return 0.0
                return (r.mean() / r.std()) * _np.sqrt(252)
            met = {
                "CAGR (Buy&Hold)": float((perf["Buy&Hold"].iloc[-1])**(252/len(perf)) - 1) if len(perf) > 0 else 0.0,
                "Sharpe (Buy&Hold)": float(sharpe(perf["Buy&Hold"])),
                "Sharpe (SMA)": float(sharpe(perf["SMA(20>50)"])),
                "Sharpe (RSI)": float(sharpe(perf["RSI<30 long"])),
                "Max DD (Buy&Hold)": float((perf["Buy&Hold"]/perf["Buy&Hold"].cummax()-1).min()),
            }
            met_df = _pd.DataFrame({"Metric": list(met.keys()), "Value": list(met.values())}).set_index("Metric")
            st.bar_chart(met_df, use_container_width=True)
            st.caption("Higher CAGR/Sharpe = better; Max DD more negative = worse. Past performance ≠ future results.")
            buf = io.StringIO(); perf.to_csv(buf)
            st.download_button("⬇️ Download equity curves (CSV)", data=buf.getvalue(), file_name=f"{ticker}_backtest_equity.csv", mime="text/csv")

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
        st.caption("Lower P/E/PEG can imply cheaper valuation; higher ROE/margins better; lower Debt/Equity safer (context needed).")
    except Exception as e:
        st.error(f"Could not load fundamentals: {e}")

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
        st.caption("Earnings often increase volatility; manage risk sizing.")
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
        st.caption("Net insider buying can be constructive; sales can be neutral or negative depending on context.")

with tabs[5]:
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
        top = wdf.sort_values("Score", ascending=False).head(20).set_index("ticker")[["Score"]]
        st.bar_chart(top, use_container_width=True); st.caption("Higher score = more bullish alignment (heuristic).")
        st.scatter_chart(wdf, x="RSI_14", y="Score", use_container_width=True); st.caption("RSI>70: stretched; RSI<30: potential mean-reversion.")
        counts = pd.DataFrame({
            "MACD Bullish":[int(wdf["MACD>Signal"].sum())],
            "Squeezes":[int(wdf["Squeeze"].sum())],
            "New 20d Highs":[int(wdf["NewHi20"].sum())],
            "New 20d Lows":[int(wdf["NewLo20"].sum())],
        }).T; counts.columns = ["Count"]
        st.bar_chart(counts, use_container_width=True); st.caption("Breadth across watchlist.")
    else:
        st.info("No valid symbols for charts.")

with tabs[6]:
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
        st.dataframe(agg, use_container_width=True); st.caption("Higher avg_score = more bullish sector context.")
    else:
        st.info("No sector data available for this watchlist/date range.")

with tabs[7]:
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

with tabs[8]:
    st.markdown("**Investment adviser risk** – Registration may be required for personalized advice for compensation.")
    st.markdown("**Safer lane** – Tools/education/publishing; avoid targeted 'Buy now' advice.")
    st.markdown("**Backtest & performance claims** – Past performance not indicative; hypothetical results have limitations.")
    st.markdown("**Brokerage/execution** – Do not execute trades or handle client funds.")
    st.markdown("**Marketing** – Avoid guarantees; describe what the tool computes/alerts.")
    st.markdown("**Policies day 1** – ToS, Privacy, Risk Disclosure, Cookie banner, GDPR/CCPA basics if applicable.")
    st.markdown("**Recordkeeping** – Keep logs of alerts, timestamps, versions.")
    st.caption("For personal educational use only. Not legal or investment advice.")

with tabs[9]:
    today = date.today().isoformat()
    tos = f"# Terms of Service\n_Last updated: {today}_\n\nNot investment advice..."
    privacy = "# Privacy Policy\n\nMinimal collection; no sale of data; cookies only for preferences; yfinance as data source."
    risk = "# Risk Disclosure\n\nTrading involves risk; past performance not indicative; hypothetical backtests."
    st.markdown("### Terms of Service"); st.markdown(tos)
    st.download_button("⬇️ Download ToS (.md)", data=tos.encode("utf-8"), file_name="TERMS_OF_SERVICE.md", mime="text/markdown")
    from fpdf import FPDF
    def to_pdf_bytes(title: str, text: str) -> bytes:
        pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
        pdf.set_font("Arial", "B", 16); pdf.multi_cell(0, 10, title); pdf.ln(2)
        pdf.set_font("Arial", "", 11); [pdf.multi_cell(0, 6, ln) for ln in text.splitlines()]
        import io; buf = io.BytesIO(); pdf.output(buf); return buf.getvalue()
    st.download_button("⬇️ Download ToS (PDF)", data=to_pdf_bytes("Terms of Service", tos), file_name="TERMS_OF_SERVICE.pdf", mime="application/pdf")
    st.divider(); st.markdown("### Privacy Policy"); st.markdown(privacy)
    st.download_button("⬇️ Download Privacy (.md)", data=privacy.encode("utf-8"), file_name="PRIVACY_POLICY.md", mime="text/markdown")
    st.download_button("⬇️ Download Privacy (PDF)", data=to_pdf_bytes("Privacy Policy", privacy), file_name="PRIVACY_POLICY.pdf", mime="application/pdf")
    st.divider(); st.markdown("### Risk Disclosure"); st.markdown(risk)
    st.download_button("⬇️ Download Risk (.md)", data=risk.encode("utf-8"), file_name="RISK_DISCLOSURE.md", mime="text/markdown")
    st.download_button("⬇️ Download Risk (PDF)", data=to_pdf_bytes("Risk Disclosure", risk), file_name="RISK_DISCLOSURE.pdf", mime="application/pdf")
