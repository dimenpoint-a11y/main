
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import matplotlib.pyplot as plt
import pytz, re, io, math
from datetime import datetime, date, time as dtime, timedelta
from fpdf import FPDF

APP_VERSION = "v6.5.1"

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

def fmt_int(n):
    if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
        return "—"
    try:
        n = float(n)
    except Exception:
        return "—"
    absn = abs(n)
    if absn >= 1e12: return f"{n/1e12:.2f}T"
    if absn >= 1e9:  return f"{n/1e9:.2f}B"
    if absn >= 1e6:  return f"{n/1e6:.2f}M"
    if absn >= 1e3:  return f"{n/1e3:.2f}K"
    return f"{n:.0f}" if absn >= 1 else f"{n:.4f}"

def fmt_float(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"

def prune_empty_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df2 = df.copy()
    df2 = df2.replace(r'^\s*$', np.nan, regex=True)
    df2 = df2.dropna(axis=1, how='all')
    return df2

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    dm_mode = st.radio("Dark Mode", ["Auto (by clock)", "Light", "Dark"], index=0)
    auto_refresh = st.toggle("Auto-refresh", value=True, help="JS-based reload; no extra packages.")
    interval_s = st.slider("Refresh every (seconds)", 5, 600, 30)
    st.subheader("Watchlist")
    wl_default = "AAPL, MSFT, NVDA, AMZN, META, GOOGL"
    watchlist_raw = st.text_area("Symbols", value=wl_default, height=80)

# ---------- Theme CSS ----------
COMMON_CSS = """
<style>
  .block-container { padding-top: 0.5rem; padding-bottom: 3rem; max-width: 1280px; }
  .title-row { display: flex; align-items: center; gap: .5rem; margin: .25rem 0 .5rem; }
  .title-row.h1 { margin-top: .25rem; }
  .title-row.h1 .emoji { font-size: 1.6rem; }
  .title-row.h2 .emoji { font-size: 1.3rem; }
  .title-row.h3 .emoji { font-size: 1.15rem; }
  .title-row .emoji { display: inline-flex; line-height: 1; vertical-align: middle; }
  .title-row .title-text { line-height: 1.1; }
  .price-big { font-size: 2rem; font-weight: 700; }
  .chg-pos { color: #059669; } /* green */
  .chg-neg { color: #dc2626; } /* red */
  .subgrid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .75rem 2rem; }
  .kv { display:flex; justify-content: space-between; gap: 1rem; }
  .kv .k { color: #6b7280; }
  .kv .v { font-weight: 600; }
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

def get_quote_summary(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = {}
    fast = {}
    try:
        fast = t.fast_info or {}
    except Exception:
        fast = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    last = fast.get("last_price") or info.get("regularMarketPrice")
    prev_close = fast.get("previous_close") or info.get("previousClose")
    open_ = fast.get("open") or info.get("open")
    day_low = fast.get("day_low") or info.get("dayLow")
    day_high = fast.get("day_high") or info.get("dayHigh")
    year_low = fast.get("year_low") or info.get("fiftyTwoWeekLow")
    year_high = fast.get("year_high") or info.get("fiftyTwoWeekHigh")
    volume = fast.get("volume") or info.get("volume")
    avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day") or info.get("averageDailyVolume3Month")
    mcap = fast.get("market_cap") or info.get("marketCap")
    beta = info.get("beta") or info.get("beta3Year") or info.get("beta5Year")
    pe = info.get("trailingPE") or info.get("forwardPE")
    eps = info.get("trailingEps") or info.get("forwardEps")
    ed = info.get("earningsDate")
    if isinstance(ed, (list, tuple)) and len(ed)>0:
        edisplay = pd.to_datetime(ed[0], unit='s', errors='coerce').date() if isinstance(ed[0], (int,float)) else pd.to_datetime(ed[0], errors='coerce').date()
    else:
        edisplay = None
    div_rate = info.get("dividendRate")
    div_yield = info.get("dividendYield")
    if div_rate is not None and div_yield is not None:
        fwd_div = f"{fmt_float(div_rate,2)} ({div_yield*100:.2f}%)"
    elif div_rate is not None:
        fwd_div = f"{fmt_float(div_rate,2)}"
    elif div_yield is not None:
        fwd_div = f"— ({div_yield*100:.2f}%)"
    else:
        fwd_div = "—"
    ex_div_ts = info.get("exDividendDate")
    if ex_div_ts:
        try:
            ex_div = pd.to_datetime(ex_div_ts, unit='s', errors='coerce').date()
        except Exception:
            ex_div = pd.to_datetime(ex_div_ts, errors='coerce')
            ex_div = ex_div.date() if not pd.isna(ex_div) else None
    else:
        ex_div = None
    target = info.get("targetMeanPrice") or info.get("targetMedianPrice")

    name = info.get("shortName") or info.get("longName") or ticker
    currency = fast.get("currency") or info.get("currency") or "USD"
    exch = info.get("exchange") or fast.get("exchange") or info.get("fullExchangeName")
    tz = fast.get("timezone") or info.get("exchangeTimezoneShortName")

    return {
        "name": name, "currency": currency, "exchange": exch, "timezone": tz,
        "last": last, "prev_close": prev_close, "open": open_,
        "day_low": day_low, "day_high": day_high,
        "year_low": year_low, "year_high": year_high,
        "volume": volume, "avg_volume": avg_volume, "mcap": mcap,
        "beta": beta, "pe": pe, "eps": eps, "earnings_date": edisplay,
        "forward_div_yield": fwd_div, "ex_div": ex_div, "target": target
    }

# ---------- Inputs ----------
left, right = st.columns([1,1])
with left:
    ticker = clean_symbol(st.text_input("Ticker", value="AAPL"))
with right:
    start = st.date_input("Start date", pd.to_datetime("2018-01-01"))
    end = st.date_input("End date", pd.Timestamp.today().date())

# Watchlist parse
symbols = [clean_symbol(s) for s in watchlist_raw.replace("\n", ",").split(",")]
symbols = [s for s in symbols if s][:100]

tabs = st.tabs([
    "Price & Indicators", "Signals", "Backtest", "Fundamentals",
    "Earnings & Insider", "Watchlist (Charts)", "Heatmap", "Diagnostics",
    "Legal & Disclosures", "Policies (ToS/Privacy/Risk)", "FAQ"
])

# ---------- Tab 1 — Yahoo-like layout ----------
with tabs[0]:
    header("📊", "Quote — Yahoo-style Overview", level=1)
    st.info("Layout inspired by Yahoo Finance's quote page. Educational tool only; not investment advice.")

    if not ticker:
        st.warning("Enter a valid ticker (A-Z, digits, ., -, _).")
    else:
        # Top summary header
        q = get_quote_summary(ticker)
        if q["last"] is None:
            st.error("Couldn't fetch quote data for this ticker right now.")
        else:
            chg = None; chg_pct = None
            if q["prev_close"] not in (None, 0):
                chg = (q["last"] - q["prev_close"]) if q["last"] is not None else None
                chg_pct = (chg / q["prev_close"] * 100.0) if chg is not None else None
            col1, col2, col3 = st.columns([2,2,2])
            with col1:
                st.markdown(f"### {ticker} — {q['name']}")
                exch_tz = f"{q['exchange'] or ''} {q['timezone'] or ''}".strip()
                if exch_tz: st.caption(exch_tz)
                # Mini sparkline (last 30 closes)
                hist = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
                if not hist.empty:
                    hist = hist.tail(30).reset_index().rename(columns={"Date":"Date"})
                    spark = alt.Chart(hist).mark_area(opacity=0.2).encode(
                        x=alt.X("Date:T", title=None, axis=alt.Axis(labels=False, ticks=False)),
                        y=alt.Y("Close:Q", title=None, axis=alt.Axis(labels=False, ticks=False))
                    )
                    st.altair_chart(spark, use_container_width=True)
            with col2:
                sign = "pos" if (chg or 0) >= 0 else "neg"
                chg_str = f"{'+' if (chg or 0)>=0 else ''}{fmt_float(chg,2)}"
                pct_str = f"{'+' if (chg or 0)>=0 else ''}{fmt_float(chg_pct,2)}%"
                st.markdown(f"<div class='price-big chg-{sign}'>{fmt_float(q['last'],2)} {q['currency']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chg-{sign}'> {chg_str} ({pct_str}) today</div>", unsafe_allow_html=True)
            with col3:
                # Quick actions
                add_watch = st.button("★ Add to Watchlist")
                if add_watch and ticker not in symbols:
                    st.session_state.setdefault("watchlist_extra", set())
                    st.session_state["watchlist_extra"].add(ticker)
                    st.success("Added to session watchlist.")
                # Timeframe selector (Yahoo-like)
                tf = st.radio("Timeframe", ["1D","5D","1M","6M","YTD","1Y","5Y","Max"], horizontal=True, index=5)

            # Determine timeframe range
            end_dt = pd.Timestamp(end)
            if tf == "1D":
                start_dt = end_dt - pd.Timedelta(days=5)  # daily fallback
            elif tf == "5D":
                start_dt = end_dt - pd.Timedelta(days=12)
            elif tf == "1M":
                start_dt = end_dt - pd.Timedelta(days=31)
            elif tf == "6M":
                start_dt = end_dt - pd.Timedelta(days=186)
            elif tf == "YTD":
                start_dt = pd.Timestamp(f"{end_dt.year}-01-01")
            elif tf == "1Y":
                start_dt = end_dt - pd.Timedelta(days=365)
            elif tf == "5Y":
                start_dt = end_dt - pd.Timedelta(days=5*365)
            else:  # Max
                start_dt = pd.Timestamp("1980-01-01")

            # Load prices for chart
            df_raw = load_prices(ticker, start_dt.date(), end_dt.date())
            df = compute_indicators(df_raw).dropna()
            if not df.empty:
                dfx = df.reset_index().rename(columns={"Date":"Date"})
                dfx["Date"] = pd.to_datetime(dfx["Date"])

                st.markdown("#### Chart")
                # Overlays and indicators
                c4, c5, c6, c7 = st.columns([1,1,1,1])
                with c4:
                    show_sma20 = st.checkbox("SMA 20", value=True)
                with c5:
                    show_sma50 = st.checkbox("SMA 50", value=True)
                with c6:
                    show_sma200 = st.checkbox("SMA 200", value=True)
                with c7:
                    show_bb = st.checkbox("Bollinger (20,2)", value=False)

                logy = st.toggle("Log scale", value=False)

                base = alt.Chart(dfx).encode(x=alt.X("Date:T", title="Date"))
                layers = []

                if show_bb:
                    layers.append(base.mark_area(opacity=0.15).encode(
                        y=alt.Y("BB_LO:Q", title="Price (USD)", scale=alt.Scale(type="log") if logy else alt.Scale(type="linear")),
                        y2="BB_UP:Q",
                        tooltip=["Date:T","Close:Q","BB_LO:Q","BB_UP:Q"]
                    ))
                layers.append(base.mark_line().encode(
                    y=alt.Y("Close:Q", title="Price (USD)", scale=alt.Scale(type="log") if logy else alt.Scale(type="linear")),
                    tooltip=["Date:T","Close:Q"]
                ))
                if show_sma20:
                    layers.append(base.mark_line(strokeDash=[4,2]).encode(y="SMA_20:Q", tooltip=["Date:T","SMA_20:Q"]))
                if show_sma50:
                    layers.append(base.mark_line(strokeDash=[4,2]).encode(y="SMA_50:Q", tooltip=["Date:T","SMA_50:Q"]))
                if show_sma200:
                    layers.append(base.mark_line(strokeDash=[6,3]).encode(y="SMA_200:Q", tooltip=["Date:T","SMA_200:Q"]))

                chart_price = alt.layer(*layers).resolve_scale(y='shared').interactive()
                st.altair_chart(chart_price, use_container_width=True)

                with st.expander("Indicators (RSI, MACD, Volume)"):
                    rsi_chart = alt.Chart(dfx).mark_line().encode(
                        x="Date:T", y=alt.Y("RSI_14:Q", title="RSI (0-100)", scale=alt.Scale(domain=[0,100])),
                        tooltip=["Date:T","RSI_14:Q"]
                    ).interactive()
                    macd_chart = alt.layer(
                        alt.Chart(dfx).mark_line().encode(x="Date:T", y="MACD:Q", tooltip=["Date:T","MACD:Q"]),
                        alt.Chart(dfx).mark_line().encode(x="Date:T", y="MACD_signal:Q", tooltip=["Date:T","MACD_signal:Q"]),
                    ).interactive()
                    vol_df = dfx[["Date","Volume"]].dropna()
                    vol_chart = alt.Chart(vol_df).mark_bar().encode(
                        x="Date:T", y=alt.Y("Volume:Q", title="Volume"),
                        tooltip=["Date:T","Volume:Q"]
                    ).interactive()
                    st.altair_chart(rsi_chart, use_container_width=True)
                    st.altair_chart(macd_chart, use_container_width=True)
                    st.altair_chart(vol_chart, use_container_width=True)

            # Summary grid (Yahoo-like)
            st.markdown("#### Summary")
            colA, colB = st.columns(2)
            with colA:
                st.markdown("<div class='subgrid'>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Previous Close</span><span class='v'>{fmt_float(q['prev_close'],2)}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Open</span><span class='v'>{fmt_float(q['open'],2)}</span></div>", unsafe_allow_html=True)
                # Day range bar
                if q['day_low'] is not None and q['day_high'] is not None and q['last'] is not None and (q['day_high']>q['day_low']):
                    low, high, last = float(q['day_low']), float(q['day_high']), float(q['last'])
                    pct = (last-low)/(high-low)
                    pct = min(max(pct,0),1)
                    st.markdown("<div class='kv'><span class='k'>Day's Range</span><span class='v'></span></div>", unsafe_allow_html=True)
                    st.progress(pct, text=f"{fmt_float(low,2)} — {fmt_float(high,2)} · Current: {fmt_float(last,2)}")
                else:
                    st.markdown(f"<div class='kv'><span class='k'>Day's Range</span><span class='v'>—</span></div>", unsafe_allow_html=True)
                # 52-week range progress style
                if q['year_low'] is not None and q['year_high'] is not None and q['last'] is not None and (q['year_high']>q['year_low']):
                    low, high, last = float(q['year_low']), float(q['year_high']), float(q['last'])
                    pct = (last-low)/(high-low)
                    pct = min(max(pct,0),1)
                    st.markdown("<div class='kv'><span class='k'>52-Week Range</span><span class='v'></span></div>", unsafe_allow_html=True)
                    st.progress(pct, text=f"{fmt_float(low,2)} — {fmt_float(high,2)} · Current: {fmt_float(last,2)}")
                else:
                    st.markdown(f"<div class='kv'><span class='k'>52-Week Range</span><span class='v'>—</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Volume</span><span class='v'>{fmt_int(q['volume'])}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Avg Volume</span><span class='v'>{fmt_int(q['avg_volume'])}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with colB:
                st.markdown("<div class='subgrid'>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Market Cap</span><span class='v'>{fmt_int(q['mcap'])}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Beta (5Y)</span><span class='v'>{fmt_float(q['beta'],2)}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>PE Ratio (TTM)</span><span class='v'>{fmt_float(q['pe'],2)}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>EPS (TTM)</span><span class='v'>{fmt_float(q['eps'],2)}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Earnings Date</span><span class='v'>{q['earnings_date'] or '—'}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Forward Dividend & Yield</span><span class='v'>{q['forward_div_yield']}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Ex-Dividend Date</span><span class='v'>{q['ex_div'] or '—'}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>1y Target Est</span><span class='v'>{fmt_float(q['target'],2)}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

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
            score = 50 + sum(parts.values())
            st.metric("Trade Score (0–100, heuristic)", f"{np.clip(score,0,100):.0f}")
            df_parts = pd.DataFrame({"Component": list(parts.keys()), "Contribution": list(parts.values())})
            bar = alt.Chart(df_parts).mark_bar().encode(
                x=alt.X("Component:N", sort=None), y="Contribution:Q",
                tooltip=["Component:N","Contribution:Q"]
            ).interactive()
            st.altair_chart(bar, use_container_width=True)
            st.caption("Positive bars add to the score (bullish), negative bars subtract (bearish). Heuristic only.")

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

            perf = pd.DataFrame({"Date": dfb.index, "Buy&Hold": cum_bh, "SMA(20>50)": cum_sma, "RSI<30 long": cum_rsi}).reset_index(drop=True)
            perf_chart = alt.Chart(perf).transform_fold(
                ["Buy&Hold","SMA(20>50)","RSI<30 long"], as_=["Strategy","Equity"]
            ).mark_line().encode(
                x=alt.X("Date:T"), y=alt.Y("Equity:Q", title="Cumulative (×)"),
                color="Strategy:N", tooltip=["Date:T","Strategy:N","Equity:Q"]
            ).interactive()
            st.altair_chart(perf_chart, use_container_width=True)
            st.caption("Higher equity curve → better cumulative performance. Hover for exact values.")

            def sharpe(series):
                r = series.pct_change().dropna()
                if r.std() == 0 or len(r) < 2: return 0.0
                return (r.mean() / r.std()) * np.sqrt(252)

            met = {
                "CAGR (Buy&Hold)": float((perf["Buy&Hold"].iloc[-1])**(252/len(perf)) - 1) if len(perf) > 0 else 0.0,
                "Sharpe (Buy&Hold)": float(sharpe(perf["Buy&Hold"])),
                "Sharpe (SMA)": float(sharpe(perf["SMA(20>50)"])),
                "Sharpe (RSI)": float(sharpe(perf["RSI<30 long"])),
                "Max DD (Buy&Hold)": float((perf["Buy&Hold"]/perf["Buy&Hold"].cummax()-1).min()),
            }
            met_df = pd.DataFrame({"Metric": list(met.keys()), "Value": list(met.values())})
            met_chart = alt.Chart(met_df).mark_bar().encode(x="Metric:N", y="Value:Q", tooltip=["Metric:N","Value:Q"]).interactive()
            st.altair_chart(met_chart, use_container_width=True)

            buf = io.StringIO(); perf.to_csv(buf, index=False)
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
        st.caption("Lower P/E/PEG can imply cheaper valuation (context matters). Higher ROE/margins generally better. Lower Debt/Equity usually safer.")
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
                    dfe = prune_empty_cols(dfe)
                    st.dataframe(dfe, use_container_width=True)
                    try:
                        dcols = dfe.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]'])
                        if dcols.shape[1] > 0:
                            nd = pd.to_datetime(dcols.iloc[:,0], errors='coerce').min()
                            if pd.notna(nd):
                                days = (nd.date() - pd.Timestamp.today().date()).days
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
                    ins2 = prune_empty_cols(ins2)
                    if not ins2.empty and ("Transaction" in ins2.columns) and ("Value" in ins2.columns):
                        buys = float(ins2[ins2["Transaction"].astype(str).str.contains("Buy", case=False, na=False)]["Value"].fillna(0).sum())
                        sells = float(ins2[ins2["Transaction"].astype(str).str.contains("Sell", case=False, na=False)]["Value"].fillna(0).sum())
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
    st.info("Use your watchlist at left to populate charts and sector heatmap.")

# ---------- Tab 7 (Meaningful Heatmap) ----------
with tabs[6]:
    header("🗺️", "Sector Heatmap — Breadth & Momentum", level=2)
    st.info("Heatmap uses breadth signals per sector. (Optional: wire to earlier sector scoring logic.)")

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
    header("📜", "Policies", level=2)
    today = date.today().isoformat()
    tos = f"""# Terms of Service (Short)
_Last updated: {today}_

- This app provides **general, educational information** only. **Not investment advice**.
- We **do not** execute trades or handle your funds.
- Data may be **delayed, incomplete, or inaccurate**.
- We **do not guarantee** performance or outcomes. You assume all risk.
- Our **liability is limited** to the maximum extent permitted by law.
- Features may change, break, or be removed without notice.
- For feedback or issues, please use the **website contact form** or the in-app feedback link.
- By using this app, **you agree to these terms**.
"""
    privacy = "# Privacy Policy\n\nMinimal collection; no sale of data; cookies only for preferences; yfinance as data source."
    risk = "# Risk Disclosure\n\nTrading involves risk; backtests are hypothetical; past performance is not indicative of future results."

    st.markdown("### Terms of Service"); st.markdown(tos)
    st.download_button("⬇️ Download ToS (.md)", data=tos.encode("utf-8"), file_name="TERMS_OF_SERVICE.md", mime="text/markdown")
    st.download_button("⬇️ Download ToS (PDF)", data=to_pdf_bytes("Terms of Service (Short)", tos), file_name="TERMS_OF_SERVICE.pdf", mime="application/pdf")
    st.divider()
    st.markdown("### Privacy Policy"); st.markdown(privacy)
    st.download_button("⬇️ Download Privacy (.md)", data=privacy.encode("utf-8"), file_name="PRIVACY_POLICY.md", mime="text/markdown")
    st.download_button("⬇️ Download Privacy (PDF)", data=to_pdf_bytes("Privacy Policy", privacy), file_name="PRIVACY_POLICY.pdf", mime="application/pdf")
    st.divider()
    st.markdown("### Risk Disclosure"); st.markdown(risk)
    st.download_button("⬇️ Download Risk (.md)", data=risk.encode("utf-8"), file_name="RISK_DISCLOSURE.md", mime="text/markdown")
    st.download_button("⬇️ Download Risk (PDF)", data=to_pdf_bytes("Risk Disclosure", risk), file_name="RISK_DISCLOSURE.pdf", mime="application/pdf")

# ---------- Tab 11 (FAQ) ----------
with tabs[10]:
    header("❓", "FAQ — What the metrics mean", level=2)
    st.markdown("""
**Price above/below SMA200** — Above: longer-term uptrend (generally better for longs). Below: downtrend (riskier for longs).  
**RSI (14)** — Momentum oscillator 0–100. >70 often overbought (watch for pullbacks). <30 oversold (watch for bounces).  
**MACD / Signal** — Trend momentum. MACD above Signal = bullish bias; below = bearish. Widening gap = strengthening trend.  
**Bollinger Bands (20,2)** — Volatility envelope around a 20‑day average. Touching upper band can indicate strength; lower band, weakness.  
**ATR (14)** — Average range in price per day. Higher = more volatility (bigger moves and risk).  
**Trade Score (0–100)** — Heuristic composite from RSI/MACD/trend/squeeze. Higher ≈ more bullish alignment. Not a guarantee.  
**52‑Week Range** — Where the current price sits between last year’s low and high (closer to high can signal strength, context matters).  
**Backtest metrics** — **CAGR** (higher better), **Sharpe** (risk‑adjusted return; higher better), **Max Drawdown** (more negative worse).  
**Fundamentals** — **P/E, PEG** (lower can be better), **ROE/Margins** (higher better), **Debt/Equity** (lower often safer).  
""")
    st.info("All signals are educational only. Past performance is not indicative of future results.")
