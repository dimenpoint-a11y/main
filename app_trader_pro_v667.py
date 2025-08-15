
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import matplotlib.pyplot as plt
import pytz, re, io, math, traceback, unicodedata, sys
from datetime import datetime, date, time as dtime
from fpdf import FPDF

APP_VERSION = "v6.6.7"

st.set_page_config(page_title="Trader Dashboard - Pro", layout="wide")

# ---------- Altair setup ----------
alt.data_transformers.disable_max_rows()
try:
    alt.theme.enable("opaque")
except Exception:
    pass

# ---------- Helpers ----------
_ASCII_MAP = {"≠":"!=", "≤":"<=", "≥":">=", "×":"x", "—":"-", "–":"-","“":'"', "”":'"', "‘":"'", "’":"'", "•":"-", "…":"..."}
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def sanitize_text_ascii(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    for k,v in _ASCII_MAP.items(): s = s.replace(k,v)
    s = _EMOJI_RE.sub("", s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("latin-1", "ignore").decode("latin-1")
    return s

def to_pdf_bytes(title: str, text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16); pdf.multi_cell(0, 10, sanitize_text_ascii(title)); pdf.ln(2)
    pdf.set_font("Arial", "", 11)
    clean = sanitize_text_ascii(re.sub(r"[#*_`>]+", "", text))
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

# ---------- Theme & Altair helpers ----------
def palette(is_dark: bool):
    if is_dark:
        return {
            "bg": "transparent",
            "fg": "#e5e7eb",      # slate-200
            "muted": "#cbd5e1",   # slate-300
            "grid": "#334155",    # slate-700
            "pos": "#22c55e",     # green-500
            "neg": "#ef4444",     # red-500
            "mid": "#94a3b8"      # slate-400
        }
    else:
        return {
            "bg": "transparent",
            "fg": "#111827",      # gray-900
            "muted": "#6b7280",   # gray-500
            "grid": "#e5e7eb",    # gray-200
            "pos": "#16a34a",     # green-600
            "neg": "#dc2626",     # red-600
            "mid": "#64748b"      # slate-500
        }

def themed(chart: alt.Chart, is_dark: bool) -> alt.Chart:
    p = palette(is_dark)
    return (chart
        .configure(background=p["bg"])
        .configure_view(strokeOpacity=0, fillOpacity=0)
        .configure_axis(
            labelColor=p["fg"], titleColor=p["fg"],
            gridColor=p["grid"], domainColor=p["grid"],
            tickColor=p["grid"]
        )
        .configure_legend(labelColor=p["fg"], titleColor=p["fg"])
    )

def safe_altair_chart(chart, is_dark: bool, **kwargs):
    try:
        st.altair_chart(themed(chart, is_dark), **kwargs)
        return True
    except Exception as e:
        st.warning(f"Chart engine fallback: {e}")
        return False

def st_line_chart_fallback(df: pd.DataFrame, ycols, xcol=None):
    data = df.copy()
    if xcol and xcol in data.columns: data = data.set_index(xcol)
    try: st.line_chart(data[ycols])
    except Exception as e: st.error(f"Fallback chart failed: {e}")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    dm_mode = st.radio("Color mode", ["Auto (by clock)", "Light", "Dark"], index=0, horizontal=True)
    auto_refresh = st.toggle("Auto-refresh", value=False, help="JS-based reload; no extra packages.")
    interval_s = st.slider("Refresh every (seconds)", 10, 600, 30)
    st.subheader("Watchlist")
    wl_default = "AAPL, MSFT, NVDA, AMZN, META, GOOGL"
    watchlist_raw = st.text_area("Symbols", value=wl_default, height=90)

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
  .chg-pos { color: #22c55e; }
  .chg-neg { color: #ef4444; }
  .subgrid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .75rem 2rem; }
  .kv { display:flex; justify-content: space-between; gap: 1rem; }
  .kv .k { opacity:.85; }
  .kv .v { font-weight: 600; }
</style>
"""
st.markdown(COMMON_CSS, unsafe_allow_html=True)

auto_dark = (dm_mode == "Auto (by clock)" and is_night_chicago()) or (dm_mode == "Dark")
is_dark = bool(auto_dark)

if is_dark:
    st.markdown("<style>.appview-container, .main { background-color: #0b1020; color: #e5e7eb; }</style>", unsafe_allow_html=True)
    plt.style.use('dark_background')
else:
    st.markdown("<style>.appview-container, .main { background-color: #ffffff; color: #0f172a; }</style>", unsafe_allow_html=True)
    plt.style.use('default')

# ---------- Auto-refresh ----------
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
    if data is None or data.empty: return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    ren = {c:str(c).title() for c in data.columns}
    data = data.rename(columns=ren).reset_index()
    if "Date" not in data.columns:
        dcols = [c for c in data.columns if pd.api.types.is_datetime64_any_dtype(data[c])]
        if dcols: data.rename(columns={dcols[0]:"Date"}, inplace=True)
        else: data.insert(0,"Date", pd.to_datetime(data.index))
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    if "Close" not in data.columns:
        for altc in ["Adj Close","AdjClose","Adj_Close","close"]:
            if altc in data.columns: data.rename(columns={altc:"Close"}, inplace=True); break
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
    if not set(["High","Low","Close"]).issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    h_l = df["High"] - df["Low"]
    h_pc = (df["High"] - df["Close"].shift()).abs()
    l_pc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_indicators(df_in: pd.DataFrame):
    if df_in is None or df_in.empty: return pd.DataFrame()
    df = df_in.copy()
    if "Close" not in df.columns: return df
    out = df.copy()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["SMA_200"] = out["Close"].rolling(200).mean()
    out["Returns"] = out["Close"].pct_change()
    out["Volatility_20d"] = out["Returns"].rolling(20).std() * np.sqrt(252)
    out["RSI_14"] = rsi(out["Close"], 14)
    macd_line, signal_line, hist = macd(out["Close"], 12, 26, 9)
    out["MACD"] = macd_line; out["MACD_signal"] = signal_line; out["MACD_hist"] = hist
    bb_ma, bb_up, bb_lo = bollinger(out["Close"], 20, 2.0)
    out["BB_MA"] = bb_ma; out["BB_UP"] = bb_up; out["BB_LO"] = bb_lo
    out["ATR_14"] = atr(out, 14)
    out["BandWidth"] = (out["BB_UP"] - out["BB_LO"]) / out["BB_MA"]
    out["Above_SMA200"] = out["Close"] > out["SMA_200"]
    # For volume coloring vs previous close
    out["PrevClose"] = out["Close"].shift(1)
    out["UpPrev"] = out["Close"] >= out["PrevClose"]
    return out

def trade_score_from_row(row):
    score = 50.0
    rsi_v = row.get("RSI_14", np.nan)
    if not np.isnan(rsi_v):
        if rsi_v < 30: score += 10
        elif rsi_v > 70: score -= 10
    macd_v, sig_v = row.get("MACD", np.nan), row.get("MACD_signal", np.nan)
    if not np.isnan(macd_v) and not np.isnan(sig_v):
        score += 10 if macd_v > sig_v else -5
    close_v, sma200_v = row.get("Close", np.nan), row.get("SMA_200", np.nan)
    if not np.isnan(close_v) and not np.isnan(sma200_v):
        score += 10 if close_v > sma200_v else -10
    bw = row.get("BandWidth", np.nan)
    if not np.isnan(bw) and bw < 0.05: score += 5
    return float(np.clip(score, 0, 100))

@st.cache_data(show_spinner=False, ttl=120)
def get_quote_summary(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info, fast = {}, {}
    try: fast = t.fast_info or {}
    except Exception: fast = {}
    try: info = t.info or {}
    except Exception: info = {}
    def pick(*keys, srcs=(fast, info)):
        for k in keys:
            for s in srcs:
                if s and k in s and s[k] not in (None, "", 0):
                    return s[k]
        return None
    last = pick("last_price", "regularMarketPrice")
    prev_close = pick("previous_close", "previousClose")
    open_ = pick("open")
    day_low = pick("day_low", "dayLow")
    day_high = pick("day_high", "dayHigh")
    year_low = pick("year_low", "fiftyTwoWeekLow")
    year_high = pick("year_high", "fiftyTwoWeekHigh")
    volume = pick("volume")
    avg_volume = pick("averageVolume", "averageDailyVolume10Day", "averageDailyVolume3Month", srcs=(info,))
    mcap = pick("market_cap", "marketCap")
    beta = pick("beta", "beta3Year", "beta5Year", srcs=(info,))
    pe = pick("trailingPE", "forwardPE", srcs=(info,))
    eps = pick("trailingEps", "forwardEps", srcs=(info,))
    ed = info.get("earningsDate")
    if isinstance(ed, (list, tuple)) and len(ed)>0:
        edisplay = pd.to_datetime(ed[0], unit='s', errors='coerce').date() if isinstance(ed[0], (int,float)) else pd.to_datetime(ed[0], errors='coerce').date()
    else:
        edisplay = None
    div_rate = info.get("dividendRate"); div_yield = info.get("dividendYield")
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
        try: ex_div = pd.to_datetime(ex_div_ts, unit='s', errors='coerce').date()
        except Exception:
            ex_div = pd.to_datetime(ex_div_ts, errors='coerce'); ex_div = ex_div.date() if not pd.isna(ex_div) else None
    else:
        ex_div = None
    target = pick("targetMeanPrice", "targetMedianPrice", srcs=(info,))

    name = pick("shortName", "longName", srcs=(info,)) or ticker
    currency = pick("currency", srcs=(fast, info)) or "USD"
    exch = pick("exchange", "fullExchangeName", srcs=(info, fast))
    tz = pick("timezone", "exchangeTimezoneShortName", srcs=(fast, info))

    return {"name": name, "currency": currency, "exchange": exch, "timezone": tz,
            "last": last, "prev_close": prev_close, "open": open_,
            "day_low": day_low, "day_high": day_high,
            "year_low": year_low, "year_high": year_high,
            "volume": volume, "avg_volume": avg_volume, "mcap": mcap,
            "beta": beta, "pe": pe, "eps": eps, "earnings_date": edisplay,
            "forward_div_yield": fwd_div, "ex_div": ex_div, "target": target}

@st.cache_data(show_spinner=False, ttl=600)
def get_sector_mcap(symbol: str):
    try:
        info = yf.Ticker(symbol).info or {}
        sector = info.get("sector") or "Other"
        mcap = info.get("marketCap")
        return sector, mcap
    except Exception:
        return "Other", None

def get_eps_series_quarterly(ticker: str):
    try:
        t = yf.Ticker(ticker)
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = qf.index.astype(str).str.lower()
            choices = ["diluted eps","dilutedeps","diluted eps (usd)","basic eps","basiceps","basic eps (usd)","eps"]
            row = None
            for ch in choices:
                hit = idx.str.contains(ch)
                if hit.any():
                    row = qf[hit].iloc[0]; break
            if row is not None:
                s = row.copy(); s.index = pd.to_datetime(s.index, errors="coerce")
                s = s.dropna().sort_index(); s.name = "EPS"; return s
        qe = getattr(t, "quarterly_earnings", None)
        if isinstance(qe, pd.DataFrame) and not qf.empty:
            cols = [c for c in qe.columns if "eps" in str(c).lower()]
            if cols:
                s = qe[cols[0]]; s.index = pd.to_datetime(qe.index, errors="coerce")
                s = s.dropna().sort_index(); s.name = "EPS"; return s
    except Exception:
        pass
    return None

# ---------- Inputs ----------
left, right = st.columns([1,1])
with left:
    ticker = clean_symbol(st.text_input("Ticker", value="AAPL"))
with right:
    start = st.date_input("Start date", pd.to_datetime("2018-01-01"))
    end = st.date_input("End date", pd.Timestamp.today().date())

# Watchlist parse
symbols = [clean_symbol(s) for s in watchlist_raw.replace("\n", ",").split(",")]
symbols = [s for s in symbols if s][:60]
if "watchlist_extra" not in st.session_state: st.session_state["watchlist_extra"] = []
extra = [s for s in st.session_state["watchlist_extra"] if s]
symbols = list(dict.fromkeys(symbols + extra))[:60]

tabs = st.tabs(["Price & Indicators", "Backtest", "Earnings & Insider", "Watchlist", "Heatmap", "Diagnostics", "Policies", "Learn"])

# ---------- Tab 1 — Price & Indicators ----------
with tabs[0]:
    header("📊", "Quote — Yahoo-style Overview", level=1)
    st.info("Hover for values • Scroll/drag to zoom • Double-click to reset. Educational only, not advice.")
    if not ticker:
        st.warning("Enter a valid ticker (A-Z, digits, ., -, _).")
    else:
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

            with col2:
                sign = "pos" if (chg or 0) >= 0 else "neg"
                chg_str = f"{'+' if (chg or 0)>=0 else ''}{fmt_float(chg,2)}"
                pct_str = f"{'+' if (chg or 0)>=0 else ''}{fmt_float(chg_pct,2)}%"
                st.markdown(f"<div class='price-big chg-{sign}'>{fmt_float(q['last'],2)} {q['currency']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chg-{sign}'> {chg_str} ({pct_str}) today</div>", unsafe_allow_html=True)

            with col3:
                add_watch = st.button("★ Add to Watchlist")
                if add_watch and ticker not in symbols:
                    if ticker not in st.session_state["watchlist_extra"]:
                        st.session_state["watchlist_extra"].append(ticker)
                    st.success("Added to session watchlist.")
                tf = st.radio("Timeframe", ["1D","5D","1M","6M","YTD","1Y","5Y","Max"], horizontal=True, index=5)
                style = st.radio("Chart style", ["Candles","Close line"], horizontal=True, index=0)
                if st.button("Reset zoom"): st.experimental_rerun()

            # Determine timeframe range
            end_dt = pd.Timestamp(end)
            if tf == "1D":  start_dt = end_dt - pd.Timedelta(days=5)
            elif tf == "5D": start_dt = end_dt - pd.Timedelta(days=12)
            elif tf == "1M": start_dt = end_dt - pd.Timedelta(days=31)
            elif tf == "6M": start_dt = end_dt - pd.Timedelta(days=186)
            elif tf == "YTD": start_dt = pd.Timestamp(f"{end_dt.year}-01-01")
            elif tf == "1Y":  start_dt = end_dt - pd.Timedelta(days=365)
            elif tf == "5Y":  start_dt = end_dt - pd.Timedelta(days=5*365)
            else:            start_dt = pd.Timestamp("1980-01-01")

            df_raw = load_prices(ticker, start_dt.date(), end_dt.date())
            dfx = compute_indicators(df_raw)
            if dfx.empty or "Close" not in dfx.columns:
                st.warning("No price data available for the selected range.")
            else:
                if "Date" not in dfx.columns and "Date" in df_raw.columns: dfx["Date"] = pd.to_datetime(df_raw["Date"])
                elif "Date" not in dfx.columns: dfx.insert(0, "Date", pd.to_datetime(dfx.index))

                # --- Main chart (SINGLE; removed sparkline duplicate) ---
                st.markdown("#### Price & Indicators")
                c4, c5, c6, c7 = st.columns([1,1,1,1])
                with c4: show_sma20 = st.checkbox("SMA 20", value=True)
                with c5: show_sma50 = st.checkbox("SMA 50", value=True)
                with c6: show_sma200 = st.checkbox("SMA 200", value=True)
                with c7: show_bb = st.checkbox("Bollinger (20,2)", value=False)
                logy = st.toggle("Log scale", value=False)

                zoom = alt.selection_interval(bind='scales', encodings=['x'])
                hover = alt.selection_point(nearest=True, fields=["Date"], on="mouseover", empty=False)
                p = palette(is_dark)
                base = alt.Chart(dfx).encode(x=alt.X("Date:T", title="Date"))

                layers = []
                if style == "Close line":
                    layers.append(base.mark_line(color=p["mid"]).encode(
                        y=alt.Y("Close:Q", title="Price", scale=alt.Scale(type="log") if logy else alt.Scale(type="linear")),
                        tooltip=["Date:T","Open:Q","High:Q","Low:Q","Close:Q","Volume:Q"]
                    ).add_params(zoom))
                    layers.append(base.mark_point(size=20, opacity=0, color=p["mid"]).encode(y="Close:Q").add_params(hover))
                    layers.append(alt.Chart(dfx).mark_rule(color=p["grid"]).encode(x="Date:T").transform_filter(hover))
                else:
                    # Candlestick
                    rule = base.mark_rule().encode(
                        y="Low:Q", y2="High:Q",
                        color=alt.condition(alt.datum.Close >= alt.datum.Open, alt.value(p["pos"]), alt.value(p["neg"])),
                        tooltip=["Date:T","Open:Q","High:Q","Low:Q","Close:Q","Volume:Q"]
                    )
                    bar = base.mark_bar().encode(
                        y=alt.Y("Open:Q", title="Price", scale=alt.Scale(type="log") if logy else alt.Scale(type="linear")),
                        y2="Close:Q",
                        color=alt.condition(alt.datum.Close >= alt.datum.Open, alt.value(p["pos"]), alt.value(p["neg"]))
                    )
                    layers.extend([rule.add_params(zoom), bar.add_params(zoom)])

                if show_bb and "BB_LO" in dfx.columns and "BB_UP" in dfx.columns:
                    layers.append(base.mark_area(opacity=0.15, color=p["mid"]).encode(
                        y=alt.Y("BB_LO:Q", title="Price", scale=alt.Scale(type="log") if logy else alt.Scale(type="linear")),
                        y2="BB_UP:Q",
                        tooltip=["Date:T","BB_LO:Q","BB_UP:Q"]
                    ).add_params(zoom))

                if show_sma20 and "SMA_20" in dfx.columns:
                    layers.append(base.mark_line(strokeDash=[4,2], color="#60a5fa").encode(y="SMA_20:Q", tooltip=["Date:T","SMA_20:Q"]).add_params(zoom))
                if show_sma50 and "SMA_50" in dfx.columns:
                    layers.append(base.mark_line(strokeDash=[4,2], color="#f59e0b").encode(y="SMA_50:Q", tooltip=["Date:T","SMA_50:Q"]).add_params(zoom))
                if show_sma200 and "SMA_200" in dfx.columns:
                    layers.append(base.mark_line(strokeDash=[6,3], color="#a78bfa").encode(y="SMA_200:Q", tooltip=["Date:T","SMA_200:Q"]).add_params(zoom))

                price_layer = alt.layer(*layers).resolve_scale(y='shared').properties(height=360)

                # Volume (colored by up vs previous close)
                vol = alt.Chart(dfx).mark_bar().encode(
                    x="Date:T",
                    y=alt.Y("Volume:Q", title="Volume"),
                    color=alt.condition(alt.datum.UpPrev, alt.value(p["pos"]), alt.value(p["neg"])),
                    tooltip=["Date:T","Volume:Q"]
                ).properties(height=80)

                chart_combo = alt.vconcat(price_layer, vol).resolve_legend(color="independent")
                ok = safe_altair_chart(chart_combo, is_dark, use_container_width=True)
                if not ok: st_line_chart_fallback(dfx, ["Close","SMA_20","SMA_50","SMA_200"], xcol="Date")
                st.caption("Candles: green up / red down; wicks show High–Low. **Higher price is better** for longs. Volume bars are green/red vs **previous close**.")

                # --- Merged Signals (kept below main chart)
                latest = dfx.iloc[-1]
                parts = {
                    "RSI": (10 if latest.get("RSI_14", np.nan) < 30 else (-10 if latest.get("RSI_14", np.nan) > 70 else 0)),
                    "MACD": (10 if latest.get("MACD", np.nan) > latest.get("MACD_signal", np.nan) else -5),
                    "Trend vs SMA200": (10 if latest.get("Close", np.nan) > latest.get("SMA_200", np.nan) else -10),
                    "Squeeze": (5 if latest.get("BandWidth", np.nan) < 0.05 else 0),
                }
                score = 50 + sum(parts.values())
                st.subheader("Signals (merged)")
                st.metric("Trade Score (0–100, heuristic)", f"{np.clip(score,0,100):.0f}")
                df_parts = pd.DataFrame({"Component": list(parts.keys()), "Contribution": list(parts.values())})
                bar = alt.Chart(df_parts).mark_bar(color="#3b82f6").encode(
                    x=alt.X("Component:N", sort=None, title=None), y=alt.Y("Contribution:Q"),
                    tooltip=["Component:N","Contribution:Q"]
                )
                ok = safe_altair_chart(bar, is_dark, use_container_width=True)
                if not ok: st_line_chart_fallback(df_parts.reset_index(), ["Contribution"], xcol="Component")
                st.caption("Signals: **positive** bars suggest bullish alignment; **negative** bars suggest bearish tilt. Heuristic only.")

            # Summary grid
            st.markdown("#### Summary")
            colA, colB = st.columns(2)
            with colA:
                st.markdown("<div class='subgrid'>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Previous Close</span><span class='v'>{fmt_float(q['prev_close'],2)}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'><span class='k'>Open</span><span class='v'>{fmt_float(q['open'],2)}</span></div>", unsafe_allow_html=True)
                if q['day_low'] is not None and q['day_high'] is not None and q['last'] is not None and (q['day_high']>q['day_low']):
                    low, high, last = float(q['day_low']), float(q['day_high']), float(q['last'])
                    pct = min(max((last-low)/(high-low),0),1)
                    st.markdown("<div class='kv'><span class='k'>Day's Range</span><span class='v'></span></div>", unsafe_allow_html=True)
                    st.progress(pct, text=f"{fmt_float(low,2)} — {fmt_float(high,2)} · Current: {fmt_float(last,2)}")
                else:
                    st.markdown(f"<div class='kv'><span class='k'>Day's Range</span><span class='v'>—</span></div>", unsafe_allow_html=True)
                if q['year_low'] is not None and q['year_high'] is not None and q['last'] is not None and (q['year_high']>q['year_low']):
                    low, high, last = float(q['year_low']), float(q['year_high']), float(q['last'])
                    pct = min(max((last-low)/(high-low),0),1)
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

# ---------- Tab 2 — Backtest ----------
with tabs[1]:
    header("🧪", "Backtest", level=2)
    if not ticker: st.info("Load price data first.")
    else:
        dfr = load_prices(ticker, start, end)
        if dfr.empty: st.info("No data.")
        else:
            st.subheader("Quick Strategy Backtests (toy models)")
            dfb = compute_indicators(dfr)
            dfb["pos_sma"] = np.where(dfb["SMA_20"] > dfb["SMA_50"], 1, 0)
            dfb["ret_sma"] = dfb["pos_sma"].shift() * dfb["Returns"]
            cum_sma = (1 + dfb["ret_sma"].fillna(0)).cumprod()
            dfb["pos_rsi"] = np.where(dfb["RSI_14"] < 30, 1, np.where(dfb["RSI_14"] > 70, 0, np.nan))
            dfb["pos_rsi"] = dfb["pos_rsi"].ffill().fillna(0)
            dfb["ret_rsi"] = dfb["pos_rsi"].shift() * dfb["Returns"]
            cum_rsi = (1 + dfb["ret_rsi"].fillna(0)).cumprod()
            cum_bh = (1 + dfb["Returns"].fillna(0)).cumprod()
            perf = pd.DataFrame({"Date": dfb["Date"], "Buy&Hold": cum_bh, "SMA(20>50)": cum_sma, "RSI<30 long": cum_rsi}).dropna(subset=["Date"]).reset_index(drop=True)
            chart = alt.Chart(perf).transform_fold(["Buy&Hold","SMA(20>50)","RSI<30 long"], as_=["Strategy","Equity"]).mark_line().encode(
                x=alt.X("Date:T"), y=alt.Y("Equity:Q", title="Cumulative (×)"),
                color="Strategy:N", tooltip=["Date:T","Strategy:N","Equity:Q"]
            ).interactive()
            ok = safe_altair_chart(chart, is_dark, use_container_width=True)
            if not ok: st_line_chart_fallback(perf, ["Buy&Hold","SMA(20>50)","RSI<30 long"], xcol="Date")
            st.caption("Equity curve: **higher is better**.")

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
            ok = safe_altair_chart(alt.Chart(met_df).mark_bar(color="#06b6d4").encode(x="Metric:N", y="Value:Q", tooltip=["Metric","Value"]).interactive(), is_dark, use_container_width=True)
            if not ok: st_line_chart_fallback(met_df, ["Value"], xcol="Metric")
            st.caption("Sharpe: **higher is better**; Max Drawdown: **less negative is better**.")

            buf = io.StringIO(); perf.to_csv(buf, index=False)
            st.download_button("⬇️ Download equity curves (CSV)", data=buf.getvalue(), file_name=f"{ticker}_backtest_equity.csv", mime="text/csv")

# ---------- Tab 3 — Earnings & Insider ----------
with tabs[2]:
    header("📅", "Earnings & Insider", level=2)
    c1, c2 = st.columns(2)
    with c1:
        header("📈", "Quarterly EPS", level=3)
        eps = get_eps_series_quarterly(ticker) if ticker else None
        if isinstance(eps, pd.Series) and len(eps) > 0:
            df_eps = eps.reset_index(); df_eps.columns = ["Date", "EPS"]
            df_eps["EPS_MA4"] = df_eps["EPS"].rolling(4).mean()
            chart = alt.layer(
                alt.Chart(df_eps).mark_line(point=True, color="#22c55e").encode(x=alt.X("Date:T", title="Quarter"), y=alt.Y("EPS:Q", title="EPS (USD)"), tooltip=["Date:T","EPS:Q"]),
                alt.Chart(df_eps).mark_line(strokeDash=[4,2], color="#a78bfa").encode(x="Date:T", y="EPS_MA4:Q", tooltip=["Date:T","EPS_MA4:Q"])
            ).interactive()
            ok = safe_altair_chart(chart, is_dark, use_container_width=True)
            if not ok: st_line_chart_fallback(df_eps, ["EPS","EPS_MA4"], xcol="Date")
            st.caption("EPS: **higher is better**; dashed = 4‑Q MA.")
        else:
            st.info("EPS series not available.")
        header("🗓️", "Earnings Calendar", level=3)
        try:
            if ticker:
                t = yf.Ticker(ticker)
                ed = None
                try: ed = t.earnings_dates
                except Exception: pass
                if ed is None or (hasattr(ed, "empty") and ed.empty):
                    try: ed = t.get_earnings_dates(limit=8)
                    except Exception: ed = None
                if isinstance(ed, pd.DataFrame) and not ed.empty:
                    dfe = prune_empty_cols(ed.reset_index())
                    st.dataframe(dfe, use_container_width=True)
                else:
                    st.info("No earnings dates available.")
        except Exception as e:
            st.error(f"Earnings lookup failed: {e}")
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

# ---------- Tab 4 — Watchlist ----------
with tabs[3]:
    header("👀", "Watchlist", level=2)
    if not symbols: st.info("Add tickers in the sidebar.")
    else:
        cols = st.columns(3)
        for i, sym in enumerate(symbols):
            if not sym: continue
            with cols[i % 3]:
                try:
                    q = get_quote_summary(sym)
                    df = load_prices(sym, pd.Timestamp.today().date()-pd.Timedelta(days=120), pd.Timestamp.today().date())
                    st.markdown(f"**{sym}**")
                    if q["last"] is not None and q["prev_close"] not in (None, 0):
                        chg = q["last"] - q["prev_close"]
                        chg_pct = chg / q["prev_close"] * 100.0
                        color = "#22c55e" if chg >= 0 else "#ef4444"
                        st.markdown(f"<div style='color:{color}'>{fmt_float(q['last'],2)} ({'+' if chg>=0 else ''}{fmt_float(chg_pct,2)}%)</div>", unsafe_allow_html=True)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # no sparkline here to avoid duplication; small bar chart instead
                        sp = alt.Chart(df.tail(60)).mark_line(color=palette(is_dark)["mid"]).encode(
                            x=alt.X("Date:T", axis=alt.Axis(labels=False, ticks=False, title=None)),
                            y=alt.Y("Close:Q", axis=alt.Axis(labels=False, ticks=False, title=None)),
                            tooltip=["Date:T","Close:Q"]
                        )
                        safe_altair_chart(sp, is_dark, use_container_width=True)
                except Exception as e:
                    st.caption(f"{sym}: {e}")

# ---------- Tab 5 — Heatmap (Enhanced & fixed sort) ----------
with tabs[4]:
    header("🗺️", "Heatmap — Multi-timeframe & Sector grouping", level=2)
    colh1, colh2, colh3 = st.columns([1,1,1])
    with colh1:
        metrics = st.multiselect("Metrics", ["1D %","1W %","1M %","3M %","YTD %"], default=["1D %","1M %","3M %","YTD %"])
    with colh2:
        group_by_sector = st.checkbox("Group by Sector", value=True)
    with colh3:
        show_volume_bubbles = st.checkbox("Show volume bubbles", value=False, help="Size by 30‑day avg volume")

    if not symbols:
        st.info("Add tickers in the sidebar.")
    else:
        try:
            data = yf.download(" ".join(symbols), period="1y", interval="1d", auto_adjust=True, progress=False, group_by='ticker', threads=True)
        except Exception:
            data = None

        rows = []
        for sym in symbols:
            try:
                if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex) and sym in data.columns.get_level_values(0):
                    d = data[sym].copy()
                else:
                    d = yf.download(sym, period="1y", interval="1d", auto_adjust=True, progress=False)
                if d is None or d.empty:
                    continue
                d = d.dropna()
                close = d["Close"]
                vals = {}
                vals["1D %"] = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close)>=2 else np.nan
                vals["1W %"] = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close)>=6 else np.nan
                vals["1M %"] = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close)>=22 else np.nan
                vals["3M %"] = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close)>=64 else np.nan
                first_this_year = close[close.index >= pd.Timestamp(f"{pd.Timestamp.today().year}-01-01")]
                vals["YTD %"] = (close.iloc[-1] / first_this_year.iloc[0] - 1) * 100 if len(first_this_year)>=1 else np.nan
                sector, mcap = get_sector_mcap(sym)
                avg_vol = float(d["Volume"].tail(30).mean()) if "Volume" in d.columns else np.nan
                rows.append({"Ticker": sym, **vals, "Sector": sector, "MktCap": mcap, "Last": float(close.iloc[-1]), "AvgVol30": avg_vol})
            except Exception:
                continue

        dfh = pd.DataFrame(rows)
        if dfh.empty:
            st.info("No data for heatmap.")
        else:
            if not metrics: metrics = ["1D %","1M %","3M %","YTD %"]
            long = dfh.melt(id_vars=["Ticker","Sector","MktCap","Last","AvgVol30"], value_vars=metrics, var_name="Metric", value_name="Pct")
            long = long.dropna(subset=["Pct"])
            if long.empty:
                st.info("Not enough data to render heatmap.")
            else:
                # Compute sector order by average Pct across selected metrics (descending)
                sector_order = long.groupby("Sector")["Pct"].mean().sort_values(ascending=False).index.tolist()
                max_abs = float(np.nanmax(np.abs(long["Pct"].values))) if not long["Pct"].isna().all() else 1.0
                dom = [-max_abs, 0, max_abs]
                base = alt.Chart(long)

                heat = base.mark_rect().encode(
                    x=alt.X("Ticker:N", sort=sorted(dfh["Ticker"].unique().tolist()), title=None),
                    y=alt.Y("Metric:N", title=None),
                    color=alt.Color("Pct:Q", scale=alt.Scale(scheme='redblue', domain=dom, domainMid=0)),
                    tooltip=["Ticker:N","Metric:N", alt.Tooltip("Pct:Q", format=".2f"), alt.Tooltip("Last:Q", title="Price", format=".2f"), alt.Tooltip("MktCap:Q", format="~s"), "Sector:N"]
                )

                if show_volume_bubbles:
                    bubbles = base.mark_circle(opacity=0.4, color="#f59e0b").encode(
                        x="Ticker:N", y="Metric:N",
                        size=alt.Size("AvgVol30:Q", scale=alt.Scale(range=[10, 800]), title="AvgVol30"),
                        tooltip=[alt.Tooltip("AvgVol30:Q", format=",.0f")]
                    )
                    heat_layer = heat + bubbles
                else:
                    heat_layer = heat

                if group_by_sector:
                    chart = heat_layer.facet(
                        row=alt.Row("Sector:N", sort=sector_order)
                    ).resolve_scale(color="shared")
                else:
                    chart = heat_layer.properties(height=240)

                ok = safe_altair_chart(chart, is_dark, use_container_width=True)
                if not ok: st.dataframe(dfh.set_index("Ticker")[metrics], use_container_width=True)
                st.caption("Heatmap: **blue = positive**, **red = negative** relative returns. Optional bubbles show **30‑day avg volume** (larger = higher activity).")

# ---------- Tab 6 — Diagnostics ----------
with tabs[5]:
    header("🔧", "Diagnostics", level=2)
    st.write(f"App version: {APP_VERSION} | Python: {sys.version.split()[0]} | Altair: {alt.__version__}")
    try:
        if not ticker:
            st.info("Enter a ticker to test.")
        else:
            df_test = load_prices(ticker, start, end)
            st.write(f"Rows: {len(df_test)}, Columns: {list(df_test.columns)}")
            ind = compute_indicators(df_test)
            na_rate = ind.isna().mean().round(3).to_dict() if not ind.empty else {}
            st.write("NA rate by column:", na_rate)
            ok_cols = all(c in ind.columns for c in ["RSI_14","MACD","MACD_signal","ATR_14","BandWidth","SMA_200"]) if not ind.empty else False
            st.write("Indicators present:", ok_cols)
            st.success("Diagnostics complete.")
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")
        st.code(traceback.format_exc())

# ---------- Tab 7 — Policies ----------
with tabs[6]:
    header("📜", "Policies (Short)", level=2)
    today = date.today().isoformat()
    tos = f"""# Terms of Service (Short)
_Last updated: {today}_

- Educational information only. **Not investment advice**.
- No trade execution; no access to funds.
- **Backtests are hypothetical**; past performance != future results.
- Data may be delayed/inaccurate. No guarantees. Use at your own risk.
- By using this app, you agree to these terms.
"""
    st.markdown(tos)
    st.download_button("⬇️ Download ToS (.md)", data=tos.encode("utf-8"), file_name="TERMS_OF_SERVICE.md", mime="text/markdown")
    st.download_button("⬇️ Download ToS (PDF)", data=to_pdf_bytes("Terms of Service (Short)", tos), file_name="TERMS_OF_SERVICE.pdf", mime="application/pdf")

# ---------- Tab 8 — Learn ----------
with tabs[7]:
    header("🎓", "Learn — Indicators & Signals", level=2)
    st.markdown("""
**Price vs SMA200** — Above 200‑day SMA: longer‑term uptrend (often better for longs). Below: downtrend (riskier).
**RSI (14)** — 0–100 momentum. >70 often "overbought"; <30 "oversold". Lower RSI may be better for bargain entries.
**MACD & Signal** — Trend momentum. MACD above Signal = bullish tilt; below = bearish.
**Bollinger Bands (20,2)** — Volatility envelope ~2σ from 20‑day mean. Narrow bandwidth can precede breakouts.
**ATR (14)** — Average daily range. Higher ATR = more risk & opportunity.
**Trade Score (0–100)** — Heuristic mix of RSI/MACD/trend/squeeze. Higher is better for bullish alignment. Not a guarantee.
**Heatmap** — Multi-timeframe returns with optional sector grouping; bubbles show average volume.
**EPS (Quarterly)** — Earnings per share trend. Higher and rising is generally better (valuation & guidance still matter).
""")
    st.info("Educational only. None of this is investment advice.")
