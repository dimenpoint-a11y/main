
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import matplotlib.pyplot as plt
import pytz, re, io, math, traceback
from datetime import datetime, date, time as dtime, timedelta
from fpdf import FPDF

APP_VERSION = "v6.6.2"

st.set_page_config(page_title="Trader Dashboard - Pro", layout="wide")

# Altair globals
alt.data_transformers.disable_max_rows()
alt.themes.enable("opaque")

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

def safe_altair_chart(chart, **kwargs):
    try:
        st.altair_chart(chart, **kwargs)
        return True
    except Exception as e:
        st.warning(f"Chart engine fallback: {e}")
        return False

def st_line_chart_fallback(df: pd.DataFrame, ycols, xcol=None):
    data = df.copy()
    if xcol and xcol in data.columns:
        data = data.set_index(xcol)
    try:
        st.line_chart(data[ycols])
    except Exception as e:
        st.error(f"Fallback chart failed: {e}")

def sparkline_chart(df: pd.DataFrame, field="Close"):
    df2 = df.copy()
    if "Date" not in df2.columns:
        df2 = df2.reset_index()
    if "Date" not in df2.columns:
        dcols = [c for c in df2.columns if pd.api.types.is_datetime64_any_dtype(df2[c])]
        if dcols:
            df2.rename(columns={dcols[0]:"Date"}, inplace=True)
        else:
            df2.insert(0, "Date", pd.to_datetime(df2.index))
    df2["Date"] = pd.to_datetime(df2["Date"])
    if field not in df2.columns:
        for cand in ["Close","Adj Close","AdjClose","Adj_Close","close"]:
            if cand in df2.columns:
                field = cand; break
    return alt.Chart(df2.tail(60)).mark_area(opacity=0.2).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(f"{field}:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
        tooltip=["Date:T", f"{field}:Q"]
    ).interactive()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    dm_mode = st.radio("Dark Mode", ["Auto (by clock)", "Light", "Dark"], index=0)
    auto_refresh = st.toggle("Auto-refresh", value=True, help="JS-based reload; no extra packages.")
    interval_s = st.slider("Refresh every (seconds)", 5, 600, 30)
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
  .chg-pos { color: #059669; }
  .chg-neg { color: #dc2626; }
  .subgrid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .75rem 2rem; }
  .kv { display:flex; justify-content: space-between; gap: 1rem; }
  .kv .k { color: #6b7280; }
  .kv .v { font-weight: 600; }
  .card { border:1px solid var(--border,#e5e7eb); border-radius:12px; padding:.5rem .75rem; }
  .muted { color: #6b7280; font-size:.9rem; }
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
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # normalize columns
    ren = {c:str(c).title() for c in data.columns}
    data = data.rename(columns=ren).reset_index()
    if "Date" not in data.columns:
        dcols = [c for c in data.columns if pd.api.types.is_datetime64_any_dtype(data[c])]
        if dcols: data.rename(columns={dcols[0]:"Date"}, inplace=True)
        else: data.insert(0,"Date", pd.to_datetime(data.index))
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    # Ensure Close present (fallback to Adj Close if needed)
    if "Close" not in data.columns:
        for altc in ["Adj Close","AdjClose","Adj_Close","close"]:
            if altc in data.columns:
                data.rename(columns={altc:"Close"}, inplace=True)
                break
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
    if "Close" not in df.columns:
        return df  # minimal graceful
    out = df.copy()
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
    out["BandWidth"] = (out["BB_UP"] - out["BB_LO"]) / out["BB_MA"]
    out["Above_SMA200"] = out["Close"] > out["SMA_200"]
    # DO NOT drop rows here — short timeframes would wipe plots
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
            ex_div = pd.to_datetime(ex_div_ts, errors='coerce')
            ex_div = ex_div.date() if not pd.isna(ex_div) else None
    else:
        ex_div = None
    target = pick("targetMeanPrice", "targetMedianPrice", srcs=(info,))

    name = pick("shortName", "longName", srcs=(info,)) or ticker
    currency = pick("currency", srcs=(fast, info)) or "USD"
    exch = pick("exchange", "fullExchangeName", srcs=(info, fast))
    tz = pick("timezone", "exchangeTimezoneShortName", srcs=(fast, info))

    return {
        "name": name, "currency": currency, "exchange": exch, "timezone": tz,
        "last": last, "prev_close": prev_close, "open": open_,
        "day_low": day_low, "day_high": day_high,
        "year_low": year_low, "year_high": year_high,
        "volume": volume, "avg_volume": avg_volume, "mcap": mcap,
        "beta": beta, "pe": pe, "eps": eps, "earnings_date": edisplay,
        "forward_div_yield": fwd_div, "ex_div": ex_div, "target": target
    }

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
        if isinstance(qe, pd.DataFrame) and not qe.empty:
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
if "watchlist_extra" in st.session_state:
    extra = [s for s in st.session_state["watchlist_extra"] if s]
    symbols = list(dict.fromkeys(symbols + extra))[:60]

tabs = st.tabs([
    "Price & Indicators", "Backtest", "Earnings & Insider", "Watchlist", "Heatmap", "Diagnostics", "Policies", "Learn"
])

# ---------- Tab 1 — Price & Indicators (with Signals merged) ----------
with tabs[0]:
    header("📊", "Quote — Yahoo-style Overview", level=1)
    st.info("Hover charts for values • Scroll/drag to zoom/pan • Double-click to reset. Educational only, not advice.")
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
                hist = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    ok = safe_altair_chart(sparkline_chart(hist, "Close"), use_container_width=True)
                    if not ok: st_line_chart_fallback(hist.reset_index(), ["Close"])

            with col2:
                sign = "pos" if (chg or 0) >= 0 else "neg"
                chg_str = f"{'+' if (chg or 0)>=0 else ''}{fmt_float(chg,2)}"
                pct_str = f"{'+' if (chg or 0)>=0 else ''}{fmt_float(chg_pct,2)}%"
                st.markdown(f"<div class='price-big chg-{sign}'>{fmt_float(q['last'],2)} {q['currency']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chg-{sign}'> {chg_str} ({pct_str}) today</div>", unsafe_allow_html=True)

            with col3:
                add_watch = st.button("★ Add to Watchlist")
                if add_watch and ticker not in symbols:
                    st.session_state.setdefault("watchlist_extra", set())
                    st.session_state["watchlist_extra"].add(ticker)
                    st.success("Added to session watchlist.")
                tf = st.radio("Timeframe", ["1D","5D","1M","6M","YTD","1Y","5Y","Max"], horizontal=True, index=5)
                if st.button("Reset zoom"):
                    st.experimental_rerun()

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
            dfi = compute_indicators(df_raw)
            if dfi.empty or "Close" not in dfi.columns:
                st.warning("No price data available for the selected range.")
            else:
                dfx = dfi.dropna(subset=["Close"]).copy()
                if "Date" not in dfx.columns:
                    dfx["Date"] = pd.to_datetime(df_raw["Date"])
                # Controls
                st.markdown("#### Chart")
                c4, c5, c6, c7 = st.columns([1,1,1,1])
                with c4: show_sma20 = st.checkbox("SMA 20", value=True)
                with c5: show_sma50 = st.checkbox("SMA 50", value=True)
                with c6: show_sma200 = st.checkbox("SMA 200", value=True)
                with c7: show_bb = st.checkbox("Bollinger (20,2)", value=False)
                logy = st.toggle("Log scale", value=False)

                zoom = alt.selection_interval(bind='scales', encodings=['x'])
                hover = alt.selection(type="single", nearest=True, on="mouseover", fields=["Date"], empty="none")
                base = alt.Chart(dfx).encode(x=alt.X("Date:T", title="Date"))
                layers = []
                if show_bb and "BB_LO" in dfx.columns and "BB_UP" in dfx.columns:
                    layers.append(base.mark_area(opacity=0.15).encode(
                        y=alt.Y("BB_LO:Q", title="Price", scale=alt.Scale(type="log") if logy else alt.Scale(type="linear")),
                        y2="BB_UP:Q",
                        tooltip=["Date:T","Close:Q","BB_LO:Q","BB_UP:Q"]
                    ).add_params(zoom))
                price_line = base.mark_line().encode(
                    y=alt.Y("Close:Q", title="Price", scale=alt.Scale(type="log") if logy else alt.Scale(type="linear")),
                    tooltip=["Date:T","Close:Q"]
                )
                points = base.mark_circle(size=35).encode(
                    y="Close:Q",
                    opacity=alt.condition(hover, alt.value(1), alt.value(0))
                ).add_params(hover)
                rule = alt.Chart(dfx).mark_rule(color="gray").encode(x="Date:T").transform_filter(hover)
                layers.extend([price_line.add_params(zoom), points.add_params(zoom), rule])
                if show_sma20 and "SMA_20" in dfx.columns:
                    layers.append(base.mark_line(strokeDash=[4,2]).encode(y="SMA_20:Q", tooltip=["Date:T","SMA_20:Q"]).add_params(zoom))
                if show_sma50 and "SMA_50" in dfx.columns:
                    layers.append(base.mark_line(strokeDash=[4,2]).encode(y="SMA_50:Q", tooltip=["Date:T","SMA_50:Q"]).add_params(zoom))
                if show_sma200 and "SMA_200" in dfx.columns:
                    layers.append(base.mark_line(strokeDash=[6,3]).encode(y="SMA_200:Q", tooltip=["Date:T","SMA_200:Q"]).add_params(zoom))
                chart_price = alt.layer(*layers).resolve_scale(y='shared').properties(height=380)
                ok = safe_altair_chart(chart_price, use_container_width=True)
                if not ok:
                    st_line_chart_fallback(dfx, ["Close","SMA_20","SMA_50","SMA_200"], xcol="Date")
                st.caption("Price chart: **higher is better** for longs (context matters). SMA lines smooth price; Bollinger bands show volatility envelope.")

                # --- Merged Signals
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
                ok = safe_altair_chart(
                    alt.Chart(df_parts).mark_bar().encode(
                        x=alt.X("Component:N", sort=None, title=None), y=alt.Y("Contribution:Q"),
                        tooltip=["Component:N","Contribution:Q"]
                    ),
                    use_container_width=True
                )
                if not ok:
                    st_line_chart_fallback(df_parts.reset_index(), ["Contribution"], xcol="Component")
                st.caption("Signals meaning: **more positive** bars = more bullish alignment; **negative** bars = bearish tilt. Heuristic only.")

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
    if not ticker:
        st.info("Load price data first.")
    else:
        dfr = load_prices(ticker, start, end)
        if dfr.empty:
            st.info("No data.")
        else:
            st.subheader("Quick Strategy Backtests (toy models)")
            dfb = compute_indicators(dfr)
            # Build signals even if some NA
            dfb["pos_sma"] = np.where(dfb["SMA_20"] > dfb["SMA_50"], 1, 0)
            dfb["ret_sma"] = dfb["pos_sma"].shift() * dfb["Returns"]
            cum_sma = (1 + dfb["ret_sma"].fillna(0)).cumprod()

            dfb["pos_rsi"] = np.where(dfb["RSI_14"] < 30, 1, np.where(dfb["RSI_14"] > 70, 0, np.nan))
            dfb["pos_rsi"] = dfb["pos_rsi"].ffill().fillna(0)
            dfb["ret_rsi"] = dfb["pos_rsi"].shift() * dfb["Returns"]
            cum_rsi = (1 + dfb["ret_rsi"].fillna(0)).cumprod()

            cum_bh = (1 + dfb["Returns"].fillna(0)).cumprod()

            perf = pd.DataFrame({"Date": dfb["Date"], "Buy&Hold": cum_bh, "SMA(20>50)": cum_sma, "RSI<30 long": cum_rsi}).dropna(subset=["Date"]).reset_index(drop=True)
            chart = alt.Chart(perf).transform_fold(
                ["Buy&Hold","SMA(20>50)","RSI<30 long"], as_=["Strategy","Equity"]
            ).mark_line().encode(
                x=alt.X("Date:T"), y=alt.Y("Equity:Q", title="Cumulative (×)"),
                color="Strategy:N", tooltip=["Date:T","Strategy:N","Equity:Q"]
            ).interactive()
            ok = safe_altair_chart(chart, use_container_width=True)
            if not ok:
                st_line_chart_fallback(perf, ["Buy&Hold","SMA(20>50)","RSI<30 long"], xcol="Date")
            st.caption("Equity curve: **higher is better**. Compare lines to see relative strategy performance.")

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
            ok = safe_altair_chart(alt.Chart(met_df).mark_bar().encode(x="Metric:N", y="Value:Q", tooltip=["Metric","Value"]).interactive(), use_container_width=True)
            if not ok:
                st_line_chart_fallback(met_df, ["Value"], xcol="Metric")
            st.caption("Sharpe: **higher is better**; Max Drawdown: **less negative is better**.")

            buf = io.StringIO(); perf.to_csv(buf, index=False)
            st.download_button("⬇️ Download equity curves (CSV)", data=buf.getvalue(), file_name=f"{ticker}_backtest_equity.csv", mime="text/csv")

# ---------- Tab 3 — Earnings & Insider (EPS chart) ----------
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
                alt.Chart(df_eps).mark_line(point=True).encode(
                    x=alt.X("Date:T", title="Quarter"),
                    y=alt.Y("EPS:Q", title="EPS (USD)"),
                    tooltip=["Date:T","EPS:Q"]
                ),
                alt.Chart(df_eps).mark_line(strokeDash=[4,2]).encode(
                    x="Date:T", y="EPS_MA4:Q", tooltip=["Date:T","EPS_MA4:Q"]
                )
            ).interactive()
            ok = safe_altair_chart(chart, use_container_width=True)
            if not ok: st_line_chart_fallback(df_eps, ["EPS","EPS_MA4"], xcol="Date")
            st.caption("EPS: **higher is better**; rising trend often constructive. Dashed line = 4‑Q rolling mean.")
        else:
            st.info("EPS series not available for this ticker.")
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
    if not symbols:
        st.info("Add tickers in the sidebar.")
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
                        color = "chg-pos" if chg >= 0 else "chg-neg"
                        st.markdown(f"<div class='{color}'>{fmt_float(q['last'],2)} ({'+' if chg>=0 else ''}{fmt_float(chg_pct,2)}%)</div>", unsafe_allow_html=True)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        ok = safe_altair_chart(sparkline_chart(df, "Close"), use_container_width=True)
                        if not ok: st_line_chart_fallback(df, ["Close"], xcol="Date")
                except Exception as e:
                    st.caption(f"{sym}: {e}")

# ---------- Tab 5 — Heatmap ----------
with tabs[4]:
    header("🗺️", "Heatmap — % Change", level=2)
    metric = st.radio("Metric", ["1D %", "1M %"], horizontal=True)
    if not symbols:
        st.info("Add tickers in the sidebar.")
    else:
        try:
            data = yf.download(" ".join(symbols), period="2mo", interval="1d", auto_adjust=True, progress=False, group_by='ticker', threads=True)
        except Exception:
            data = None
        rows = []
        for sym in symbols:
            try:
                if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex) and sym in data.columns.get_level_values(0):
                    d = data[sym].copy()
                else:
                    d = yf.download(sym, period="2mo", interval="1d", auto_adjust=True, progress=False)
                if d is None or d.empty: 
                    rows.append({"Ticker": sym, "1D %": np.nan, "1M %": np.nan})
                    continue
                d = d.dropna()
                d1 = (d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) if len(d) >= 2 else np.nan
                d21 = (d["Close"].iloc[-1] / d["Close"].iloc[-21] - 1) if len(d) >= 22 else np.nan
                rows.append({"Ticker": sym, "1D %": d1*100.0, "1M %": d21*100.0})
            except Exception:
                rows.append({"Ticker": sym, "1D %": np.nan, "1M %": np.nan})
        dfh = pd.DataFrame(rows)
        if dfh.empty:
            st.info("No data for heatmap.")
        else:
            dfm = dfh[["Ticker", metric]].dropna()
            if dfm.empty:
                st.info("Not enough data to render heatmap.")
            else:
                dfm["Metric"] = metric
                heat = alt.Chart(dfm).mark_rect().encode(
                    x=alt.X("Ticker:N", sort=dfm["Ticker"].tolist(), title=None),
                    y=alt.Y("Metric:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme='redblue', domainMid=0)),
                    tooltip=["Ticker:N", f"{metric}:Q"]
                ).properties(height=140)
                ok = safe_altair_chart(heat, use_container_width=True)
                if not ok:
                    st.dataframe(dfh.set_index("Ticker"), use_container_width=True)
            st.caption("Heatmap: **blue = positive %, red = negative %**, midpoint 0% is white.")

# ---------- Tab 6 — Diagnostics ----------
with tabs[5]:
    header("🔧", "Diagnostics", level=2)
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

# ---------- Tab 7 — Policies (simplified) ----------
with tabs[6]:
    header("📜", "Policies (Short)", level=2)
    today = date.today().isoformat()
    tos = f"""# Terms of Service (Short)
_Last updated: {today}_

- Educational information only. **Not investment advice**.
- No trade execution; no access to funds.
- **Backtests are hypothetical**; **past performance ≠ future results**.
- Data may be delayed/inaccurate. No guarantees. Use at your own risk.
- By using this app, you agree to these terms.
"""
    st.markdown(tos)
    st.download_button("⬇️ Download ToS (.md)", data=tos.encode("utf-8"), file_name="TERMS_OF_SERVICE.md", mime="text/markdown")
    st.download_button("⬇️ Download ToS (PDF)", data=to_pdf_bytes("Terms of Service (Short)", tos), file_name="TERMS_OF_SERVICE.pdf", mime="application/pdf")

# ---------- Tab 8 — Learn (Novice-friendly explanations) ----------
with tabs[7]:
    header("🎓", "Learn — Indicators & Signals", level=2)
    st.markdown("""
**Price vs SMA200** — Above 200‑day SMA: longer‑term uptrend (often better for longs). Below: downtrend (riskier).  
**RSI (14)** — 0–100 momentum. >70 often "overbought"; <30 "oversold". **Lower RSI is better** for bargain entries; **extreme highs can persist** in strong trends.  
**MACD & Signal** — Trend momentum. MACD above Signal = bullish tilt; below = bearish. **Higher MACD minus Signal is better** for momentum.  
**Bollinger Bands (20,2)** — Volatility envelope ~2σ from 20‑day mean. **Narrow bandwidth can precede breakouts**; touches indicate strength/weakness.  
**ATR (14)** — Average daily range. **Higher ATR = more risk & opportunity**; size positions accordingly.  
**Trade Score (0–100)** — Heuristic mix of RSI/MACD/trend/squeeze. **Higher is better** for bullish alignment. Not a guarantee.  
**Heatmap (1D/1M)** — Quick relative performance snapshot across your tickers. **Blue up / Red down** vs 0%.  
**EPS (Quarterly)** — Earnings per share trend. **Higher and rising is better** (valuation & guidance still matter).
""")
    st.info("Educational only. None of this is investment advice.")
