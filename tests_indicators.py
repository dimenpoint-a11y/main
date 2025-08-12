
# tests_indicators.py
import pandas as pd, numpy as np, yfinance as yf

def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
def _macd(s, f=12, sl=26, sg=9):
    m = _ema(s, f) - _ema(s, sl)
    sig = _ema(m, sg)
    return m, sig, m - sig
def _bbands(s, w=20, k=2.0):
    ma = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return ma, ma + k*sd, ma - k*sd
def _atr(df, n=14):
    h_l = df["High"] - df["Low"]
    h_pc = (df["High"] - df["Close"].shift()).abs()
    l_pc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def run_tests():
    df = yf.download("AAPL", start="2024-01-01", end="2025-01-01", auto_adjust=True, progress=False)
    assert not df.empty, "No data downloaded."
    close = df["Close"].copy()
    rsi = _rsi(close, 14).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    assert ema12.diff().abs().mean() >= ema26.diff().abs().mean()
    macd_line, signal_line, hist = _macd(close)
    assert ((macd_line - signal_line).dropna().sub(hist.dropna(), fill_value=0).abs() < 1e-6).all()
    _, up, lo = _bbands(close, 20, 2.0)
    assert ((up - lo).dropna() >= 0).all()
    atr = _atr(df, 14).dropna()
    assert (atr >= 0).all() and atr.mean() > 0
    print("All indicator tests PASSED ✅")

if __name__ == "__main__":
    run_tests()
