# tests_indicators.py
import pandas as pd, numpy as np, yfinance as yf

def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def run_tests():
    df = yf.download("AAPL", start="2024-01-01", end="2025-01-01", auto_adjust=True, progress=False)
    assert not df.empty, "No data downloaded."
    close = df["Close"].copy()
    rsi = _rsi(close, 14).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    assert ema12.diff().abs().mean() >= ema26.diff().abs().mean()
    print("All indicator tests PASSED ✅")

if __name__ == "__main__":
    run_tests()