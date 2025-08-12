# tests_score.py
import numpy as np

def trade_score(row):
    score = 50.0
    rsi = row.get("RSI_14", np.nan)
    macd = row.get("MACD", np.nan)
    macd_signal = row.get("MACD_signal", np.nan)
    sma200 = row.get("SMA_200", np.nan)
    close = row.get("Close", np.nan)
    bw = row.get("BandWidth", np.nan)
    if not np.isnan(rsi):
        if rsi < 30: score += 10
        elif rsi > 70: score -= 10
    if not np.isnan(macd) and not np.isnan(macd_signal):
        if macd > macd_signal: score += 10
        else: score -= 5
    if not np.isnan(sma200) and not np.isnan(close):
        if close > sma200: score += 10
        else: score -= 10
    if not np.isnan(bw) and bw < 0.05: score += 5
    return float(np.clip(score, 0, 100))

def run_tests():
    base = {"RSI_14":50, "MACD":0.0, "MACD_signal":0.0, "SMA_200":100.0, "Close":100.0, "BandWidth":0.10}
    s0 = trade_score(base); assert 0 <= s0 <= 100
    assert trade_score({**base, "RSI_14":25}) >= s0
    assert trade_score({**base, "RSI_14":80}) <= s0
    assert trade_score({**base, "Close":120}) >= s0
    assert trade_score({**base, "Close":80}) <= s0
    assert trade_score({**base, "MACD":1, "MACD_signal":0}) >= s0
    assert trade_score({**base, "MACD":0, "MACD_signal":1}) <= s0
    assert trade_score({**base, "BandWidth":0.03}) >= s0
    print("Trade score tests PASSED ✅")

if __name__ == "__main__":
    run_tests()