"""
NSE Nifty 100 intraday scanner (5-min candles, Yahoo Finance via yfinance)

Indicators: VWAP, EMA 9/21, RSI(14), MACD histogram, ADX/DI(14),
            Choppiness Index(14), relative volume.
No backtesting. Stop loss and target are left to you.

Install : pip install yfinance pandas numpy requests
Run once: python nse_intraday_scanner.py
Loop    : python nse_intraday_scanner.py --loop        (re-scans every 5 min)
Telegram: set env TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, then add --notify
          (--new-only sends a signal only the first time it appears that day)
Options : --min-score 6   (conditions needed out of 7)
          --show-all      (print every stock's score, not only signals)

Educational tool, not investment advice. Yahoo's NSE intraday data can be
delayed or have gaps, so always confirm the price on your broker terminal.
"""
import argparse
import io
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- settings ------------------------------------
INTERVAL = "5m"
PERIOD = "5d"          # extra days only warm up the indicators; VWAP uses today
CHUNK = 20             # tickers per Yahoo request (small batches avoid throttling)
RSI_N, CI_N, ADX_N = 14, 14, 14
CI_MAX_TREND = 50      # CI below this = trending enough (61.8+ choppy, 38.2- strongly trending)
ADX_MIN = 20
VOL_RATIO_MIN = 1.2    # last candle volume vs average of previous 20 candles
BUY_RSI = (55, 75)     # momentum zone for longs
SELL_RSI = (25, 45)    # momentum zone for shorts

IST = ZoneInfo("Asia/Kolkata")
STATE_FILE = Path(os.environ.get("STATE_FILE", "last_signals.json"))

NSE_CSV = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"

# Fallback list, used only if the NSE download fails. Index membership changes,
# so the live NSE list is preferred. Tickers with no Yahoo data are skipped.
FALLBACK = """RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL SBIN ITC HINDUNILVR LT
KOTAKBANK AXISBANK BAJFINANCE HCLTECH MARUTI SUNPHARMA M&M NTPC ONGC POWERGRID
TITAN ULTRACEMCO ASIANPAINT ADANIENT ADANIPORTS COALINDIA WIPRO BAJAJFINSV
NESTLEIND JSWSTEEL TATASTEEL TECHM HINDALCO GRASIM CIPLA DRREDDY EICHERMOT
APOLLOHOSP BPCL TATACONSUM BRITANNIA HEROMOTOCO INDUSINDBK SBILIFE HDFCLIFE
BAJAJ-AUTO SHRIRAMFIN TRENT ETERNAL BEL HAL ADANIPOWER ADANIGREEN AMBUJACEM
BANKBARODA CANBK PNB UNIONBANK BOSCHLTD CHOLAFIN COLPAL DABUR DLF DIVISLAB GAIL
GODREJCP HAVELLS ICICIGI ICICIPRULI IOC INDIGO IRCTC IRFC JINDALSTEL JIOFIN LICI
LODHA LTIM MARICO MOTHERSON NAUKRI PIDILITIND PFC RECLTD SIEMENS SRF TORNTPHARM
TVSMOTOR VEDL VBL ZYDUSLIFE ABB BAJAJHLDG BERGEPAINT CGPOWER HDFCAMC MAXHEALTH
MUTHOOTFIN POLYCAB SHREECEM TATAPOWER NHPC MAZDOCK HINDZINC SWIGGY""".split()


def get_universe():
    """Nifty 100 symbols from NSE; falls back to the built-in list."""
    try:
        r = requests.get(NSE_CSV, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        syms = pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()
        if len(syms) >= 90:
            return [s + ".NS" for s in syms]
    except Exception as e:
        print(f"[info] NSE list download failed ({e}); using fallback list.")
    return [s + ".NS" for s in FALLBACK]


# ----------------------------- indicators ----------------------------------
def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n=RSI_N):
    d = close.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn)


def true_range(df):
    pc = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"],
                      (df["High"] - pc).abs(),
                      (df["Low"] - pc).abs()], axis=1).max(axis=1)


def choppiness(df, n=CI_N):
    tr_sum = true_range(df).rolling(n).sum()
    rng = df["High"].rolling(n).max() - df["Low"].rolling(n).min()
    return 100 * np.log10(tr_sum / rng.replace(0, np.nan)) / np.log10(n)


def adx_di(df, n=ADX_N):
    up, down = df["High"].diff(), -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = wilder(true_range(df), n)
    pdi = 100 * wilder(plus_dm, n) / atr
    mdi = 100 * wilder(minus_dm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n), pdi, mdi


def session_vwap(df):
    """VWAP that resets every trading day."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.normalize()
    cum_pv = (tp * df["Volume"]).groupby(day).cumsum()
    cum_v = df["Volume"].groupby(day).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def analyze(df, min_score):
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 60:
        return None

    c = df["Close"]
    ema9, ema21 = c.ewm(span=9, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    adx, pdi, mdi = adx_di(df)
    r, ci, vwap = rsi(c), choppiness(df), session_vwap(df)
    vol_ratio = df["Volume"] / df["Volume"].shift().rolling(20).mean()

    i = -1
    last = dict(price=c.iloc[i], vwap=vwap.iloc[i], rsi=r.iloc[i], ci=ci.iloc[i],
                adx=adx.iloc[i], pdi=pdi.iloc[i], mdi=mdi.iloc[i],
                vr=vol_ratio.iloc[i], e9=ema9.iloc[i], e21=ema21.iloc[i], h=hist.iloc[i])
    if any(pd.isna(v) for v in last.values()):
        return None

    trending = last["ci"] < CI_MAX_TREND
    vol_ok = last["vr"] >= VOL_RATIO_MIN
    buy = [last["price"] > last["vwap"],
           last["e9"] > last["e21"],
           BUY_RSI[0] <= last["rsi"] <= BUY_RSI[1],
           last["h"] > 0,
           trending,
           last["adx"] > ADX_MIN and last["pdi"] > last["mdi"],
           vol_ok]
    sell = [last["price"] < last["vwap"],
            last["e9"] < last["e21"],
            SELL_RSI[0] <= last["rsi"] <= SELL_RSI[1],
            last["h"] < 0,
            trending,
            last["adx"] > ADX_MIN and last["mdi"] > last["pdi"],
            vol_ok]
    bs, ss = sum(buy), sum(sell)

    signal = "BUY" if bs >= min_score and bs > ss else "SELL" if ss >= min_score and ss > bs else "-"
    return {"Signal": signal,
            "Price": round(last["price"], 2),
            "VWAP": round(last["vwap"], 2),
            "RSI": round(last["rsi"], 1),
            "CI": round(last["ci"], 1),
            "ADX": round(last["adx"], 1),
            "VolX": round(last["vr"], 2),
            "Score": max(bs, ss),
            "Candle": df.index[-1].strftime("%d-%b %H:%M")}


# ----------------------------- data + scan ---------------------------------
def drop_forming_candle(df):
    """Yahoo returns the in-progress candle too; indicators should use closed ones."""
    if len(df) and df.index[-1] + timedelta(minutes=5) > pd.Timestamp.now(tz=df.index.tz):
        return df.iloc[:-1]
    return df


def _get_one(data, t, single):
    """Pull one ticker's OHLCV out of a yf.download result."""
    try:
        d = data if (single and not isinstance(data.columns, pd.MultiIndex)) else data[t]
    except KeyError:
        return None
    d = d.dropna(how="all")
    return d if len(d) else None


def download(tickers, retries=3):
    """Small batches, no parallel threads, and retries, to avoid Yahoo throttling."""
    frames, pending = {}, list(tickers)
    for attempt in range(retries):
        for k in range(0, len(pending), CHUNK):
            part = pending[k:k + CHUNK]
            try:
                data = yf.download(part, period=PERIOD, interval=INTERVAL, group_by="ticker",
                                   auto_adjust=False, threads=False, progress=False)
            except Exception as e:
                print(f"[warn] batch failed: {e}")
                data = None
            if data is not None and len(data):
                for t in part:
                    d = _get_one(data, t, len(part) == 1)
                    if d is not None:
                        frames[t] = d
            time.sleep(1.5)                      # be gentle with Yahoo
        pending = [t for t in pending if t not in frames]
        if not pending:
            break
        if attempt < retries - 1:
            wait = 8 * (attempt + 1)
            print(f"[info] {len(pending)} tickers missing, retrying in {wait}s...")
            time.sleep(wait)
    if pending:
        print(f"[info] No data for {len(pending)} tickers (new listings or not on Yahoo): "
              f"{', '.join(t.replace('.NS', '') for t in pending)}")
    return frames


def send_telegram(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[warn] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; message not sent.")
        return
    for k in range(0, len(text), 3500):          # Telegram limit is 4096 chars
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat, "text": text[k:k + 3500]}, timeout=15)
            if not r.ok:
                print(f"[warn] Telegram error: {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"[warn] Telegram send failed: {e}")


def build_message(sig, show, candle_time, new_only):
    head = f"Nifty100 5m scan | candle {candle_time} IST" + (" | NEW signals" if new_only else "")
    lines = [head]
    for name, icon in (("BUY", "\U0001F7E2"), ("SELL", "\U0001F534")):
        syms = [x for x in sig.index if x in show and sig.loc[x, "Signal"] == name]
        if syms:
            lines.append(f"\n{icon} {name}")
            for x in syms:
                r = sig.loc[x]
                lines.append(f"{x}  Rs {r['Price']}  VWAP {r['VWAP']}  RSI {r['RSI']}  "
                             f"ADX {r['ADX']}  Vol {r['VolX']}x")
    lines.append("\nNot advice. Verify live price on your broker before trading.")
    return "\n".join(lines)


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def scan(tickers, min_score, show_all, notify=False, new_only=False):
    frames = download(tickers)
    rows, last_ts = {}, None
    for t, d in frames.items():
        d = drop_forming_candle(d)
        if len(d):
            ts = d.index[-1]
            last_ts = ts if last_ts is None or ts > last_ts else last_ts
        res = analyze(d, min_score)
        if res:
            rows[t.replace(".NS", "")] = res
    if not rows:
        print("No data returned. Market closed, or Yahoo is throttling. Try again shortly.")
        return None

    out = pd.DataFrame(rows).T
    latest = out["Candle"].max()
    live = (pd.Timestamp.now(tz=last_ts.tz) - last_ts) < timedelta(minutes=25)
    print(f"\n=== Scan at {datetime.now(IST):%d-%b-%Y %H:%M:%S} IST | latest 5-min candle: {latest} "
          f"| scanned {len(out)}/{len(tickers)} stocks ===")
    stale = out[out["Candle"] != latest]
    if len(stale):
        print(f"[note] {len(stale)} stocks have older candles (illiquid or delayed data).")

    sig = out[out["Signal"] != "-"].sort_values(["Signal", "Score", "ADX"], ascending=[True, False, False])
    for name in ("BUY", "SELL"):
        sub = sig[sig["Signal"] == name]
        print(f"\n--- {name} candidates ({len(sub)}) ---")
        print(sub.drop(columns="Signal").to_string() if len(sub) else "none")
    if show_all:
        print("\n--- All stocks ---")
        print(out.sort_values("Score", ascending=False).to_string())
    sig.to_csv("scan_results.csv")

    if notify:
        if not live:
            print("[skip] Data is not from a live session (holiday, closed, or stale). No Telegram sent.")
            return sig
        today = datetime.now(IST).strftime("%Y-%m-%d")
        prev = load_state()
        prev_sigs = prev.get("signals", {}) if prev.get("date") == today else {}
        cur = {x: sig.loc[x, "Signal"] for x in sig.index}
        show = {x for x, v in cur.items() if not new_only or prev_sigs.get(x) != v}
        try:
            STATE_FILE.write_text(json.dumps({"date": today, "signals": cur}))
        except Exception as e:
            print(f"[warn] could not save state: {e}")
        if show:
            candle_time = last_ts.tz_convert(IST).strftime("%H:%M")
            send_telegram(build_message(sig, show, candle_time, new_only))
            print(f"[telegram] sent {len(show)} signals")
        else:
            print("[telegram] nothing new to send")
    return sig


def market_open_now():
    now = datetime.now(IST)
    return now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=int, default=6, help="conditions met out of 7")
    ap.add_argument("--loop", action="store_true", help="re-scan every 5 minutes")
    ap.add_argument("--show-all", action="store_true")
    ap.add_argument("--notify", action="store_true", help="send signals to Telegram")
    ap.add_argument("--new-only", action="store_true", help="notify only signals new today")
    a = ap.parse_args()

    tickers = get_universe()
    print(f"Universe: {len(tickers)} stocks")
    if not a.loop:
        scan(tickers, a.min_score, a.show_all, a.notify, a.new_only)
        return
    while True:
        if market_open_now():
            scan(tickers, a.min_score, a.show_all, a.notify, a.new_only)
        else:
            print(f"{datetime.now(IST):%H:%M} IST market closed, waiting...")
        time.sleep(300 - (time.time() % 300) + 10)   # wake 10s after each 5-min mark


if __name__ == "__main__":
    main()
