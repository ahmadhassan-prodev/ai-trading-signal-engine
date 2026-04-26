import ccxt
import pandas as pd
import time

def fetch_data():
    exchange = ccxt.binance()

    symbol = "SOL/USDT"
    timeframe = "1h" 

    since_days = 1100
    now = exchange.milliseconds()
    since = now - since_days * 24 * 60 * 60 * 1000 

    all_ohlcv = []
    limit = 1000 

    print(f"Fetching {symbol} {timeframe} data for last {since_days} days...")

    while since < now:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
        if len(ohlcv) < limit:
            break


    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.to_csv("sol_1h_raw.csv", index=False)

    print(f"Done! Saved {len(df)} rows to sol_1h_raw.csv")
    print(df.head())
    print(df.tail()) 