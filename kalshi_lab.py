"""Read-only Kalshi 15-minute market research starter. Never places orders."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE = "https://external-api.kalshi.com/trade-api/v2"
DATA = Path(__file__).resolve().parent / "data"


def get(session, path, **params):
    for attempt in range(5):
        try:
            r = session.get(BASE + path, params=params, timeout=25)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("API did not respond successfully")


def list_markets(session, path, series, limit):
    found, cursor = [], None
    for _ in range(max(2, math.ceil(limit / 100) + 2)):
        params = dict(series_ticker=series, limit=min(100, limit - len(found)))
        if path == "/markets":
            params["status"] = "settled"
        if cursor:
            params["cursor"] = cursor
        page = get(session, path, **params)
        found += page.get("markets", [])
        cursor = page.get("cursor")
        if not cursor or len(found) >= limit:
            break
    return found[:limit]


def extract_close(obj):
    if not isinstance(obj, dict):
        return np.nan
    val = obj.get("close_dollars")
    if val is None:
        val = obj.get("close")
        if val is not None:
            return float(val) / 100
    return float(val) if val is not None else np.nan


def fetch(series, limit):
    DATA.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "kalshi-research-example/0.1"
    cutoff = get(session, "/historical/cutoff")["market_settled_ts"]
    print("Kalshi historical market cutoff:", cutoff)
    seen = {}
    for path in ("/markets", "/historical/markets"):
        try:
            for market in list_markets(session, path, series, limit):
                seen[market["ticker"]] = (market, path.startswith("/historical"))
            print(f"{path}: total unique markets so far {len(seen)}")
        except requests.HTTPError as ex:
            print("Skipped", path, ex)
    rows = []
    for i, (ticker, (m, archived)) in enumerate(seen.items()):
        if archived:
            path = f"/historical/markets/{ticker}/candlesticks"
        else:
            path = f"/series/{series}/markets/{ticker}/candlesticks"
        start = pd.to_datetime(m.get("open_time") or m.get("created_time"), utc=True, errors="coerce")
        end = pd.to_datetime(m.get("close_time") or m.get("expiration_time"), utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        try:
            candles = get(session, path, start_ts=int(start.timestamp()) - 60,
                          end_ts=int(end.timestamp()) + 60, period_interval=1).get("candlesticks", [])
        except requests.HTTPError as ex:
            print("Skipped candle", ticker, ex)
            continue
        for c in candles:
            rows.append({"ticker": ticker, "end_period_ts": c.get("end_period_ts"),
                         "bid": extract_close(c.get("yes_bid")),
                         "ask": extract_close(c.get("yes_ask")),
                         "volume": float(c.get("volume_fp") or c.get("volume") or 0),
                         "result": m.get("result")})
        if (i + 1) % 50 == 0:
            print("Processed", i + 1, "markets")
        time.sleep(.08)
    if not rows:
        raise RuntimeError("No candles found. Try another series or a different date range.")
    frame = pd.DataFrame(rows).drop_duplicates(["ticker", "end_period_ts"])
    frame.to_csv(DATA / f"{series}_minute.csv", index=False)
    print("Saved", len(frame), "rows from", frame.ticker.nunique(), "markets")


def prepare(df, horizon):
    df = df.copy().sort_values(["ticker", "end_period_ts"])
    df[["bid", "ask"]] = df[["bid", "ask"]].apply(pd.to_numeric, errors="coerce")
    df["mid"] = (df.bid + df.ask) / 2
    df["spread"] = df.ask - df.bid
    g = df.groupby("ticker", sort=False)
    df["age"] = g.cumcount()
    for lag in [1, 2, 3, 5]:
        df[f"ret_{lag}"] = df.mid - g.mid.shift(lag)
    df["spread_prev"] = g.spread.shift(1)
    df["vol_prev"] = g.volume.shift(1)
    df["future_bid"] = g.bid.shift(-horizon)
    df["future_mid"] = g.mid.shift(-horizon)
    df["entry_ask"] = g.ask.shift(-1)
    df["y"] = df.future_mid - df.mid
    cols = ["mid", "spread", "age", "ret_1", "ret_2", "ret_3", "ret_5", "spread_prev", "vol_prev"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["y", "entry_ask", "future_bid"])
    df = df[(df.spread >= 0) & (df.entry_ask >= 0) & (df.future_bid >= 0) &
            (df.entry_ask <= 1) & (df.future_bid <= 1)]
    return df, cols


def fee(price):
    return np.ceil(0.07 * price * (1 - price) * 10000) / 10000


def evaluate(name, pred, frame, threshold):
    buy = frame.entry_ask.to_numpy()
    sell = frame.future_bid.to_numpy()
    entry_fee = fee(buy)
    exit_fee = fee(sell)
    trade = pred > (frame.spread.to_numpy() + threshold)
    pnl = (sell - buy - entry_fee - exit_fee)[trade]
    print(f"{name:12} | MAE {mean_absolute_error(frame.y, pred):.4f} | "
          f"trades {len(pnl):4d} | total ${pnl.sum():8.2f} | "
          f"average {pnl.mean() if len(pnl) else 0: .4f}")


def test(series, horizon, threshold):
    path = DATA / f"{series}_minute.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fetch first.")
    data, features = prepare(pd.read_csv(path), horizon)
    market_time = data.groupby("ticker").end_period_ts.min().sort_values()
    cut = int(len(market_time) * .7)
    training = set(market_time.index[:cut])
    testing = set(market_time.index[cut:])
    train = data[data.ticker.isin(training)]
    hold = data[data.ticker.isin(testing)]
    if len(train) < 50 or len(hold) < 20 or len(testing) < 5:
        raise RuntimeError("Too few observations. Fetch more markets or use shorter horizon.")
    Xtr = train[features].to_numpy()
    Xte = hold[features].to_numpy()
    ytr = train.y.to_numpy()
    print(f"Train: {len(training)} markets, {len(train)} rows; "
          f"holdout: {len(testing)} markets, {len(hold)} rows")
    print("Strategy      | prediction MAE | trades | total net P&L  | avg net per trade")
    evaluate("No edge", np.zeros(len(hold)), hold, threshold)
    evaluate("Momentum", hold.ret_1.to_numpy(), hold, threshold)
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=20)),
        "MLP neural": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=350,
            random_state=42, alpha=5, learning_rate_init=.0005)),
    }
    for name, model in models.items():
        model.fit(Xtr, ytr)
        evaluate(name, model.predict(Xte), hold, threshold)
    print("WARNING: minute-level quotes are not executable fill evidence. "
          "Do not use these P&L figures as live-trading estimates.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--series", default="KX2YRRATE15M")
    f.add_argument("--limit", type=int, default=300)
    t = sub.add_parser("test")
    t.add_argument("--series", default="KX2YRRATE15M")
    t.add_argument("--horizon", type=int, default=3, choices=[2, 3, 5])
    t.add_argument("--threshold", type=float, default=.01,
                   help="extra required predicted edge in dollars (default $0.01)")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.series, a.limit)
    else:
        test(a.series, a.horizon, a.threshold)


if __name__ == "__main__":
    main()
