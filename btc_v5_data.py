"""V5: read-only BTC/USD historical 1-minute candles + Kalshi market metadata.

Coinbase spot is a FEATURE PROXY, not the Kalshi settlement index.
No fabricated strike; missing metadata remains missing. Output files are gitignored.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"data"/"v5"
COINBASE="https://api.exchange.coinbase.com/products/BTC-USD/candles"
KALSHI="https://external-api.kalshi.com/trade-api/v2"

def iso(ts):
    return datetime.fromtimestamp(int(ts),timezone.utc).isoformat()

def fetch_coinbase(start,end):
    if end<=start:raise ValueError("end must exceed start")
    session=requests.Session()
    session.headers.update({"User-Agent":"KalshiResearch-v5-readonly/0.1","Accept":"application/json"})
    rows=[]
    # Coinbase caps requests to 300 candles; use 240/minute request spans.
    for a in range(int(start)//60*60,int(end),240*60):
        b=min(a+240*60,int(end))
        res=session.get(COINBASE,params={"start":iso(a),"end":iso(b),"granularity":60},timeout=25)
        res.raise_for_status()
        for r in res.json():
            if len(r)>=6:
                rows.append({"ts":int(r[0]),"spot_low":float(r[1]),"spot_high":float(r[2]),
                             "spot_open":float(r[3]),"spot_close":float(r[4]),
                             "spot_volume":float(r[5])})
        time.sleep(.22)
    out=pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
    # Coinbase timestamps label candle START, so spot_* is only known AFTER +60 seconds.
    out["available_ts"]=out.ts+60
    out=out[(out.available_ts>=start)&(out.available_ts<=end)]
    return out

def kalshi_meta(tickers):
    session=requests.Session()
    rows=[]
    for ticker in tickers:
        try:
            r=session.get(KALSHI+f"/markets/{ticker}",timeout=25)
            r.raise_for_status()
            m=r.json().get("market",{})
            rows.append({"ticker":ticker,"status":m.get("status"),"result":m.get("result"),
                        "open_time":m.get("open_time"),"close_time":m.get("close_time"),
                        "expiration_time":m.get("expiration_time"),
                        "floor_strike":m.get("floor_strike"),
                        "cap_strike":m.get("cap_strike"),
                        "strike_type":m.get("strike_type"),
                        "title":m.get("title"),"rules_primary":m.get("rules_primary")})
        except requests.RequestException as exc:
            print(f"Metadata unavailable for {ticker}: {exc}")
        time.sleep(.08)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KXBTC15M")
    p.add_argument("--days",type=int,default=14)
    p.add_argument("--metadata-only",action="store_true")
    a=p.parse_args()
    if not 1<=a.days<=180:p.error("--days must be 1..180")
    source=ROOT/"data"/f"{a.series}_minute.csv"
    if not source.exists():raise FileNotFoundError(f"Run kalshi_lab.py fetch --series {a.series} first")
    d=pd.read_csv(source,usecols=["ticker","end_period_ts"])
    d.end_period_ts=pd.to_numeric(d.end_period_ts,errors="coerce")
    d=d.dropna().sort_values("end_period_ts")
    end=int(d.end_period_ts.max())+120
    start=max(int(d.end_period_ts.min())-120,end-a.days*86400)
    tickers=sorted(d.loc[d.end_period_ts>=start,"ticker"].unique())
    if not tickers:raise RuntimeError("No markets in selected history")
    OUT.mkdir(parents=True,exist_ok=True)
    meta=kalshi_meta(tickers)
    meta.to_csv(OUT/f"{a.series}_metadata.csv",index=False)
    print(f"Saved metadata for {len(meta)}/{len(tickers)} markets")
    print("Numeric floor strike available:",
          int(pd.to_numeric(meta.get("floor_strike",pd.Series(dtype=float)),errors="coerce").notna().sum()))
    if not a.metadata_only:
        spot=fetch_coinbase(start,end)
        if spot.empty:raise RuntimeError("No Coinbase BTC/USD candles in date interval")
        spot.to_csv(OUT/"coinbase_btcusd_1m.csv",index=False)
        print(f"Saved {len(spot)} Coinbase candles (with availability timestamps)")
    print("WARNING: Coinbase BTC/USD is NOT Kalshi's settlement reference.")

if __name__=="__main__":main()
