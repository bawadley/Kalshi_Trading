"""Backfill V7.4 matching spot candles and market metadata from V7.2 features.

Reads only recording timestamps/tickers; read-only public REST queries.
Merges with existing V5 CSVs without discarding previously downloaded rows.
Coinbase BTC-USD remains a proxy, NOT the Kalshi settlement index.
"""
import argparse
from pathlib import Path
import time
import pandas as pd
from btc_v5_data import fetch_coinbase, kalshi_meta

ROOT=Path(__file__).resolve().parent


def merge_csv(path,new,key):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=pd.read_csv(path)
        # Preserve the most recently fetched value for identical timestamp/ticker.
        new=pd.concat([old,new],ignore_index=True)
    new=(new.drop_duplicates(subset=[key],keep="last")
            .sort_values(key).reset_index(drop=True))
    new.to_csv(path,index=False)
    return len(new)


def run(features,spot_file,meta_file):
    d=pd.read_csv(features,usecols=["ticker","second_ts"])
    d.second_ts=pd.to_numeric(d.second_ts,errors="coerce")
    d=d.dropna(subset=["ticker","second_ts"])
    if d.empty: raise RuntimeError("No timestamped V7.2 rows to backfill")
    start=int(d.second_ts.min())-600
    end=int(d.second_ts.max())+120
    names=sorted(d.ticker.unique())
    print(f"Recorded tickers={len(names)}; epoch range={start}..{end}",flush=True)
    metadata=kalshi_meta(names)
    if metadata.empty:
        print("WARNING: no matching Kalshi market metadata returned")
    else:
        count=merge_csv(meta_file,metadata,"ticker")
        print(f"Metadata rows stored={count}; fetched={len(metadata)}")
    print("Fetching Coinbase BTC/USD history that matches WebSocket timestamps...")
    spot=fetch_coinbase(start,end)
    if spot.empty:
        print("WARNING: no overlapping Coinbase spot candles returned")
    else:
        count=merge_csv(spot_file,spot,"available_ts")
        print(f"Coinbase rows stored={count}; fetched={len(spot)}")
    print("Complete. Run v74_orderflow.py again.")
    print("Coinbase prices are NOT Kalshi settlement-index prices.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features",default=str(ROOT/"data/v72/btc_1s_features.csv"))
    p.add_argument("--spot",default=str(ROOT/"data/v5/coinbase_btcusd_1m.csv"))
    p.add_argument("--metadata",default=str(ROOT/"data/v5/KXBTC15M_metadata.csv"))
    a=p.parse_args()
    run(a.features,a.spot,a.metadata)


if __name__=="__main__":
    main()
