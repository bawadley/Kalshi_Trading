"""Read-only Kalshi order-book poller, depth feature builder and snapshot stress replay.

REST snapshots are NOT event-level historical books or evidence of fills.
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "depth"
API = "https://external-api.kalshi.com/trade-api/v2"

def parse_levels(v):
    """Kalshi REST shows YES and NO bids, ascending in price."""
    return sorted([(float(p), float(q)) for p, q in (v or [])
                   if float(q) > 0], reverse=True)

def normalize(raw):
    book = raw.get("orderbook_fp", raw)
    if "yes_dollars" in book:
        return parse_levels(book["yes_dollars"]), parse_levels(book.get("no_dollars", []))
    if "yes_dollars_fp" in book:
        return parse_levels(book["yes_dollars_fp"]), parse_levels(book.get("no_dollars_fp", []))
    raise ValueError("Unsupported book schema: expected yes_dollars/no_dollars")

def snapshot(ticker, session):
    r = session.get(f"{API}/markets/{ticker}/orderbook", timeout=20)
    r.raise_for_status()
    yes, no = normalize(r.json())
    return {"ts_ms":int(time.time()*1000),"ticker":ticker,"yes_bids":yes,"no_bids":no,
            "source":"kalshi_rest_snapshot", "valid":bool(yes and no)}

def collect(ticker, seconds, interval):
    if seconds < 0 or interval < 1:
        raise ValueError("seconds must be nonnegative and interval >= 1")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{ticker}.jsonl"
    session=requests.Session()
    print(f"Recording READ-ONLY snapshots to {path}; Ctrl+C to stop.")
    deadline=time.monotonic()+seconds
    try:
        while True:
            start=time.monotonic()
            try:
                snap=snapshot(ticker,session)
                with path.open("a",encoding="utf-8") as f:
                    f.write(json.dumps(snap,separators=(",",":"))+"\n")
                print(datetime.fromtimestamp(snap["ts_ms"]/1000,timezone.utc).isoformat(),
                      "levels",len(snap["yes_bids"]),len(snap["no_bids"]))
            except (requests.RequestException, ValueError) as err:
                print("Snapshot missed:",err)
            if seconds and time.monotonic()>=deadline:break
            if not seconds or start < deadline:
                time.sleep(max(0,interval-(time.monotonic()-start)))
            else:break
    except KeyboardInterrupt:
        print("Recording stopped.")
    return path

def metrics(record, levels=5):
    yes=parse_levels(record["yes_bids"])
    no=parse_levels(record["no_bids"])
    if not yes or not no:
        return None
    bid= yes[0][0]
    ask=1-no[0][0]
    if ask<bid-1e-6:
        return None
    spread=ask-bid
    top_buy=yes[0][1]
    top_sell=no[0][1]
    result={"ticker":record["ticker"],"ts_ms":record["ts_ms"],"bid":bid,
            "ask":ask,"mid":(bid+ask)/2,"spread":spread,"best_bid_qty":top_buy,
            "best_ask_qty":top_sell,"book_levels_yes":len(yes),"book_levels_no":len(no),
            "top_imbalance":(top_buy-top_sell)/(top_buy+top_sell),
            "microprice":(bid*top_sell+ask*top_buy)/(top_buy+top_sell),
            "source":record.get("source","unknown")}
    result["microprice_minus_mid"]=result["microprice"]-result["mid"]
    for n in (1,3,5,10,20):
        b=yes[:n]; a=no[:n]
        bqty=sum(q for _,q in b)
        aqty=sum(q for _,q in a)
        result[f"bid_depth_{n}"]=bqty
        result[f"ask_depth_{n}"]=aqty
        result[f"imbalance_{n}"]=(bqty-aqty)/(bqty+aqty) if bqty+aqty else 0.
        result[f"bid_vwap_{n}"]=sum(p*q for p,q in b)/bqty if bqty else np.nan
        result[f"ask_vwap_{n}"]=1-sum(p*q for p,q in a)/aqty if aqty else np.nan
    result["bid_wall_2500"]=int(any(q>=2500 for _,q in yes))
    result["ask_wall_2500"]=int(any(q>=2500 for _,q in no))
    result["max_bid_wall"]=max(q for _,q in yes)
    result["max_ask_wall"]=max(q for _,q in no)
    result["bid_concentration"]=result["max_bid_wall"]/sum(q for _,q in yes)
    result["ask_concentration"]=result["max_ask_wall"]/sum(q for _,q in no)
    return result

def load(path):
    records=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda x:(x["ticker"],x["ts_ms"]))
    return records

def build(path, out=None):
    records=load(path)
    rows=[m for r in records if (m:=metrics(r)) is not None]
    if not rows:raise ValueError("No usable two-sided snapshots")
    d=pd.DataFrame(rows).sort_values(["ticker","ts_ms"])
    g=d.groupby("ticker",sort=False)
    d["mid_change"]=g.mid.diff()
    d["spread_change"]=g.spread.diff()
    d["imbalance_change"]=g.imbalance_5.diff()
    d["seconds_from_prev"]=g.ts_ms.diff()/1000
    for period in (3,5,10,20):
        d[f"realized_vol_{period}"]=g.mid.transform(lambda x:x.diff().rolling(period).std())
        d[f"abs_return_{period}"]=g.mid.transform(lambda x:x.diff().abs().rolling(period).sum())
        d[f"mid_return_{period}"]=d.mid-g.mid.shift(period)
        d[f"imbalance_mean_{period}"]=g.imbalance_5.transform(lambda x:x.rolling(period).mean())
        d[f"spread_mean_{period}"]=g.spread.transform(lambda x:x.rolling(period).mean())
    d["bid_wall_persistence_5"]=g.bid_wall_2500.transform(lambda x:x.rolling(5).sum())
    d["ask_wall_persistence_5"]=g.ask_wall_2500.transform(lambda x:x.rolling(5).sum())
    dest=Path(out) if out else ROOT/"reports"/"depth_features.csv"
    dest.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(dest,index=False)
    print(f"Wrote {len(d)} quote states, {len(d.columns)} columns to {dest}")
    print("These are observed REST snapshots, not full order-book events.")
    return d

def stress(path, tick=.01, hold=120):
    """Conservative two-sided quote-crossing upper-bound proxy, NOT fills."""
    d=build(path)
    p=[]; entries=0; complete=0; unresolved=0
    for ticker,part in d.groupby("ticker"):
        part=part.sort_values("ts_ms").reset_index(drop=True)
        i=0
        while i+1<len(part):
            row=part.iloc[i]
            if row.spread<3*tick:
                i+=1;continue
            our_bid=round(float(row.bid+tick),4)
            our_ask=round(float(row.ask-tick),4)
            until=row.ts_ms+hold*1000
            future=part.iloc[i+1:]
            future=future[future.ts_ms<=until]
            crosses=future[future.ask <= our_bid-tick+1e-9]
            if crosses.empty:
                i+=1;continue
            entries+=1
            entry_idx=crosses.index[0]
            exits=future[future.index>entry_idx]
            crossed=exits[exits.bid>=our_ask+tick-1e-9]
            if not crossed.empty:
                exit_idx=crossed.index[0]
                p.append(our_ask-our_bid)
                complete+=1
                i=exit_idx+1
            elif not exits.empty:
                last=exits.iloc[-1]
                p.append(float(last.bid)-our_bid)
                i=int(last.name)+1
            else:
                unresolved+=1
                i=entry_idx+1
    print(f"Quote-crossing stress proxy: assumed entries={entries}, paired={complete}, "
          f"unresolved={unresolved}, scored={len(p)}, gross_proxy={sum(p):.4f} dollars")
    print("No trade prints, maker fees, queue priority or actual fill modeling. "
          "Gross proxy is NOT an executable strategy result.")

def main():
    a=argparse.ArgumentParser(description=__doc__)
    sub=a.add_subparsers(dest="cmd",required=True)
    c=sub.add_parser("collect")
    c.add_argument("--ticker",required=True)
    c.add_argument("--seconds",type=int,default=120)
    c.add_argument("--interval",type=float,default=5.0)
    b=sub.add_parser("features")
    b.add_argument("--input",required=True)
    b.add_argument("--output")
    s=sub.add_parser("stress")
    s.add_argument("--input",required=True)
    s.add_argument("--hold",type=int,default=120)
    x=a.parse_args()
    if x.cmd=="collect":collect(x.ticker,x.seconds,x.interval)
    if x.cmd=="features":build(x.input,x.output)
    if x.cmd=="stress":stress(x.input,hold=x.hold)

if __name__=="__main__":main()
