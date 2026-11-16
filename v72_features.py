"""V7.2 stream raw Kalshi JSONL into 1-second order-flow training rows.

Processes each capture independently; never joins across reconnects. Labels at
5/15/30/60 seconds are made only if *every* intermediate second has a valid
book. Writes compact CSV and a per-file integrity audit. Raw files untouched.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from v7_replay import Book, dec

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"v7"
OUT=ROOT/"data"/"v72"
HORIZONS=(5,15,30,60)

FEATURES=["source_file","ticker","second_ts","local_end_ts_ms","quote_age_ms",
          "yes_bid","yes_ask","mid","spread",
          "yes_bid_qty","yes_ask_qty","yes_depth5","no_depth5",
          "imbalance5","microprice","book_deltas","trade_count",
          "trade_volume","yes_taker_volume","no_taker_volume","unknown_taker_volume",
          "book_valid","book_epoch","segment"]
TARGETS=[f"future_mid_change_{h}s" for h in HORIZONS]
FIELDS=FEATURES+TARGETS


def depth_levels(book,side,n):
    vals=sorted(book.sides[side].items(),reverse=True)[:n]
    return sum(float(q) for _,q in vals)


def make_row(book,bucket,counts,file_name,ticker,last_quote_ms,epoch):
    top=book.top()
    if not top:
        return None
    bid=float(top["yes_bid"])
    ask=float(top["yes_ask"])
    bid_qty=float(top["yes_bid_qty"])
    ask_qty=float(top["yes_ask_qty"])
    yes_depth=depth_levels(book,"yes",5)
    no_depth=depth_levels(book,"no",5)
    denom=yes_depth+no_depth
    microdenom=bid_qty+ask_qty
    return {"source_file":file_name,"ticker":ticker,"second_ts":bucket,
            "local_end_ts_ms":bucket*1000+999,
            "quote_age_ms":max(0,bucket*1000+999-last_quote_ms),
            "yes_bid":bid,"yes_ask":ask,"mid":(bid+ask)/2,
            "spread":ask-bid,"yes_bid_qty":bid_qty,"yes_ask_qty":ask_qty,
            "yes_depth5":yes_depth,"no_depth5":no_depth,
            "imbalance5":((yes_depth-no_depth)/denom if denom else None),
            "microprice":((ask*bid_qty+bid*ask_qty)/microdenom
                          if microdenom else None),
            "book_deltas":counts["orderbook_delta"],"trade_count":counts["trade"],
            "trade_volume":counts["volume"],
            "yes_taker_volume":counts["yes_taker_volume"],
            "no_taker_volume":counts["no_taker_volume"],
            "unknown_taker_volume":counts["unknown_taker_volume"],
            "book_valid":1,"book_epoch":epoch}


def extract_file(path):
    """Return rows + counts for a single capture. O(events + seconds) memory."""
    book=Book()
    audit=Counter()
    rows=[]
    first_ticker=None
    bucket=None
    counts=Counter()
    last_quote_ms=None
    epoch=0
    source=Path(path).name
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                item=json.loads(raw)
                if not isinstance(item,dict):
                    raise ValueError("not an object")
            except (ValueError,TypeError):
                audit["malformed_json"]+=1
                book.valid=False
                continue
            event=item.get("event")
            if not isinstance(event,dict):
                kind=item.get("type")
                if kind in ("capture_start","disconnect","book_invalid",
                            "reconnect_required","subscription_error"):
                    if kind!="capture_start":
                        audit[kind]+=1
                    book.valid=False
                continue
            kind=event.get("type")
            msg=event.get("msg") or {}
            if not isinstance(msg,dict):
                msg={}
            if kind not in ("orderbook_snapshot","orderbook_delta","trade"):
                continue
            ticker=msg.get("market_ticker")
            if first_ticker is None and ticker:
                first_ticker=ticker
            if ticker and first_ticker and ticker!=first_ticker:
                audit["unexpected_ticker"]+=1
                book.valid=False
                continue
            receipt=item.get("local_ts_ms")
            try:
                receipt=int(receipt)
            except (TypeError,ValueError):
                audit["invalid_timestamp"]+=1
                book.valid=False
                continue
            current=receipt//1000
            if bucket is not None and current<bucket:
                audit["nonmonotonic_timestamp"]+=1
                book.valid=False
                continue
            if bucket is not None and current>bucket:
                if book.valid and last_quote_ms is not None:
                    row=make_row(book,bucket,counts,source,first_ticker,last_quote_ms,epoch)
                    if row is not None:rows.append(row)
                # DO NOT fill no-event seconds with stale quotes.
                counts=Counter()
            bucket=current

            if kind=="orderbook_snapshot":
                audit["snapshots"]+=1
                try:
                    book.snapshot(msg,event.get("seq"))
                    if book.seq is None or event.get("sid") is None:
                        raise ValueError("snapshot missing seq/sid")
                    audit["valid_snapshots"]+=1
                    epoch+=1
                    last_quote_ms=receipt
                except (ValueError,TypeError,KeyError):
                    book.valid=False
                    audit["invalid_snapshot"]+=1
            elif kind=="orderbook_delta":
                audit["deltas"]+=1
                try:
                    success=book.delta(msg,event.get("seq"))
                except (ValueError,TypeError,KeyError,ArithmeticError):
                    book.valid=False
                    success=False
                if not success:
                    audit["invalid_deltas"]+=1
                else:
                    last_quote_ms=receipt
                    counts["orderbook_delta"]+=1
            elif kind=="trade":
                audit["trades"]+=1
                counts["trade"]+=1
                try:
                    volume=float(msg.get("count_fp",0))
                    if not math.isfinite(volume) or volume<0:
                        raise ValueError
                except (ValueError,TypeError):
                    volume=0
                    audit["invalid_trade_size"]+=1
                counts["volume"]+=volume
                side=msg.get("taker_outcome_side")
                if side=="yes":
                    counts["yes_taker_volume"]+=volume
                elif side=="no":
                    counts["no_taker_volume"]+=volume
                else:
                    counts["unknown_taker_volume"]+=volume

    if bucket is not None and book.valid and last_quote_ms is not None:
        row=make_row(book,bucket,counts,source,first_ticker,last_quote_ms,epoch)
        if row is not None:rows.append(row)

    audit["valid_seconds"]=len(rows)
    audit["book_errors"]=book.errors
    audit["ticker"]=first_ticker or ""
    audit["source_file"]=source
    return rows,audit


def add_targets(rows):
    if not rows:
        return []
    d=pd.DataFrame(rows)
    d=d.sort_values("second_ts").drop_duplicates("second_ts").reset_index(drop=True)
    # Each continuous second gets the same segment; missing/invalid seconds split.
    d["segment"]=((d.second_ts.diff().fillna(1)!=1) |
                  (d.book_epoch!=d.book_epoch.shift(1))).cumsum().astype(int)
    for h in HORIZONS:
        future=d.mid.shift(-h)
        future_second=d.second_ts.shift(-h)
        future_segment=d.segment.shift(-h)
        valid=((future_second-d.second_ts)==h)&(future_segment==d.segment)
        d[f"future_mid_change_{h}s"]=(future-d.mid).where(valid)
    return d.reindex(columns=FIELDS).to_dict("records")


def run(directory,output,audit_output):
    files=sorted(Path(directory).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files in {directory}")
    output=Path(output)
    audit_output=Path(audit_output)
    output.parent.mkdir(parents=True,exist_ok=True)
    audit_output.parent.mkdir(parents=True,exist_ok=True)
    audit_fields=["source_file","ticker","snapshots","valid_snapshots","deltas",
                  "invalid_deltas","trades","malformed_json","invalid_timestamp",
                  "nonmonotonic_timestamp","unexpected_ticker","invalid_snapshot",
                  "book_invalid","reconnect_required","disconnect",
                  "subscription_error","invalid_trade_size","book_errors",
                  "valid_seconds","labeled_5s","labeled_15s",
                  "labeled_30s","labeled_60s"]
    total_seconds=0
    with output.open("w",newline="",encoding="utf-8") as features_handle,\
         audit_output.open("w",newline="",encoding="utf-8") as audit_handle:
        writer=csv.DictWriter(features_handle,fieldnames=FIELDS)
        writer.writeheader()
        audit_writer=csv.DictWriter(audit_handle,fieldnames=audit_fields)
        audit_writer.writeheader()
        for n,file in enumerate(files,1):
            rows,stats=extract_file(file)
            labeled=add_targets(rows)
            writer.writerows(labeled)
            for h in HORIZONS:
                stats[f"labeled_{h}s"]=sum(
                    1 for row in labeled if pd.notna(row[f"future_mid_change_{h}s"]))
            audit_writer.writerow({k:stats.get(k,0) for k in audit_fields})
            total_seconds+=len(labeled)
            if n%20==0 or n==len(files):
                print(f"Processed {n}/{len(files)} files | {total_seconds:,} one-second rows",
                      flush=True)
    print(f"Features: {output}")
    print(f"Integrity audit: {audit_output}")
    print("Labels NEVER cross capture-file or invalid-second boundaries.")
    print("Local receipt timestamps are NOT exchange-time synchronized.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",default=str(DATA))
    p.add_argument("--output",default=str(OUT/"btc_1s_features.csv"))
    p.add_argument("--audit",default=str(OUT/"btc_integrity.csv"))
    a=p.parse_args()
    run(a.directory,a.output,a.audit)


if __name__=="__main__":
    main()
