"""V7 offline order-book reconstruction and conservative depth-based taker replay.

Only recorded WS snapshots/deltas; rejects sequence gaps and missing snapshots.
Execution is hypothetical even if displayed depth exists.
"""
import argparse
import json
from decimal import Decimal
from pathlib import Path
from collections import defaultdict

def dec(x):return Decimal(str(x))

class Book:
    def __init__(self):
        self.sides={"yes":{},"no":{}}
        self.valid=False
        self.seq=None
        self.clock=None
        self.errors=0

    def snapshot(self,msg,seq):
        sides={}
        for side in ("yes","no"):
            levels=msg.get(f"{side}_dollars_fp",[])
            sides[side]={dec(p):dec(q) for p,q in levels if dec(q)>0}
        self.sides=sides
        self.seq=seq
        self.clock=msg.get("clock")
        self.valid=True

    def delta(self,msg,seq):
        if not self.valid or self.seq is None or seq is None or seq!=self.seq+1:
            self.valid=False
            self.errors+=1
            return False
        side=msg.get("side")
        if side not in self.sides:
            self.valid=False;self.errors+=1;return False
        price=dec(msg["price_dollars"])
        size=self.sides[side].get(price,Decimal(0))+dec(msg["delta_fp"])
        if size<0:
            self.valid=False;self.errors+=1;return False
        if size==0:self.sides[side].pop(price,None)
        else:self.sides[side][price]=size
        self.seq=seq
        self.clock=msg.get("clock")
        return True

    def top(self):
        if not self.valid or not self.sides["yes"] or not self.sides["no"]:
            return None
        yes_bid=max(self.sides["yes"])
        no_bid=max(self.sides["no"])
        return {"yes_bid":yes_bid,"yes_ask":Decimal(1)-no_bid,
                "yes_bid_qty":self.sides["yes"][yes_bid],
                "yes_ask_qty":self.sides["no"][no_bid]}

    def taker(self,action,qty):
        """Hypothetical immediate execution against displayed aggregate depth.

        buy_yes uses NO bids, sell_yes uses YES bids. Complements are exact.
        Returns None if insufficient displayed depth; never partially books a trade.
        """
        side="no" if action=="buy_yes" else "yes"
        if action not in ("buy_yes","sell_yes") or not self.valid:return None
        levels=sorted(self.sides[side].items(),key=lambda x:x[0],reverse=True)
        need=dec(qty)
        spent=Decimal(0)
        for price,size in levels:
            taken=min(size,need)
            tradeprice=Decimal(1)-price if side=="no" else price
            spent+=tradeprice*taken
            need-=taken
            if need<=0:break
        if need>0:return None
        return spent / dec(qty)

def parse_file(path):
    books=defaultdict(Book)
    market_sid={}
    trades=[]
    ignored=0
    by_market=defaultdict(list)
    for ln,line in enumerate(Path(path).open(encoding="utf-8"),1):
        if not line.strip():continue
        row=json.loads(line)
        event=row.get("event",{})
        typ=event.get("type")
        if row.get("type") in ("disconnect","capture_start"):
            for b in books.values():b.valid=False
            market_sid.clear()
            continue
        msg=event.get("msg") or {}
        ticker=msg.get("market_ticker")
        seq=event.get("seq")
        sid=event.get("sid")
        if typ=="orderbook_snapshot" and ticker:
            book=books[ticker]
            book.snapshot(msg,seq)
            market_sid[ticker]=sid
        elif typ=="orderbook_delta" and ticker:
            book=books[ticker]
            if sid!=market_sid.get(ticker) or not book.delta(msg,seq):
                book.valid=False;ignored+=1
        elif typ=="trade":
            trades.append({"ts_ms":row.get("local_ts_ms"),"msg":msg})
        else:continue
        book=books[ticker] if ticker else None
        if book:
            quote=book.top()
            if quote:
                by_market[ticker].append({"ts_ms":row.get("local_ts_ms"),
                                         "book":book.sides["yes"].copy(),
                                         "no_book":book.sides["no"].copy(),
                                         **quote})
    return by_market,trades,ignored,sum(b.errors for b in books.values())

def price_from_snapshot(entry,action,qty):
    b=Book()
    b.sides={"yes":entry["book"],"no":entry["no_book"]}
    b.valid=True
    return b.taker(action,qty)

def replay(path,qty=1,hold_s=120,slippage=.01,fee_rate=.07):
    """Illustrative immediate entry and timed exit benchmark, not an alpha strategy."""
    books,trades,ignored,errors=parse_file(path)
    records=[]
    for ticker,states in books.items():
        # no signal model: illustrates conservative executable-depth feasibility.
        for i,first in enumerate(states):
            after=first["ts_ms"]+hold_s*1000
            future=next((v for v in states[i+1:] if v["ts_ms"]>=after),None)
            if future is None:break
            if future["ts_ms"]-after>5000:continue
            buy=price_from_snapshot(first,"buy_yes",qty)
            sell=price_from_snapshot(future,"sell_yes",qty)
            if buy is None or sell is None:continue
            buy=min(Decimal(1),buy+dec(slippage))
            sell=max(Decimal(0),sell-dec(slippage))
            fees=(dec(fee_rate)*dec(qty)*
                  (buy*(1-buy)+sell*(1-sell)))
            net=(sell-buy)*dec(qty)-fees
            records.append({"ticker":ticker,"entry_ms":first["ts_ms"],
                            "exit_ms":future["ts_ms"],"buy":float(buy),
                            "sell":float(sell),"net_proxy":float(net)})
            # one demo trade per market
            break
    print("Markets:",len(books),"valid-trade events:",len(trades),
          "ignored deltas:",ignored,"book errors:",errors)
    print("Demonstration trades:",len(records),
          "net quote proxy:",round(sum(x["net_proxy"] for x in records),4))
    print("WARNING: Depth is displayed, not guaranteed. Queue, latency, fees, "
          "order replenishment, adverse selection and real fills remain unknown.")
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--contracts",type=int,default=1)
    p.add_argument("--hold-seconds",type=int,default=120)
    p.add_argument("--slippage",type=float,default=.01)
    a=p.parse_args()
    if a.contracts<1 or a.hold_seconds<1 or a.slippage<0:p.error("Invalid size/hold/slippage")
    replay(a.input,a.contracts,a.hold_seconds,a.slippage)
if __name__=="__main__":main()
