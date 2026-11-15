"""BTC 15m V3: read-only, non-overlapping, fee-aware quote-proxy backtest.

Candle quotes are NOT proof of executable volume/fills. Never places orders.
Train/validation/test splits by market window, no data crossing a market boundary.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
FEATURES = ["mid", "spread", "age", "volume", "ret1", "ret2", "ret3",
            "volatility3", "spread3", "volume3", "bid_change", "ask_change"]

def load(series, hold):
    file = HERE / "data" / f"{series}_minute.csv"
    if not file.exists():
        raise FileNotFoundError(f"{file} missing; run kalshi_lab.py fetch --series {series} first.")
    d = pd.read_csv(file)
    for col in ("bid", "ask", "volume", "end_period_ts"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["ticker", "bid", "ask", "volume", "end_period_ts"])
    d = d[(d.bid >= 0) & (d.ask <= 1) & (d.bid <= d.ask)].copy()
    d = d.sort_values(["ticker", "end_period_ts"]).drop_duplicates(["ticker", "end_period_ts"])
    d["mid"] = (d.bid + d.ask) / 2
    d["spread"] = d.ask - d.bid
    g = d.groupby("ticker", sort=False)
    d["age"] = g.cumcount()
    for lag in (1, 2, 3):
        d[f"ret{lag}"] = d.mid - g.mid.shift(lag)
    d["volatility3"] = g.mid.transform(lambda x: x.diff().rolling(3).std())
    d["spread3"] = g.spread.transform(lambda x: x.rolling(3).mean())
    d["volume3"] = g.volume.transform(lambda x: x.rolling(3).mean())
    d["bid_change"] = d.bid - g.bid.shift(1)
    d["ask_change"] = d.ask - g.ask.shift(1)
    # Signal at the close of minute t; hypothetical entry at t+1,
    # liquidation at t+1+hold. Never same-timestamp entry and exit.
    d["entry_ts"] = g.end_period_ts.shift(-1)
    d["exit_ts"] = g.end_period_ts.shift(-(hold + 1))
    d["entry_yes_ask"] = g.ask.shift(-1)
    d["entry_yes_bid"] = g.bid.shift(-1)
    d["exit_yes_bid"] = g.bid.shift(-(hold + 1))
    d["exit_yes_ask"] = g.ask.shift(-(hold + 1))
    d["future_mid"] = g.mid.shift(-(hold + 1))
    d["future_change"] = d.future_mid - d.mid
    valid = ((d.entry_ts - d.end_period_ts == 60) &
             (d.exit_ts - d.end_period_ts == 60 * (hold + 1)))
    d = d.loc[valid].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES + ["future_change", "entry_yes_ask",
                                    "entry_yes_bid", "exit_yes_ask", "exit_yes_bid"])
    if len(d) == 0:
        raise RuntimeError("No contiguous quote windows; try --hold 1 or download more candles.")
    return d

def chronological(d):
    tickers = d.groupby("ticker").end_period_ts.min().sort_values().index
    if len(tickers) < 45:
        raise RuntimeError(f"Only {len(tickers)} usable markets; need at least 45.")
    a, b = int(len(tickers)*.6), int(len(tickers)*.8)
    train, valid, test = [d[d.ticker.isin(tickers[l:r])].copy()
                          for l, r in ((0,a), (a,b), (b,len(tickers)))]
    print(f"Markets: train={a}, validation={b-a}, untouched_test={len(tickers)-b}")
    return train, valid, test

def regressors():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=20)),
        "extra_trees": ExtraTreesRegressor(n_estimators=160, min_samples_leaf=9,
                                          random_state=42, n_jobs=-1),
        "hist_boost": HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=8,
                                                    l2_regularization=5,random_state=42)
    }

def taker_fee(price, contracts, multiplier=1):
    """July 2026 general formula; $0.0001 centicent rounding per order.
    Validate the current specific series multiplier before live use.
    """
    raw = .07 * multiplier * contracts * price * (1-price)
    return np.ceil(raw * 10000 - 1e-10)/10000

def evaluate(trades, test, pred, hold, threshold, contracts, slippage, multiplier):
    data=test.copy()
    data["pred"]=pred
    records=[]
    for ticker, group in data.groupby("ticker", sort=False):
        # One position per market; no overlapping trades or hidden borrowing.
        # Signals can only be evaluated at the close of an observed candle.
        group=group.sort_values("end_period_ts")
        positions = group[group.pred.abs() >= (threshold + group.spread)]
        if positions.empty:
            continue
        # take the first qualifying signal only, limiting selection bias & inventory
        row=positions.iloc[0]
        side="YES" if row.pred>0 else "NO"
        if side=="YES":
            buy=float(row.entry_yes_ask)
            sell=float(row.exit_yes_bid)
        else:
            buy=float(1-row.entry_yes_bid)
            sell=float(1-row.exit_yes_ask)
        buy=min(1,buy+slippage)
        sell=max(0,sell-slippage)
        fees=float(taker_fee(buy,contracts,multiplier)+taker_fee(sell,contracts,multiplier))
        gross=contracts*(sell-buy)
        net=gross-fees
        records.append({"ticker":ticker,"signal_ts":int(row.end_period_ts),
                        "entry_ts":int(row.entry_ts),"exit_ts":int(row.exit_ts),
                        "side":side,"predicted_change":float(row.pred),
                        "buy_ask":buy,"sell_bid":sell,"contracts":contracts,
                        "gross":gross,"fees":fees,"net":net})
    out=pd.DataFrame(records)
    print(f"{trades:12}: {len(out):4} trades, net=${out.net.sum() if len(out) else 0:.2f}, "
          f"mean=${out.net.mean() if len(out) else 0:.4f}, "
          f"win_rate={100*(out.net.gt(0).mean() if len(out) else 0):.1f}%")
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KXBTC15M")
    p.add_argument("--hold",type=int,default=2,help="Full minutes held AFTER the following-minute entry")
    p.add_argument("--threshold",type=float,default=.06,
                   help="Predicted midpoint movement beyond observed spread (dollars)")
    p.add_argument("--slippage",type=float,default=.01,
                   help="Extra adverse price movement on EACH side ($/contract)")
    p.add_argument("--contracts",type=int,default=1)
    p.add_argument("--fee-multiplier",type=float,default=1.)
    args=p.parse_args()
    if args.hold<1 or args.hold>8 or args.threshold<0 or args.slippage<0:
        p.error("Require 1<=hold<=8, threshold>=0, slippage>=0")
    if args.contracts<1 or args.fee_multiplier<0:
        p.error("Require contracts>=1 and fee multiplier>=0")
    d=load(args.series,args.hold)
    train,valid,test=chronological(d)
    estimates={}
    validation_mae={}
    for name, model in regressors().items():
        model.fit(train[FEATURES],train.future_change)
        validation_mae[name]=mean_absolute_error(valid.future_change,model.predict(valid[FEATURES]))
        estimates[name]=model.predict(test[FEATURES])
    chosen=min(validation_mae,key=validation_mae.get)
    print("Validation MAE:", {k:round(v,4) for k,v in validation_mae.items()})
    print("Selected BEFORE test:",chosen)
    print("Test model MAE:",round(mean_absolute_error(test.future_change,estimates[chosen]),4),
          "No-change MAE:",round(mean_absolute_error(test.future_change,np.zeros(len(test))),4))
    path=HERE/"reports"
    path.mkdir(exist_ok=True)
    selected=evaluate(chosen,test,estimates[chosen],args.hold,args.threshold,
                      args.contracts,args.slippage,args.fee_multiplier)
    selected.to_csv(path/f"{args.series}_v3_trades.csv",index=False)
    print(f"Report: {path / (args.series + '_v3_trades.csv')}")
    print("SIMULATED QUOTE PROXY ONLY: unknown fill availability, book depth, latency and fees. "
          "Never evidence of live profitability.")

if __name__=="__main__":
    main()
