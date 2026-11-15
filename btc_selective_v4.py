"""BTC V4: selective, fee-aware, read-only paper research.

Uses the V3 feature/timing builder. Trains on past markets, selects a strategy on
validation markets, then scores later test markets. Candle quotes are NOT fills.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from btc_execution_v3 import FEATURES, load, chronological, taker_fee

ROOT=Path(__file__).resolve().parent

def market_quotes(df, side, qty, slip, fee_multiplier):
    """Per-contract quote-based net proceeds (no fill assumptions proven)."""
    if side=="YES":
        entry=df.entry_yes_ask.to_numpy(dtype=float)
        exit_=df.exit_yes_bid.to_numpy(dtype=float)
    else:
        entry=1-df.entry_yes_bid.to_numpy(dtype=float)
        exit_=1-df.exit_yes_ask.to_numpy(dtype=float)
    buy=np.minimum(1.,entry+slip)
    sell=np.maximum(0.,exit_-slip)
    fees=(taker_fee(buy,qty,fee_multiplier)+
          taker_fee(sell,qty,fee_multiplier))
    net=qty*(sell-buy)-fees
    return buy,sell,fees,net

def train_models(train, valid):
    candidates={
        "extra_trees":ExtraTreesRegressor(n_estimators=150,min_samples_leaf=10,
                                          random_state=42,n_jobs=-1),
        "hist_boost":HistGradientBoostingRegressor(max_iter=130,max_leaf_nodes=9,
                                                    l2_regularization=7,random_state=42)}
    scored=[]
    for name,model in candidates.items():
        model.fit(train[FEATURES],train.future_change)
        pred=model.predict(valid[FEATURES])
        scored.append((mean_absolute_error(valid.future_change,pred),name,model))
    scored.sort(key=lambda x:x[0])
    print("Validation forecasting MAE:",{name:round(mae,4) for mae,name,_ in scored})
    print("Selected price model:",scored[0][1])
    return scored[0][2]

def fit_profit_models(train, qty, slip, multiplier):
    """Profit outcome labels from historical future quotes; never used as features."""
    result={}
    for side in ("YES","NO"):
        labels=(market_quotes(train,side,qty,slip,multiplier)[3]>0).astype(int)
        if labels.sum()<15 or (len(labels)-labels.sum())<15:
            print("Skipping classifier for",side,": insufficient class balance.")
            result[side]=None
            continue
        model=RandomForestClassifier(n_estimators=160,min_samples_leaf=14,
                                      class_weight="balanced",random_state=42,n_jobs=-1)
        model.fit(train[FEATURES],labels)
        result[side]=model
    return result

def score_candidates(df, reg, classifiers, qty, slip, multiplier):
    z=df.copy()
    predicted_future=np.clip(z.mid.to_numpy()+reg.predict(z[FEATURES]),0,1)
    # Forecast price edge is measured against TODAY's executable quote; the
    # simulated trade still enters one minute later, so this is imperfect.
    today_buy_yes=z.ask.to_numpy()
    today_buy_no=1-z.bid.to_numpy()
    fee_yes=taker_fee(today_buy_yes,qty,multiplier)/qty
    fee_no=taker_fee(today_buy_no,qty,multiplier)/qty
    z["edge_yes"]=predicted_future-today_buy_yes-2*slip-fee_yes
    z["edge_no"]=(1-predicted_future)-today_buy_no-2*slip-fee_no
    for side in ("YES","NO"):
        classifier=classifiers.get(side)
        if classifier is None:
            z[f"p_{side.lower()}"]=0.
        else:
            z[f"p_{side.lower()}"]=classifier.predict_proba(z[FEATURES])[:,1]
    return z

def simulate(df, qty, slip, multiplier, edge_min, probability_min):
    """Only the first qualifying signal per market; no overlapping holdings."""
    rows=[]
    for ticker,part in df.groupby("ticker",sort=False):
        part=part.sort_values("end_period_ts")
        for _,r in part.iterrows():
            candidates=[]
            if r.edge_yes >= edge_min and r.p_yes >= probability_min:
                candidates.append(("YES",r.edge_yes,r.p_yes))
            if r.edge_no >= edge_min and r.p_no >= probability_min:
                candidates.append(("NO",r.edge_no,r.p_no))
            if not candidates:
                continue
            side,edge,prob=max(candidates,key=lambda item:(item[1]*item[2]))
            buy,sell,fees,net=market_quotes(pd.DataFrame([r]),side,qty,slip,multiplier)
            rows.append({"ticker":ticker,"signal_ts":int(r.end_period_ts),
                         "entry_ts":int(r.entry_ts),"exit_ts":int(r.exit_ts),
                         "side":side,"predicted_edge":float(edge),
                         "profitability_probability":float(prob),
                         "entry_ask":float(buy[0]),"exit_bid":float(sell[0]),
                         "contracts":qty,"fees":float(fees[0]),
                         "proxy_net":float(net[0])})
            break
    return pd.DataFrame(rows)

def stats(trades, markets):
    if trades.empty:return {"trades":0,"markets":markets,"net":0.,"mean":0.,"win":0.}
    return {"trades":len(trades),"markets":markets,
            "net":float(trades.proxy_net.sum()),
            "mean":float(trades.proxy_net.mean()),
            "win":float((trades.proxy_net>0).mean())}

def choose_threshold(val, reg, clf, qty, slip, multiplier, min_trades):
    pred=score_candidates(val,reg,clf,qty,slip,multiplier)
    grid=[(e,p) for e in (.03,.06,.10,.15,.20) for p in (.50,.60,.70,.80)]
    rows=[]
    for e,p in grid:
        trades=simulate(pred,qty,slip,multiplier,e,p)
        s=stats(trades,val.ticker.nunique())
        rows.append((e,p,s))
    viable=[r for r in rows if r[2]["trades"]>=min_trades]
    if not viable:
        print(f"No threshold meets minimum {min_trades} validation trades.")
        return None
    # Conservative gate: only select if aggregate validation net is positive.
    best=max(viable,key=lambda r:(r[2]["mean"],r[2]["net"]))
    e,p,s=best
    print(f"Validation threshold: edge={e:.2f}, probability={p:.2f}; "
          f"trades={s['trades']}, net=${s['net']:.2f}, mean=${s['mean']:.4f}")
    if s["net"]<=0 or s["mean"]<=0:
        print("No positive validation candidate. POLICY: NO TRADE on test set.")
        return None
    return e,p

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series",default="KXBTC15M")
    ap.add_argument("--hold",type=int,default=2)
    ap.add_argument("--contracts",type=int,default=1)
    ap.add_argument("--slippage",type=float,default=.01)
    ap.add_argument("--fee-multiplier",type=float,default=1.)
    ap.add_argument("--min-validation-trades",type=int,default=15)
    args=ap.parse_args()
    if not 1<=args.hold<=8 or args.contracts<1 or args.slippage<0 or args.fee_multiplier<0:
        ap.error("Require hold 1..8, positive contracts, nonnegative slippage/fee multiplier")
    if args.min_validation_trades<1:ap.error("Need at least one validation trade")
    df=load(args.series,args.hold)
    train,val,test=chronological(df)
    model=train_models(train,val)
    classifiers=fit_profit_models(train,args.contracts,args.slippage,args.fee_multiplier)
    choice=choose_threshold(val,model,classifiers,args.contracts,args.slippage,
                            args.fee_multiplier,args.min_validation_trades)
    output=ROOT/"reports"/f"{args.series}_v4_selective_trades.csv"
    output.parent.mkdir(exist_ok=True)
    if choice is None:
        pd.DataFrame(columns=["ticker","signal_ts","entry_ts","exit_ts","side",
                 "predicted_edge","profitability_probability","entry_ask","exit_bid",
                 "contracts","fees","proxy_net"]).to_csv(output,index=False)
        print("Test not traded: no validation-supported strategy.")
        return
    scored=score_candidates(test,model,classifiers,args.contracts,args.slippage,
                            args.fee_multiplier)
    trades=simulate(scored,args.contracts,args.slippage,args.fee_multiplier,*choice)
    trades.to_csv(output,index=False)
    s=stats(trades,test.ticker.nunique())
    print(f"UNTOUCHED TEST: markets={s['markets']} trades={s['trades']} "
          f"net=${s['net']:.2f}, per_trade=${s['mean']:.4f}, win_rate={s['win']:.1%}")
    print("Saved",output)
    print("QUOTE-ONLY RESEARCH: no verified fills, queue depth, latency, or actual order fees.")

if __name__=="__main__":main()
