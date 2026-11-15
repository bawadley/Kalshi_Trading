"""Bitcoin V6: walk-forward, selective YES/NO/NO-TRADE quote-proxy evaluation.

No order placement. Uses only historical Kalshi minute quotes and V5 spot
features. Simulation quotes are NOT executable fill evidence.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, brier_score_loss
from btc_v5_walkforward import read, BASE_FEATURES
from btc_execution_v3 import taker_fee

ROOT = Path(__file__).resolve().parent
COLUMNS = ["ticker","signal_ts","entry_ts","exit_ts","side","model_edge",
           "profit_probability","entry_ask","exit_bid","fees","net_proxy","fold"]

def dataset(series, hold, max_spot_age):
    # V5 target is the exit quote at t+(1+hold) minutes.
    d = read(series, hold+1, max_spot_age)
    original = pd.read_csv(ROOT/"data"/f"{series}_minute.csv")
    for c in ["end_period_ts","bid","ask"]:
        original[c] = pd.to_numeric(original[c], errors="coerce")
    original = original.dropna(subset=["ticker","end_period_ts","bid","ask"])
    original = original.sort_values(["ticker","end_period_ts"]).drop_duplicates(
        ["ticker","end_period_ts"])
    g = original.groupby("ticker", sort=False)
    original["entry_ts"] = g.end_period_ts.shift(-1)
    original["exit_ts"] = g.end_period_ts.shift(-(hold+1))
    original["entry_yes_ask"] = g.ask.shift(-1)
    original["entry_yes_bid"] = g.bid.shift(-1)
    original["exit_yes_ask"] = g.ask.shift(-(hold+1))
    original["exit_yes_bid"] = g.bid.shift(-(hold+1))
    keep = ["ticker","end_period_ts","entry_ts","exit_ts","entry_yes_ask",
            "entry_yes_bid","exit_yes_ask","exit_yes_bid"]
    d = d.merge(original[keep], on=["ticker","end_period_ts"], how="left",
                validate="one_to_one")
    # No entry/exit at the same timestamp, and reject missing candles.
    d = d[(d.entry_ts-d.end_period_ts==60) &
          (d.exit_ts-d.entry_ts==hold*60)].copy()
    required = BASE_FEATURES+["target","entry_yes_ask","entry_yes_bid",
                               "exit_yes_ask","exit_yes_bid"]
    d=d.replace([np.inf,-np.inf], np.nan).dropna(subset=required)
    d=d[(d.entry_yes_bid<=d.entry_yes_ask) &
        (d.exit_yes_bid<=d.exit_yes_ask)]
    return d

def p_and_l(d, side, contracts, slippage, fee_multiplier):
    if side=="YES":
        buy=d.entry_yes_ask.to_numpy(dtype=float)
        sell=d.exit_yes_bid.to_numpy(dtype=float)
    else:
        buy=1-d.entry_yes_bid.to_numpy(dtype=float)
        sell=1-d.exit_yes_ask.to_numpy(dtype=float)
    buy=np.minimum(1,buy+slippage)
    sell=np.maximum(0,sell-slippage)
    fees=(taker_fee(buy,contracts,fee_multiplier)+
          taker_fee(sell,contracts,fee_multiplier))
    return buy,sell,fees,contracts*(sell-buy)-fees

def model_pair(train, features, contracts, slippage, fee_multiplier):
    price=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=9,
                                        l2_regularization=8,random_state=42)
    price.fit(train[features], train.target)
    classifiers={}
    for side in ("YES","NO"):
        label=(p_and_l(train,side,contracts,slippage,fee_multiplier)[3]>0).astype(int)
        if label.sum()<20 or len(label)-label.sum()<20:
            classifiers[side]=None
            continue
        m=RandomForestClassifier(n_estimators=120,min_samples_leaf=15,
                                 class_weight="balanced",n_jobs=-1,random_state=42)
        m.fit(train[features],label)
        classifiers[side]=m
    return price,classifiers

def score(d,price,classifiers,features,contracts,slippage,fee_multiplier):
    s=d.copy()
    pred=np.clip(s.mid.to_numpy()+price.predict(s[features]),0,1)
    # Conservative signal-time proxy. Future execution prices are not features.
    buy_yes=s.ask.to_numpy()
    buy_no=1-s.bid.to_numpy()
    for side,buy,future in (("YES",buy_yes,pred),("NO",buy_no,1-pred)):
        expected_roundtrip_fee=2*taker_fee(buy,contracts,fee_multiplier)/contracts
        s[f"edge_{side}"]=future-buy-2*slippage-expected_roundtrip_fee
        classifier=classifiers[side]
        s[f"prob_{side}"]=(classifier.predict_proba(s[features])[:,1]
                             if classifier is not None else 0.)
    return s

def select_one_per_market(scored,edge_cut,prob_cut,contracts,slippage,
                          fee_multiplier,fold=0):
    chosen=[]
    for ticker,g in scored.groupby("ticker",sort=False):
        g=g.sort_values("end_period_ts")
        for _,r in g.iterrows():
            options=[(side,float(r[f"edge_{side}"]),float(r[f"prob_{side}"]))
                     for side in ("YES","NO")
                     if r[f"edge_{side}"]>=edge_cut and r[f"prob_{side}"]>=prob_cut]
            if not options:continue
            side,edge,prob=max(options,key=lambda x:x[1]*x[2])
            buy,sell,fees,net=p_and_l(pd.DataFrame([r]),side,contracts,
                                      slippage,fee_multiplier)
            chosen.append({"ticker":ticker,"signal_ts":int(r.end_period_ts),
                           "entry_ts":int(r.entry_ts),"exit_ts":int(r.exit_ts),
                           "side":side,"model_edge":edge,"profit_probability":prob,
                           "entry_ask":float(buy[0]),"exit_bid":float(sell[0]),
                           "fees":float(fees[0]),"net_proxy":float(net[0]),"fold":fold})
            break
    return pd.DataFrame(chosen,columns=COLUMNS)

def tune(validation,contracts,slippage,fee_multiplier,min_trades):
    trials=[]
    for edge in (.03,.06,.10,.15,.20):
        for prob in (.55,.65,.75):
            trades=select_one_per_market(validation,edge,prob,contracts,
                                         slippage,fee_multiplier)
            if len(trades)>=min_trades:
                trials.append((float(trades.net_proxy.mean()),float(trades.net_proxy.sum()),
                               len(trades),edge,prob))
    if not trials:return None,"No candidate met minimum validation trades"
    best=max(trials,key=lambda x:(x[0],x[1]))
    if best[0]<=0 or best[1]<=0:
        return None,"All eligible validation strategies had nonpositive net P&L"
    return (best[3],best[4]),f"validation trades={best[2]} mean={best[0]:.4f}"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KXBTC15M")
    p.add_argument("--hold",type=int,default=2)
    p.add_argument("--min-train-markets",type=int,default=250)
    p.add_argument("--validation-markets",type=int,default=100)
    p.add_argument("--test-markets",type=int,default=50)
    p.add_argument("--min-validation-trades",type=int,default=10)
    p.add_argument("--slippage",type=float,default=.01)
    p.add_argument("--contracts",type=int,default=1)
    p.add_argument("--fee-multiplier",type=float,default=1.)
    p.add_argument("--max-spot-age",type=int,default=120)
    a=p.parse_args()
    if not 1<=a.hold<=8 or min(a.min_train_markets,a.validation_markets,a.test_markets,
         a.min_validation_trades,a.contracts)<1 or min(a.slippage,a.fee_multiplier)<0:
        p.error("Invalid positive size/hold or negative cost parameter.")
    d=dataset(a.series,a.hold,a.max_spot_age)
    markets=d.groupby("ticker").end_period_ts.min().sort_values().index.to_list()
    print(f"V6 usable markets: {len(markets)}, rows={len(d)}")
    summary=[]; all_trades=[]
    start=a.min_train_markets+a.validation_markets
    for end in range(start,len(markets),a.test_markets):
        selected=markets[end:end+a.test_markets]
        if len(selected)<max(10,a.test_markets//2):break
        tr=d[d.ticker.isin(markets[:end-a.validation_markets])]
        va=d[d.ticker.isin(markets[end-a.validation_markets:end])]
        te=d[d.ticker.isin(selected)]
        features=list(BASE_FEATURES)
        if all(x.strike_distance_pct.notna().all() for x in (tr,va,te)):
            features+=["strike_distance_pct"]
        price,clf=model_pair(tr,features,a.contracts,a.slippage,a.fee_multiplier)
        val=score(va,price,clf,features,a.contracts,a.slippage,a.fee_multiplier)
        choice,reason=tune(val,a.contracts,a.slippage,a.fee_multiplier,
                            a.min_validation_trades)
        pred=price.predict(te[features])
        mae=float(mean_absolute_error(te.target,pred))
        base=float(mean_absolute_error(te.target,np.zeros(len(te))))
        if choice is None:
            trades=pd.DataFrame(columns=COLUMNS)
        else:
            test=score(te,price,clf,features,a.contracts,a.slippage,a.fee_multiplier)
            trades=select_one_per_market(test,*choice,a.contracts,a.slippage,
                                          a.fee_multiplier,len(summary)+1)
        pnl=float(trades.net_proxy.sum()) if len(trades) else 0.
        print(f"Fold {len(summary)+1}: train={tr.ticker.nunique()} val={va.ticker.nunique()} "
              f"test={te.ticker.nunique()} MAE={mae:.4f} base={base:.4f} "
              f"signals={len(trades)} proxy_net=${pnl:.2f} | {reason}")
        summary.append({"fold":len(summary)+1,"test_markets":te.ticker.nunique(),
                        "test_rows":len(te),"price_mae":mae,"baseline_mae":base,
                        "trades":len(trades),"net_proxy":pnl,
                        "validation_policy":reason,"trade_threshold":str(choice)})
        if len(trades):all_trades.append(trades)
    if not summary:raise RuntimeError("Not enough chronologically distinct markets for V6 folds")
    dest=ROOT/"reports";dest.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(dest/f"{a.series}_v6_folds.csv",index=False)
    (pd.concat(all_trades,ignore_index=True) if all_trades else
        pd.DataFrame(columns=COLUMNS)).to_csv(dest/f"{a.series}_v6_trades.csv",index=False)
    s=pd.DataFrame(summary)
    print(f"Total quote-proxy trades={int(s.trades.sum())}; net=${s.net_proxy.sum():.2f} "
          f"across {len(s)} folds")
    print("READ-ONLY, NOT REAL FILL EVIDENCE. Prior V1-V5 analysis reused these historical "
          "markets; seek a fresh future holdout.")

if __name__=="__main__":main()
