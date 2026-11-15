"""BTC 15-minute dual-target research: settlement outcome + short-horizon tradable moves.

Read-only. Uses historical one-minute quotes as an imperfect research proxy.
Never places orders and does not use future settlement labels as features.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error

ROOT=Path(__file__).resolve().parent
FEATURES=["mid","spread","age","volume","ret1","ret2","ret3",
          "rollvol3","rollspread3","rollvolume3","bidchange","askchange"]
def modelset(task):
    if task=="class":
        return {
            "logistic":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                                     LogisticRegression(max_iter=600)),
            "forest":RandomForestClassifier(n_estimators=130,min_samples_leaf=8,
                                            random_state=42,n_jobs=-1),
            "hist_boost":HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=9,
                                                        l2_regularization=8,random_state=42),
            "neural":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(32,16),alpha=5,
                                      early_stopping=True,max_iter=320,random_state=42))}
    return {
        "ridge":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=20)),
        "extra_trees":ExtraTreesRegressor(n_estimators=130,min_samples_leaf=8,
                                         random_state=42,n_jobs=-1),
        "neural":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                 MLPRegressor(hidden_layer_sizes=(32,16),alpha=5,
                              early_stopping=True,max_iter=320,random_state=42))}

def fee(price):
    """Illustrative taker fee, not a confirmed series-specific fee."""
    p=np.asarray(price)
    return np.ceil(.07*p*(1-p)*10000)/10000

def load(series,horizons):
    path=ROOT/"data"/f"{series}_minute.csv"
    if not path.exists():raise FileNotFoundError(f"Run kalshi_lab.py fetch --series {series} first.")
    d=pd.read_csv(path)
    for c in ("end_period_ts","bid","ask","volume"):
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["ticker","end_period_ts","bid","ask","volume"])
    d=d[(d.bid>=0)&(d.ask<=1)&(d.ask>=d.bid)].copy()
    d=d.sort_values(["ticker","end_period_ts"]).drop_duplicates(["ticker","end_period_ts"])
    d["mid"]=(d.bid+d.ask)/2
    d["spread"]=d.ask-d.bid
    g=d.groupby("ticker",sort=False)
    d["age"]=g.cumcount()
    for h in (1,2,3):
        d[f"ret{h}"]=d.mid-g.mid.shift(h)
    d["rollvol3"]=g.mid.transform(lambda x:x.diff().rolling(3).std())
    d["rollspread3"]=g.spread.transform(lambda x:x.rolling(3).mean())
    d["rollvolume3"]=g.volume.transform(lambda x:x.rolling(3).mean())
    d["bidchange"]=d.bid-g.bid.shift(1)
    d["askchange"]=d.ask-g.ask.shift(1)
    d["entry_ask"]=g.ask.shift(-1)
    d["entry_ts"]=g.end_period_ts.shift(-1)
    for h in horizons:
        d[f"exit_bid_{h}"]=g.bid.shift(-h)
        d[f"exit_ts_{h}"]=g.end_period_ts.shift(-h)
        d[f"target_{h}"]=g.mid.shift(-h)-d.mid
        d[f"gross_{h}"]=d[f"exit_bid_{h}"]-d.entry_ask
    d["outcome"]=d.get("result",pd.Series(index=d.index,dtype=str)).astype(str).str.lower().map(
        {"yes":1,"no":0})
    d=d.replace([np.inf,-np.inf],np.nan)
    feat=d.dropna(subset=FEATURES).copy()
    return feat

def splits(d):
    markets=d.groupby("ticker").end_period_ts.min().sort_values().index
    if len(markets)<45:raise ValueError(f"Need >=45 market windows; have {len(markets)}")
    a,b=int(.6*len(markets)),int(.8*len(markets))
    out=[d[d.ticker.isin(markets[i:j])].copy()
         for i,j in ((0,a),(a,b),(b,len(markets)))]
    print(f"Markets train={a}, validation={b-a}, test={len(markets)-b}")
    return out

def classification(train,val,test):
    train=train.dropna(subset=["outcome"])
    val=val.dropna(subset=["outcome"])
    test=test.dropna(subset=["outcome"])
    if min(len(train),len(val),len(test))<40 or train.outcome.nunique()<2:
        print("Outcome forecast skipped: too few settled YES/NO observations.");return pd.DataFrame()
    print("\nSETTLEMENT probability — Brier lower is better (market midpoint baseline)")
    print(f"Market-midpoint baseline: {brier_score_loss(test.outcome,test.mid.clip(.001,.999)):.4f}")
    preds={}
    selection={}
    for name,model in modelset("class").items():
        model.fit(train[FEATURES],train.outcome.astype(int))
        pv=model.predict_proba(val[FEATURES])[:,1]
        pt=model.predict_proba(test[FEATURES])[:,1]
        selection[name]=brier_score_loss(val.outcome,pv)
        preds[name]=pt
        auc=roc_auc_score(test.outcome,pt) if test.outcome.nunique()==2 else float("nan")
        print(f"{name:12} Brier={brier_score_loss(test.outcome,pt):.4f} AUC={auc:.3f} "
              f"validation_Brier={selection[name]:.4f}")
    best=min(selection,key=selection.get)
    report=test[["ticker","end_period_ts","mid","bid","ask","outcome"]].copy()
    report["prob_yes"]=preds[best]
    report["market_implied_mid"]=test.mid
    print("Validation-selected model:",best)
    return report

def prices(train,val,test,horizons,threshold):
    results=[]
    for h in horizons:
        def eligible(f):
            z=f.dropna(subset=[f"target_{h}","entry_ask",f"exit_bid_{h}"]).copy()
            ok=((z.entry_ts-z.end_period_ts)==60)&(
                (z[f"exit_ts_{h}"]-z.end_period_ts)==60*h)
            return z.loc[ok]
        tr,va,te=(eligible(f) for f in (train,val,test))
        if min(len(tr),len(va),len(te))<25:
            print(f"h{h} skipped: insufficient continuous observations");continue
        models=modelset("reg")
        selection={}
        preds={}
        for name,m in models.items():
            m.fit(tr[FEATURES],tr[f"target_{h}"])
            selection[name]=mean_absolute_error(va[f"target_{h}"],m.predict(va[FEATURES]))
            preds[name]=m.predict(te[FEATURES])
        winner=min(selection,key=selection.get)
        pred=preds[winner]
        baseline=mean_absolute_error(te[f"target_{h}"],np.zeros(len(te)))
        mae=mean_absolute_error(te[f"target_{h}"],pred)
        # Do not trigger until predicted change exceeds current spread and buffer.
        # Enter next minute at ask, exit horizon at bid: only quote proxy, NOT actual fills.
        take=pred>te.spread.to_numpy()+threshold
        gross=te[f"exit_bid_{h}"].to_numpy()-te.entry_ask.to_numpy()
        cost=fee(te.entry_ask.to_numpy())+fee(te[f"exit_bid_{h}"].to_numpy())
        net=gross-cost
        print(f"h{h} winner={winner:11} MAE={mae:.4f} no-change={baseline:.4f} "
              f"proxy_signals={take.sum()} proxy_PnL=${net[take].sum():.2f}")
        r=te[["ticker","end_period_ts","mid","bid","ask"]].copy()
        r["horizon_minutes"]=h
        r["pred_future_mid"]=te.mid.to_numpy()+pred
        r["actual_future_mid"]=te.mid.to_numpy()+te[f"target_{h}"].to_numpy()
        r["estimated_net_if_bought"]=net
        r["signal_long_yes"]=take
        results.append(r)
    return pd.concat(results,ignore_index=True) if results else pd.DataFrame()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KXBTC15M")
    p.add_argument("--horizons",default="1,2,3")
    p.add_argument("--threshold",type=float,default=.02,
                   help="Additional predicted-edge buffer in dollars")
    p.add_argument("--mode",choices=("all","outcome","price"),default="all")
    a=p.parse_args()
    hs=sorted({int(x) for x in a.horizons.split(",")})
    if not hs or min(hs)<1 or max(hs)>8:p.error("Horizon must be 1..8 minutes")
    d=load(a.series,hs)
    tr,va,te=splits(d)
    folder=ROOT/"reports"
    folder.mkdir(exist_ok=True)
    if a.mode in ("all","outcome"):
        report=classification(tr,va,te)
        if not report.empty:
            report.to_csv(folder/f"{a.series}_outcome_predictions.csv",index=False)
    if a.mode in ("all","price"):
        report=prices(tr,va,te,hs,a.threshold)
        if not report.empty:
            report.to_csv(folder/f"{a.series}_intrawindow_predictions.csv",index=False)
    print("\nRESEARCH ONLY. Minute quotes are not executable fills; overlapping signals "
          "are not independent; costs are illustrative, and test results are not live P&L.")
if __name__=="__main__":main()
