"""Kalshi Research V2: read-only forecasting, profit labels and market-making proxy."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parent
FEATURES = ["mid", "spread", "age", "volume", "r1", "r2", "r3", "r5",
            "rvol3", "rvol5", "spread3", "vol3", "bid_delta1", "ask_delta1"]

def load(series):
    path = ROOT / "data" / f"{series}_minute.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run kalshi_lab.py fetch first.")
    d = pd.read_csv(path)
    for c in ("bid", "ask", "volume", "end_period_ts"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["ticker", "bid", "ask", "volume", "end_period_ts"])
    d = d[(d.bid >= 0) & (d.ask <= 1) & (d.bid <= d.ask)]
    d = d.sort_values(["ticker", "end_period_ts"]).drop_duplicates(["ticker", "end_period_ts"]).copy()
    d["mid"] = (d.bid + d.ask) / 2
    d["spread"] = d.ask - d.bid
    return d

def features(d, horizons):
    g = d.groupby("ticker", sort=False)
    valid = np.ones(len(d), dtype=bool)
    for j in range(1, max(horizons) + 1):
        valid &= ((g.end_period_ts.shift(-j) - d.end_period_ts).to_numpy() == j * 60)
    d = d.copy()
    g = d.groupby("ticker", sort=False)
    d["age"] = g.cumcount()
    for j in (1, 2, 3, 5):
        d[f"r{j}"] = d.mid - g.mid.shift(j)
    d["rvol3"] = g.mid.transform(lambda x: x.diff().rolling(3).std())
    d["rvol5"] = g.mid.transform(lambda x: x.diff().rolling(5).std())
    d["spread3"] = g.spread.transform(lambda x: x.rolling(3).mean())
    d["vol3"] = g.volume.transform(lambda x: x.rolling(3).mean())
    d["bid_delta1"] = d.bid - g.bid.shift(1)
    d["ask_delta1"] = d.ask - g.ask.shift(1)
    d["entry_ask"] = g.ask.shift(-1)
    for h in horizons:
        d[f"future_mid_{h}"] = g.mid.shift(-h)
        d[f"future_bid_{h}"] = g.bid.shift(-h)
        d[f"target_{h}"] = d[f"future_mid_{h}"] - d.mid
        d[f"vol_target_{h}"] = sum((g.mid.shift(-j) - g.mid.shift(-(j - 1))).abs()
                                      for j in range(1, h + 1))
    required = FEATURES + ["entry_ask"] + [f"target_{h}" for h in horizons]
    required += [f"vol_target_{h}" for h in horizons] + [f"future_bid_{h}" for h in horizons]
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.loc[d.index[valid]].dropna(subset=required).copy()

def split(d):
    ticks = d.groupby("ticker").end_period_ts.min().sort_values().index
    if len(ticks) < 25:
        raise RuntimeError("At least 25 separate markets required.")
    a, b = int(.6 * len(ticks)), int(.8 * len(ticks))
    result = [d[d.ticker.isin(ticks[s:e])].copy() for s, e in ((0,a),(a,b),(b,len(ticks)))]
    if min(map(len, result)) < 30:
        raise RuntimeError("Insufficient complete observations for chronological split.")
    return result

def regressors():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=15)),
        "forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=8, n_jobs=-1, random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=100, min_samples_leaf=8, n_jobs=-1, random_state=42),
        "neural": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(48,24),
                     alpha=2, early_stopping=True, max_iter=300, random_state=42)),
    }

def fee(price):
    """Illustrative taker fee; verify this series' actual fee schedule."""
    p = np.asarray(price)
    return np.ceil(.07 * p * (1-p) * 10000) / 10000

def forecasts(train, valid, test, horizons):
    ys = [f"target_{h}" for h in horizons]
    ytr, yva, yte = (x[ys].to_numpy() for x in (train, valid, test))
    val, preds = {}, {}
    print("\nMulti-horizon future YES-midpoint forecasts, MAE ($/contract)")
    for name, model in regressors().items():
        model.fit(train[FEATURES], ytr)
        val[name] = model.predict(valid[FEATURES])
        preds[name] = model.predict(test[FEATURES])
        print(f"{name:14} " + " ".join(f"h{h}:{mean_absolute_error(yte[:,i], preds[name][:,i]):.4f}"
                                          for i,h in enumerate(horizons)))
    best = sorted(val, key=lambda n: np.mean(np.abs(val[n]-yva)))[:2]
    preds["ensemble"] = np.mean([preds[n] for n in best], axis=0)
    print("no_change      " + " ".join(f"h{h}:{mean_absolute_error(yte[:,i], np.zeros(len(test))):.4f}"
                                       for i,h in enumerate(horizons)))
    print("ensemble       " + " ".join(f"h{h}:{mean_absolute_error(yte[:,i], preds['ensemble'][:,i]):.4f}"
                                       for i,h in enumerate(horizons)) + f" (validation-selected: {best})")
    result = test[["ticker","end_period_ts","mid","bid","ask"]].copy()
    for i,h in enumerate(horizons):
        result[f"actual_mid_h{h}"] = test[f"future_mid_{h}"].to_numpy()
        for name,p in preds.items():
            result[f"pred_{name}_h{h}"] = test.mid.to_numpy() + p[:,i]
    folder = ROOT / "reports"
    folder.mkdir(exist_ok=True)
    result.to_csv(folder / "candle_forecasts.csv", index=False)
    print("Saved reports/candle_forecasts.csv")

def volatility(train, test, horizons):
    print("\nRealized absolute-path volatility forecasting: MAE ($/contract)")
    for h in horizons:
        yt = train[f"vol_target_{h}"]
        y = test[f"vol_target_{h}"]
        forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=8, n_jobs=-1, random_state=42)
        neural = make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),
                      alpha=2, early_stopping=True, max_iter=250, random_state=42))
        forest.fit(train[FEATURES],yt)
        neural.fit(train[FEATURES],yt)
        print(f"h{h}: baseline={mean_absolute_error(y, np.full(len(y),yt.median())):.4f} "
              f"forest={mean_absolute_error(y,np.maximum(0,forest.predict(test[FEATURES]))):.4f} "
              f"neural={mean_absolute_error(y,np.maximum(0,neural.predict(test[FEATURES]))):.4f}")

def profitability(train, valid, test, horizons, buffer):
    """Model probability that next ask -> future bid exceeds assumed taker fees."""
    print("\nPredicting profitable YES long quote paths (not verified fills)")
    for h in horizons:
        def label(x):
            a = x.entry_ask.to_numpy()
            b = x[f"future_bid_{h}"].to_numpy()
            return ((b-a-fee(a)-fee(b)) > buffer).astype(int)
        yt, yv, ye = label(train), label(valid), label(test)
        if min(yt.sum(),len(yt)-yt.sum()) < 8:
            print(f"h{h}: insufficient balanced training examples")
            continue
        candidates = {
            "logistic": make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)),
            "forest": RandomForestClassifier(n_estimators=120,min_samples_leaf=10,n_jobs=-1,random_state=42),
            "neural": make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),
                       alpha=2,early_stopping=True,max_iter=300,random_state=42))}
        score = {}
        for name,m in candidates.items():
            m.fit(train[FEATURES],yt)
            score[name] = (m.predict_proba(valid[FEATURES])[:,1],
                           m.predict_proba(test[FEATURES])[:,1])
        rank = sorted(score, key=lambda n: roc_auc_score(yv,score[n][0])
                      if len(np.unique(yv))==2 else .5, reverse=True)
        scores = {n:score[n][1] for n in rank}
        scores["ensemble"] = np.mean([score[n][1] for n in rank[:2]],axis=0)
        actual = test[f"future_bid_{h}"].to_numpy()-test.entry_ask.to_numpy()
        actual -= fee(test.entry_ask.to_numpy())+fee(test[f"future_bid_{h}"].to_numpy())
        for name,pred in scores.items():
            auc=roc_auc_score(ye,pred) if len(np.unique(ye))==2 else float("nan")
            selected=pred>=.65
            print(f"h{h} {name:9} AUC={auc:.3f} signals={selected.sum():4d} "
                  f"quote-proxy-net=${actual[selected].sum():.2f}")

def maker(data, hold=3, maker_fee=0.0, tick=.01):
    """Strict crossing proxy: actual maker fills CANNOT be inferred from candles."""
    ticks=data.groupby("ticker").end_period_ts.min().sort_values().index
    groups=data[data.ticker.isin(ticks[int(.8*len(ticks)):])]
    pnl=[]; buys=0; paired=0
    for _,g in groups.groupby("ticker"):
        g=g.sort_values("end_period_ts").reset_index(drop=True)
        i=0
        while i<len(g)-1:
            now=g.iloc[i]
            if now.ask-now.bid<3*tick:
                i+=1; continue
            buy=round(float(now.bid+tick),2)
            sellquote=round(float(now.ask-tick),2)
            nxt=g.iloc[i+1]
            if nxt.end_period_ts-now.end_period_ts!=60 or nxt.ask>buy-tick+1e-9:
                i+=1; continue
            buys+=1
            end=min(len(g)-1,i+1+hold)
            sold=False
            for j in range(i+2,end+1):
                bar=g.iloc[j]
                if bar.end_period_ts-now.end_period_ts!=60*(j-i): break
                if bar.bid>=sellquote+tick-1e-9:
                    pnl.append(sellquote-buy-2*maker_fee)
                    paired+=1; i=j; sold=True; break
            if not sold:
                bar=g.iloc[end]
                if bar.end_period_ts-now.end_period_ts==60*(end-i):
                    exit_price=float(bar.bid)
                    pnl.append(exit_price-buy-maker_fee-float(fee(exit_price)))
                i=end
            i+=1
    p=np.array(pnl)
    print("\nMarket-maker quote-crossing stress proxy (NOT order-book replay)")
    print(f"Holdout markets={groups.ticker.nunique()} assumed buy crosses={buys} "
          f"assumed paired exits={paired} scored={len(p)} "
          f"proxy net=${p.sum():.2f} mean=${p.mean() if len(p) else 0:.4f}")
    print("No queue, trade prints, cancellation, market impact or genuine execution evidence.")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series",default="KX2YRRATE15M")
    ap.add_argument("--horizons",default="1,2,3,4,5")
    ap.add_argument("--mode",choices=["all","forecast","volatility","profitability","maker"],default="all")
    ap.add_argument("--profit-buffer",type=float,default=0.0)
    ap.add_argument("--maker-fee",type=float,default=0.0,help="Per contract per maker fill")
    args=ap.parse_args()
    hs=sorted({int(x) for x in args.horizons.split(",")})
    if not hs or min(hs)<1 or max(hs)>8: ap.error("horizons must be between 1 and 8")
    d=load(args.series)
    if args.mode=="maker":
        maker(d,maker_fee=args.maker_fee); return
    e=features(d,hs)
    tr,va,te=split(e)
    print(f"Chronological markets: train={tr.ticker.nunique()}, "
          f"validation={va.ticker.nunique()}, test={te.ticker.nunique()}")
    if args.mode in ("all","forecast"): forecasts(tr,va,te,hs)
    if args.mode in ("all","volatility"): volatility(tr,te,hs)
    if args.mode in ("all","profitability"): profitability(tr,va,te,hs,args.profit_buffer)
    if args.mode=="all": maker(d,maker_fee=args.maker_fee)
    print("\nRESEARCH ONLY: no live orders. Candle quotes do not establish executable fills.")

if __name__=="__main__":
    main()
