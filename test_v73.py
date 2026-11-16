"""Offline unit checks for V7.3 market-safe evaluation and price/flow ablation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from v73_orderflow_models import (FLOW_EXTRA, PRICE_ONLY, prepare, spaced_indices,
                                 split_folds, markets_in_order)


class V73Tests(unittest.TestCase):
    def test_chronological_market_split_no_overlap(self):
        names=[f"market{i}" for i in range(13)]
        folds=list(split_folds(names,train_min=6,val_count=2,test_count=2))
        self.assertEqual(len(folds),2)
        for train,val,test in folds:
            self.assertTrue(set(train).isdisjoint(val))
            self.assertTrue(set(train).isdisjoint(test))
            self.assertTrue(set(val).isdisjoint(test))
            self.assertLess(names.index(train[-1]),names.index(val[0]))
            self.assertLess(names.index(val[-1]),names.index(test[0]))
        self.assertEqual(folds[0][2],["market8","market9"])
        self.assertEqual(folds[1][2],["market10","market11"])

    def test_nonoverlapping_test_signals(self):
        d=pd.DataFrame({"ticker":["A"]*17+["B"]*17,
                        "second_ts":list(range(100,117))+list(range(500,517))})
        chosen=d.loc[spaced_indices(d,5)]
        for _,market in chosen.groupby("ticker"):
            differences=np.diff(market.second_ts.to_numpy())
            self.assertTrue(all(differences>=5))
            self.assertTrue(all(differences%5==0))

    def test_price_only_excludes_trade_flow(self):
        self.assertEqual(PRICE_ONLY,["mid","spread"])
        self.assertNotIn("trade_count",PRICE_ONLY)
        self.assertIn("trade_count",FLOW_EXTRA)
        self.assertIn("imbalance5",FLOW_EXTRA)

    def test_prepare_preserves_market_chronology_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[]
            for name,start in (("LATE",500),("EARLY",100)):
                for i in range(8):
                    row={"ticker":name,"source_file":name+".jsonl",
                         "second_ts":start+i,"mid":.50+i*.001,
                         "spread":.02,"future_mid_change_5s":.005}
                    row.update({c:1 for c in FLOW_EXTRA})
                    rows.append(row)
            file=Path(tmp)/"features.csv"
            pd.DataFrame(rows).to_csv(file,index=False)
            d=prepare(file,5)
            self.assertEqual(markets_in_order(d),["EARLY","LATE"])
            self.assertEqual(len(d),16)
            self.assertTrue(np.allclose(d.target,.005))


if __name__=="__main__":
    unittest.main()
