"""Offline checks for V6 quote arithmetic, no-trade filtering, market exclusivity."""
import unittest
import numpy as np
import pandas as pd
from btc_v6_strategy import p_and_l, select_one_per_market, tune

class V6Tests(unittest.TestCase):
    def test_yes_no_quote_conversion(self):
        d=pd.DataFrame([{"entry_yes_ask":.4,"entry_yes_bid":.38,
                         "exit_yes_bid":.61,"exit_yes_ask":.63}])
        by,sy,fy,ny=p_and_l(d,"YES",1,0,1)
        bn,sn,fn,nn=p_and_l(d,"NO",1,0,1)
        self.assertAlmostEqual(by[0],.4)
        self.assertAlmostEqual(sy[0],.61)
        self.assertAlmostEqual(bn[0],.62)
        self.assertAlmostEqual(sn[0],.37)
        self.assertGreater(ny[0],0)
        self.assertLess(nn[0],0)

    def test_one_trade_per_market_and_no_trade(self):
        row={"ticker":"ONE","end_period_ts":60,"entry_ts":120,"exit_ts":240,
             "entry_yes_ask":.4,"entry_yes_bid":.38,
             "exit_yes_bid":.61,"exit_yes_ask":.63,
             "edge_YES":.15,"prob_YES":.8,"edge_NO":-.2,"prob_NO":.2}
        d=pd.DataFrame([row,dict(row,end_period_ts=120,entry_ts=180,exit_ts=300)])
        none=select_one_per_market(d,.2,.7,1,0,1)
        yes=select_one_per_market(d,.1,.7,1,0,1)
        self.assertEqual(len(none),0)
        self.assertEqual(len(yes),1)
        self.assertEqual(yes.iloc[0].side,"YES")
        self.assertGreater(yes.iloc[0].exit_ts,yes.iloc[0].entry_ts)

    def test_no_trade_on_negative_validation(self):
        row={"ticker":"A","end_period_ts":60,"entry_ts":120,"exit_ts":240,
             "entry_yes_ask":.75,"entry_yes_bid":.73,
             "exit_yes_bid":.4,"exit_yes_ask":.42,
             "edge_YES":.22,"prob_YES":.9,"edge_NO":-.1,"prob_NO":.1}
        d=pd.DataFrame([dict(row,ticker=str(i)) for i in range(12)])
        choice,_=tune(d,1,.01,1,10)
        self.assertIsNone(choice)

if __name__=="__main__":
    unittest.main()
