"""Unit checks for selective V4 gating and NO-side accounting."""
import unittest
import numpy as np
import pandas as pd
from btc_selective_v4 import market_quotes, simulate, choose_threshold

class SelectiveV4Tests(unittest.TestCase):
    def test_yes_no_quotes(self):
        d=pd.DataFrame([dict(entry_yes_ask=.42, entry_yes_bid=.40,
                             exit_yes_bid=.60,exit_yes_ask=.62)])
        by,sy,fy,ny=market_quotes(d,"YES",1,0,1)
        bn,sn,fn,nn=market_quotes(d,"NO",1,0,1)
        self.assertAlmostEqual(by[0],.42)
        self.assertAlmostEqual(sy[0],.60)
        self.assertAlmostEqual(bn[0],.60)
        self.assertAlmostEqual(sn[0],.38)
        self.assertLess(nn[0],0)
        self.assertGreater(ny[0],0)

    def test_no_trade_gate_and_one_per_market(self):
        common=dict(ticker="M1",mid=.5,end_period_ts=60,entry_ts=120,
                    exit_ts=240,entry_yes_ask=.45,entry_yes_bid=.43,
                    exit_yes_bid=.55,exit_yes_ask=.57,
                    edge_yes=.12,edge_no=-.1,p_yes=.8,p_no=.2)
        d=pd.DataFrame([common,dict(common,end_period_ts=120,entry_ts=180,exit_ts=300)])
        self.assertEqual(len(simulate(d,1,0,1,.2,.7)),0)
        self.assertEqual(len(simulate(d,1,0,1,.1,.7)),1)
        self.assertEqual(simulate(d,1,0,1,.1,.7).iloc[0].side,"YES")

if __name__=="__main__":unittest.main()
