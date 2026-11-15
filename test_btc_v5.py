"""V5 offline data-integrity tests (no network calls)."""
import unittest
from pathlib import Path
import tempfile
import pandas as pd
from unittest.mock import patch
import btc_v5_walkforward as wf

class V5Tests(unittest.TestCase):
    def test_backward_spot_availability(self):
        quotes=pd.DataFrame({"end_period_ts":[120,150,180]})
        spot=pd.DataFrame({"available_ts":[120,180],"spot_close":[100,200]})
        joined=pd.merge_asof(quotes,spot,left_on="end_period_ts",right_on="available_ts",
                             direction="backward",tolerance=60)
        self.assertEqual(list(joined.spot_close),[100,100,200])

    def test_earlier_candle_never_uses_later_spot(self):
        quotes=pd.DataFrame({"end_period_ts":[119]})
        spot=pd.DataFrame({"available_ts":[120],"spot_close":[500]})
        joined=pd.merge_asof(quotes,spot,left_on="end_period_ts",right_on="available_ts",
                             direction="backward",tolerance=120)
        self.assertTrue(pd.isna(joined.spot_close.iloc[0]))

    def test_strike_feature_is_optional(self):
        self.assertIn("strike_distance_pct",wf.OPTIONAL)
        self.assertNotIn("strike_distance_pct",wf.BASE_FEATURES)

if __name__=="__main__":unittest.main()
