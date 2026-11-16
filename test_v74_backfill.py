"""Offline V7.4 backfill merge tests, no external API calls."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from v74_backfill import merge_csv

class V74BackfillTests(unittest.TestCase):
    def test_merge_preserves_old_rows_and_updates_matching_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"spot.csv"
            pd.DataFrame([{"available_ts":60,"spot_close":1},
                          {"available_ts":120,"spot_close":2}]).to_csv(path,index=False)
            count=merge_csv(path,pd.DataFrame([
                {"available_ts":120,"spot_close":3},
                {"available_ts":180,"spot_close":4}]),"available_ts")
            found=pd.read_csv(path)
            self.assertEqual(count,3)
            self.assertEqual(dict(zip(found.available_ts,found.spot_close)),
                             {60:1,120:3,180:4})

if __name__=="__main__":
    unittest.main()
