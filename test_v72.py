"""V7.2 offline tests; use synthetic events, not your personal API key."""
import json
import tempfile
import unittest
from pathlib import Path
from v72_features import extract_file, add_targets
from v72_rolling_ws import choose_active


def entry(ts,typ,msg,seq=1,sid=1):
    return {"local_ts_ms":ts,"event":{"type":typ,"sid":sid,"seq":seq,"msg":msg}}


class V72Tests(unittest.TestCase):
    def test_market_discovery(self):
        m=[
            {"ticker":"OLD","status":"open","close_time":"2026-11-15T00:00:00Z"},
            {"ticker":"NEXT","status":"active","close_time":"2026-12-01T00:30:00Z"},
            {"ticker":"SOON","status":"open","close_time":"2026-12-01T00:15:00Z"}]
        self.assertEqual(choose_active(m,now=1796083200),"SOON")

    def test_sequence_and_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"BTC_fake.jsonl"
            snap={"market_ticker":"BTC_FAKE","yes_dollars_fp":[["0.40","10"]],
                  "no_dollars_fp":[["0.50","20"]]}
            change={"market_ticker":"BTC_FAKE","side":"yes",
                    "price_dollars":"0.40","delta_fp":"2"}
            events=[
                {"type":"capture_start"},
                entry(100000,"orderbook_snapshot",snap,2),
                entry(100100,"orderbook_delta",change,3),
                entry(100500,"trade",{"market_ticker":"BTC_FAKE","count_fp":"3",
                       "taker_outcome_side":"yes"},sid=2),
                entry(101100,"orderbook_delta",change,4),
                entry(102100,"orderbook_delta",change,6), # missing seq 5
                entry(103100,"orderbook_delta",change,7),
                entry(104100,"orderbook_snapshot",snap,8),
                entry(105100,"orderbook_delta",change,9)
            ]
            path.write_text("".join(json.dumps(x)+"\n" for x in events),
                            encoding="utf-8")
            rows,audit=extract_file(path)
            self.assertEqual(audit["invalid_deltas"],2)
            self.assertEqual(audit["trades"],1)
            self.assertEqual(audit["valid_snapshots"],2)
            self.assertGreater(len(rows),0)
            first=rows[0]
            self.assertAlmostEqual(first["yes_bid"],.4)
            self.assertAlmostEqual(first["yes_ask"],.5)
            self.assertAlmostEqual(first["imbalance5"],(12-20)/(12+20))  # 10 YES + 2 delta; 20 NO
            self.assertEqual(first["trade_count"],1)
            self.assertAlmostEqual(first["yes_taker_volume"],3.0)
            labeled=add_targets(rows)
            self.assertTrue(all(x["future_mid_change_5s"]!=
                                x["future_mid_change_5s"] for x in labeled))

    def test_gap_safe_target(self):
        records=[]
        for ts in (100,101,102,104,105,106):
            records.append({"ticker":"M","source_file":"a","second_ts":ts,
                            "mid":ts/1000,"book_epoch":1})
        labeled=add_targets(records)
        self.assertEqual(len(labeled),6)
        self.assertTrue(all(row["future_mid_change_5s"]!=
                            row["future_mid_change_5s"] for row in labeled))

    def test_snapshot_reset_breaks_target(self):
        records=[]
        for sec in range(100,108):
            records.append({"ticker":"M","source_file":"a","second_ts":sec,
                            "mid":sec/1000,"book_epoch":1 if sec<104 else 2})
        labeled=add_targets(records)
        self.assertTrue(all(row["future_mid_change_5s"]!=
                            row["future_mid_change_5s"] for row in labeled))

if __name__=="__main__":
    unittest.main()
