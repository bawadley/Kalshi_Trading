"""V7.6 read-only BTC15M execution research: longer holds and maker uncertainty.

NO order APIs. Compare 30/60/120s taker buy/sell and conservative
maker-bid/trade-through fill proxies on WHOLE chronological markets.
Policies are picked using validation markets only; NO_TRADE is permitted.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v75_execution import taker_fee

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "reports" / "v74" / "v74_enriched_features.csv"
OUTPUT = ROOT / "reports" / "v76"
HOLDS = (30, 60, 120)
RULES = {
    "momentum": (.003, .008, .015),
    "reversion": (.003, .008, .015),
    "imbalance": (.10, .25, .45),
}
EXECUTIONS = ("taker", "maker_through")


def prepared(data):
    columns = ["ticker", "source_file", "book_epoch", "segment", "second_ts",
               "yes_bid", "yes_ask", "yes_bid_qty", "yes_ask_qty",
               "quote_age_ms", "seconds_to_close", "mid_change_5s",
               "imbalance5"]
    missing = [x for x in columns if x not in data]
    if missing:
        raise ValueError(f"Missing V7.4 columns: {missing}")
    d = data.copy()
    for c in columns[2:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=columns[:5])
    dupe = d.duplicated(["ticker", "second_ts"], keep=False)
    if dupe.any():
        print(f"Excluding {int(dupe.sum())} overlapping market-second rows")
        d = d.loc[~dupe].copy()
    d = d.sort_values(["ticker", "second_ts", "source_file"]).reset_index(drop=True)
    return d


def quote_ok(row, age_ms):
    if row is None:
        return False
    return bool(np.isfinite(row.yes_bid) and np.isfinite(row.yes_ask)
                and 0 < row.yes_bid < row.yes_ask < 1
                and np.isfinite(row.quote_age_ms)
                and row.quote_age_ms <= age_ms)


def index_book(data):
    result = {}
    for row in data.itertuples(index=False):
        key = (row.ticker, row.source_file, row.book_epoch,
               row.segment, int(row.second_ts))
        result[key] = row
    return result


def lookup(book, signal, when):
    key = (signal.ticker, signal.source_file, signal.book_epoch,
           signal.segment, int(when))
    return book.get(key)


def direction(signal, rule, threshold):
    if rule in ("momentum", "reversion"):
        value = signal.mid_change_5s
    else:
        value = signal.imbalance5
    if not np.isfinite(value) or abs(value) < threshold:
        return None
    if rule == "reversion":
        value = -value
    return "YES" if value > 0 else "NO"


def fee(price, qty, multiplier, maker=False):
    if maker:
        raw = .0175 * multiplier * qty * price * (1 - price)
        return float(np.ceil(raw * 10000 - 1e-9) / 10000)
    return float(taker_fee(price, qty, multiplier))


def result(status, net, entered=False, unresolved=False, buy=None, sell=None,
           fill_time=None):
    return {"status": status, "net_proxy": float(net), "filled_proxy": entered,
            "unresolved": unresolved, "entry_price": buy, "exit_price": sell,
            "fill_second": fill_time}


def simulate(book, signal, side, execution, hold, latency, maker_wait, qty,
             slip, taker_multiplier, maker_multiplier, age_ms):
    """A conservative quote-proxy outcome, including uncertain exit penalties.

    When an entry observation cannot be found, conservatively attribute -qty
    rather than silently dropping the signal. When a booked entry has no
    observed exit, assume a worthless position (worst-case mark).
    Both are *data-uncertainty bounds*, not actual realized losses.
    """
    if side not in ("YES", "NO"):
        raise ValueError("Side must be YES or NO")
    if execution not in ("taker", "maker_through", "maker_touch"):
        raise ValueError("Unknown execution mode")
    t = int(signal.second_ts)
    entry = lookup(book, signal, t + latency)
    if not quote_ok(entry, age_ms):
        return result("ENTRY_UNOBSERVED", -qty, unresolved=True)
    if execution == "taker":
        buy = entry.yes_ask if side == "YES" else 1 - entry.yes_bid
        size = entry.yes_ask_qty if side == "YES" else entry.yes_bid_qty
        if not np.isfinite(size) or size < qty:
            return result("NO_DISPLAYED_ENTRY_DEPTH", 0)
        filled_at = t + latency
        maker_entry = False
    else:
        # Hypothetical resting bid at the currently observable best bid;
        # assumes a passive order can be posted there without crossing.
        buy = entry.yes_bid if side == "YES" else 1 - entry.yes_ask
        filled_at = None
        for offset in range(1, maker_wait + 1):
            future = lookup(book, signal, t + latency + offset)
            if not quote_ok(future, age_ms):
                return result("FILL_WINDOW_UNOBSERVED", -qty, unresolved=True)
            # This is a proxy for price moving THROUGH the resting order,
            # not a queue-aware proof of actual fill. Touch is more optimistic.
            ask = future.yes_ask if side == "YES" else 1 - future.yes_bid
            eligible = ask < buy if execution == "maker_through" else ask <= buy
            if eligible:
                filled_at = t + latency + offset
                break
        if filled_at is None:
            return result("MAKER_NOT_FILLED_PROXY", 0)
        maker_entry = True
    buy_exec = buy if maker_entry else min(1., buy + slip)
    entry_fee = fee(buy_exec, qty, maker_multiplier if maker_entry
                    else taker_multiplier, maker=maker_entry)
    exit_book = lookup(book, signal, filled_at + hold)
    if not quote_ok(exit_book, age_ms):
        # Full loss bound even if the reason is file truncation, not the real market.
        return result("EXIT_UNOBSERVED_LOSS_BOUND", -qty * buy_exec - entry_fee,
                      entered=True, unresolved=True, buy=buy_exec,
                      fill_time=filled_at)
    sell = exit_book.yes_bid if side == "YES" else 1 - exit_book.yes_ask
    depth = exit_book.yes_bid_qty if side == "YES" else exit_book.yes_ask_qty
    if not np.isfinite(depth) or depth < qty:
        return result("EXIT_NO_DISPLAYED_DEPTH_LOSS_BOUND",
                      -qty * buy_exec - entry_fee, entered=True, unresolved=True,
                      buy=buy_exec, fill_time=filled_at)
    sell_exec = max(0., sell - slip)
    exit_fee = fee(sell_exec, qty, taker_multiplier)
    return result("QUOTE_COMPLETE", qty * (sell_exec - buy_exec) -
                  entry_fee - exit_fee, entered=True, buy=buy_exec,
                  sell=sell_exec, fill_time=filled_at)


def eligible_signals(data, hold, latency, maker_wait, age_ms, sample_step):
    group = data.sort_values("second_ts")
    start = int(group.second_ts.min())
    window = latency + maker_wait + hold + 2
    for row in group.itertuples(index=False):
        if (int(row.second_ts) - start) % sample_step != 0:
            continue
        if not quote_ok(row, age_ms):
            continue
        if not np.isfinite(row.seconds_to_close) or row.seconds_to_close < window:
            continue
        yield row


def choose_signal(data, rule, threshold, hold, latency, maker_wait, age_ms,
                  sample_step):
    for signal in eligible_signals(data, hold, latency, maker_wait,
                                   age_ms, sample_step):
        side = direction(signal, rule, threshold)
        if side:
            return signal, side
    return None


def strategy_results(data, book, policy, latency, wait, qty, slippage,
                     taker_multiplier, maker_multiplier, age_ms, sample_step):
    rule, threshold, hold, execution = policy
    results = []
    for ticker, subset in data.groupby("ticker", sort=False):
        found = choose_signal(subset, rule, threshold, hold, latency, wait,
                              age_ms, sample_step)
        if found is None:
            continue
        signal, side = found
        outcome = simulate(book, signal, side, execution, hold, latency,
                           wait, qty, slippage, taker_multiplier,
                           maker_multiplier, age_ms)
        results.append({"ticker": ticker, "signal_second": int(signal.second_ts),
                        "side":side, "hold_s":hold,"execution":execution,
                        "rule":rule,"threshold":threshold, **outcome})
    return pd.DataFrame(results)


def choose_policy(val, book, holds, latency, wait, qty, worst_slip,
                  taker_multiplier, maker_multiplier, age_ms, sample_step,
                  min_filled):
    """Validation-only search; no-trade is always a valid default."""
    winner = None
    for hold in holds:
        for mode in EXECUTIONS:
            for rule, thresholds in RULES.items():
                for threshold in thresholds:
                    policy = (rule, threshold, hold, mode)
                    rows = strategy_results(val, book, policy, latency, wait,
                                            qty, worst_slip, taker_multiplier,
                                            maker_multiplier, age_ms, sample_step)
                    if len(rows) < min_filled:
                        continue
                    if rows.unresolved.any():
                        continue
                    executed = rows.filled_proxy.sum()
                    successes = (rows.net_proxy > 0).sum()
                    net = rows.net_proxy.sum()
                    if executed < min_filled or successes < min_filled or net <= 0:
                        continue
                    # A small robustness hurdle, not a guarantee of performance.
                    score = (float(net), int(executed))
                    if winner is None or score > winner[0]:
                        winner = (score, policy, rows)
    return winner


def chronological_folds(names, train_n, val_n, test_n):
    for pos in range(train_n+val_n, len(names)-test_n+1, test_n):
        yield names[:pos-val_n], names[pos-val_n:pos], names[pos:pos+test_n]


def audit_windows(data, book, holds, latency, wait, age_ms, step):
    audit = []
    for hold in holds:
        eligible = 0
        observed_entry = 0
        observed_taker_exit = 0
        observed_maker_window = 0
        for _, market in data.groupby("ticker", sort=False):
            for signal in eligible_signals(market, hold, latency, wait,
                                           age_ms, step):
                eligible += 1
                t = int(signal.second_ts)
                entry = lookup(book, signal, t + latency)
                if not quote_ok(entry, age_ms):
                    continue
                observed_entry += 1
                exit_row = lookup(book, signal, t + latency + hold)
                if quote_ok(exit_row, age_ms):
                    observed_taker_exit += 1
                if all(quote_ok(lookup(book, signal, t + latency + k), age_ms)
                       for k in range(1, wait + 1)):
                    observed_maker_window += 1
        audit.append({"hold_s":hold,"signal_windows":eligible,
                      "entry_observed":observed_entry,
                      "taker_exit_observed":observed_taker_exit,
                      "complete_maker_wait_windows":observed_maker_window})
    return pd.DataFrame(audit)


def run(input_path, output, holds, train_n, val_n, test_n, latency,
        maker_wait, sample_step, qty, slips, taker_multiplier,
        maker_multiplier, max_quote_age, min_val_filled):
    if min_val_filled > val_n:
        raise ValueError("Minimum filled validation markets must be <= validation markets")
    data = prepared(pd.read_csv(input_path))
    book = index_book(data)
    tickers = data.groupby("ticker").second_ts.min().sort_values().index.tolist()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"V7.6: {len(data):,} quotes across {len(tickers)} markets. "
          f"Validation-only policy search; worst slippage {max(slips):.3f}/leg.")
    audit = audit_windows(data, book, holds, latency, maker_wait,
                          max_quote_age, sample_step)
    audit.to_csv(output/"v76_coverage.csv", index=False)
    print("\nRecording coverage (not trading opportunities):")
    print(audit.to_string(index=False))
    fold_rows = []
    trade_rows = []
    for fold, (tr_ids, va_ids, te_ids) in enumerate(
            chronological_folds(tickers,train_n,val_n,test_n),1):
        validation = data[data.ticker.isin(va_ids)]
        test = data[data.ticker.isin(te_ids)]
        selected = choose_policy(validation,book,holds,latency,maker_wait,
                                 qty,max(slips),taker_multiplier,maker_multiplier,
                                 max_quote_age,sample_step,min_val_filled)
        if selected is None:
            for slip in slips:
                fold_rows.append({"fold":fold,"policy":"NO_TRADE",
                    "slippage_each_taker_leg":slip,"test_markets":len(te_ids),
                    "test_signals":0,"filled_proxies":0,"unresolved":0,
                    "net_quote_proxy":0.})
            print(f"\nFold {fold}: NO_TRADE — no strategy survived validation")
            continue
        (_, policy, validated) = selected
        name = f"{policy[0]}/{policy[1]}/{policy[2]}s/{policy[3]}"
        print(f"\nFold {fold}: validation selected {name} "
              f"validation net bound USD {validated.net_proxy.sum():+.4f}")
        for slip in slips:
            outcomes = strategy_results(test,book,policy,latency,maker_wait,
                                        qty,slip,taker_multiplier,maker_multiplier,
                                        max_quote_age,sample_step)
            n = len(outcomes)
            net = float(outcomes.net_proxy.sum()) if n else 0.
            filled = int(outcomes.filled_proxy.sum()) if n else 0
            unresolved = int(outcomes.unresolved.sum()) if n else 0
            fold_rows.append({"fold":fold,"policy":name,
                "slippage_each_taker_leg":slip,"test_markets":len(te_ids),
                "test_signals":n,"filled_proxies":filled,
                "unresolved":unresolved,"net_quote_proxy":net})
            if n:
                outcomes["fold"] = fold
                outcomes["slippage_each_taker_leg"] = slip
                trade_rows.append(outcomes)
            print(f"  slip={slip:.3f} signals={n} fills_proxy={filled} "
                  f"unresolved={unresolved} net-bound USD {net:+.4f}")
    if not fold_rows:
        raise RuntimeError("Need more independent markets to form a chronological fold")
    pd.DataFrame(fold_rows).to_csv(output/"v76_folds.csv",index=False)
    if trade_rows:
        pd.concat(trade_rows,ignore_index=True).to_csv(
            output/"v76_selected_signals.csv",index=False)
    else:
        pd.DataFrame(columns=["ticker","side","status","net_proxy"]).to_csv(
            output/"v76_selected_signals.csv",index=False)
    print("\nNo market orders submitted. Maker fill is a BOOK TRADE-THROUGH PROXY, "
          "not observed personal fills. Unobserved entry/exit prices receive "
          "pessimistic loss bounds, NOT realized P&L.")
    print("Results are exploratory and were tuned using previously studied markets.")
    print(f"Reports: {output}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",default=str(FEATURES))
    p.add_argument("--output",default=str(OUTPUT))
    p.add_argument("--holds",default="30,60,120")
    p.add_argument("--train-markets",type=int,default=6)
    p.add_argument("--val-markets",type=int,default=2)
    p.add_argument("--test-markets",type=int,default=2)
    p.add_argument("--latency-seconds",type=int,default=1)
    p.add_argument("--maker-wait",type=int,default=5)
    p.add_argument("--sample-step",type=int,default=5)
    p.add_argument("--contracts",type=int,default=1)
    p.add_argument("--slippages",default="0,0.005,0.01")
    p.add_argument("--taker-multiplier",type=float,default=1.)
    p.add_argument("--maker-multiplier",type=float,default=0.)
    p.add_argument("--max-quote-age-ms",type=int,default=1200)
    p.add_argument("--min-validation-filled",type=int,default=2)
    a=p.parse_args()
    holds=[int(s.strip()) for s in a.holds.split(",") if s.strip()]
    slips=sorted(float(s.strip()) for s in a.slippages.split(",") if s.strip())
    if (not holds or not slips or len(set(holds)) != len(holds)
        or any(h<1 or h>300 for h in holds)
        or any(s<0 or s>.1 for s in slips)
        or min(a.train_markets,a.val_markets,a.test_markets,
               a.latency_seconds,a.maker_wait,a.sample_step,a.contracts,
               a.max_quote_age_ms,a.min_validation_filled)<1
        or min(a.taker_multiplier,a.maker_multiplier)<0):
        p.error("Invalid holds, splits, fee multipliers, timing or slippage")
    run(a.input,a.output,holds,a.train_markets,a.val_markets,a.test_markets,
        a.latency_seconds,a.maker_wait,a.sample_step,a.contracts,slips,
        a.taker_multiplier,a.maker_multiplier,a.max_quote_age_ms,
        a.min_validation_filled)


if __name__=="__main__":
    main()
