# V7.6 — Longer holds, passive-maker fill bounds, and no-trade fallback

Read-only research on KXBTC15M. **No live orders**. This extension
tries to answer whether a cheaper execution method (posting maker bids) or
a longer 30–120-second holding period can overcome the spread and fees.

## PowerShell in the Kalshi_Trading repo

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m unittest test_v76 -v
.\.venv\Scripts\python.exe v76_execution.py --holds 30,60,120 --contracts 1 --latency-seconds 1 --maker-wait 5 --slippages 0,0.005,0.01
```

Input is `reports/v74/v74_enriched_features.csv` and outputs are
`reports/v76/v76_coverage.csv` (recording continuity diagnostic),
`reports/v76/v76_folds.csv` (validation-selected policy and test quote
results), and `reports/v76/v76_selected_signals.csv` (one signal per
market, when selected). Existing code and raw files remain unchanged.

To incorporate **newly collected market windows**, stop/avoid reading an
actively changing JSONL, refresh the V7.2 dataset with
`v72_features.py`, backfill needed BTC data with `v74_backfill.py`,
and rebuild `reports/v74/v74_enriched_features.csv` using
`v74_orderflow.py`. Then rerun V7.6.

## The trading assumptions

Every candidate is selected on information visible at end-of-second t.
The entry quote is examined one second later by default; it cannot use
future prices to select the signal. Evaluation samples only every 5
seconds per market and allows no more than one hypothetical position
per market.

- **Taker entry**: YES at next observed YES ask, NO at next observed
  NO ask (=1-YES bid), provided at least one contract of top-book depth
  is displayed. Exit is held for 30, 60 or 120 seconds, then hypothetically
  sold at YES/NO best bid with a latency/slippage stress.
- **Maker entry**: post hypothetically at the future observation's
  currently displayed best bid. The pessimistic **book trade-through
  proxy** requires an opposing future ask to fall **strictly below**
  the resting bid within a five-second window. This is evidence the
  historical price moved through the order, **NOT proof of a fill**;
  book changes can occur without trades, and our order would have
  changed queue dynamics. No observation of trade-through => assumed
  no fill and zero hypothetical P&L.
- **Unobserved entry**: score a full-contract loss bound for that signal,
  rather than silently remove it. An actual missing entry may mean no
  execution; this severe bound quantifies unknown-data sensitivity,
  not actual lost money.
- **Unobserved exit / missing exit depth**: assume a worthless contract
  after purchase, charging the entry price and fee. This is a *loss
  bound*, NOT actual realized P&L or proof the real exchange lacked
  an exit. The recording may simply have ended.
- **Known current risk gates**: a recorded quote must be reasonably
  fresh, valid, and have enough time until the scheduled market close.
  Book sequences cannot cross recording files, epochs or V7.2 gaps.

Fee assumptions follow Kalshi's **July 7, 2026 published schedule**:
taker fee uses `ceil_0.0001(0.07 × multiplier × contracts × P × (1−P))`
and maker fee uses `ceil_0.0001(0.0175 × multiplier × contracts × P ×
(1−P))`. Default general taker multiplier 1, default general maker
multiplier 0. A maker-fee exception can change this; check **KXBTC15M
live series fee configuration before trading**. There is a taker fee on
the exit even if the entry was a passive maker fill. Slippage scenarios
(0, 0.5¢ and 1¢ per taker leg) are stress-tested. Per-order
ceil-to-centicent rounding is approximated independently on each leg.

Official docs:
- https://kalshi.com/fee-schedule
- https://kalshi.com/docs/kalshi-fee-schedule.pdf

## Why the result can be NO_TRADE

The policy search includes fixed, interpretable momentary signals
(momentum, mean-reversion, depth imbalance) and trade types (taker,
maker with strict trade-through) at each hold length. It chooses a
policy *on adjacent validation markets only*, using the worst
slippage scenario. It requires at least two observed fill proxies,
two positive validation trades, **no unobserved entry/exit in any
validation trade**, and a positive total net quote P&L.

Only selected policies are evaluated on the *subsequent, untouched
whole-market test blocks*. If nothing passes, the policy is NO_TRADE.

We avoid tuning on test results, but **previous versions already
repeatedly explored these same 13 markets**, so none of this proves a
fresh independent out-of-sample edge.

The `v76_coverage.csv` file is especially important for 60/120s
holds: most V7.1 files were ~30 seconds and may contain no intact
long-horizon exit. A NO_TRADE result can therefore mean insufficient
recording coverage, not necessarily that every long hold would lose.

## What this study cannot establish

- Passive maker fill probability; being at the correct price level
  does not prove queue priority or order execution.
- Fair latency/partial-fill model from aggregated level-2 data alone.
- Whether an external cancellation or a websocket recording gap
  masked an otherwise executable trade.
- Actual production profit. Historical quote proxy profits can
  differ dramatically from realized execution outcomes.

Continue collecting **genuinely new market windows** with the V7.2
collector. Any later profitability claim needs paper/live shadow
execution with actual order acknowledgments and fills, held-out data,
and verified current per-series fee settings.
