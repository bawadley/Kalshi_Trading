# Market-agnostic Kalshi research: Bitcoin first

The project is no longer restricted to Treasury yields. `rolling_collector.py` continually discovers active contracts in one or more specified Kalshi series, polls their public order books and rotates to new contracts. It does not trade.

## Bitcoin (24-hour experiment)
```powershell
git pull origin main
.\.venv\Scripts\python.exe rolling_collector.py --series KXBTC15M --seconds 86400 --interval 5
```
For indefinite collection use `--seconds 0` and stop with Ctrl+C. A terminal and computer must remain awake; interrupting is safe. Outputs are written to `data/depth/<individual-market-ticker>.jsonl`.

## Multiple markets
```powershell
.\.venv\Scripts\python.exe rolling_collector.py --series KXBTC15M,KXETH15M,KXSOL15M --seconds 3600 --interval 5
```
You can supply any verified Kalshi series ticker, including Treasury or metals; availability varies. **The collector only records active markets**, not future `initialized` contracts, and auto-discovers again every 15 seconds.

Known examples: Bitcoin `KXBTC15M`, Ether `KXETH15M`, Solana `KXSOL15M`, 2-year Treasury yield `KX2YRRATE15M`. Verify exact series identifiers for other commodities instead of assuming a naming pattern.

The current REST recorder provides **sampled displayed depth**: YES/NO bid prices and sizes, from which the YES ask can be derived. It does not capture every intermediary book event, executed trade, external reference price, or settlement inputs. Avoid using it to make claims about actual fill probability or market-maker earnings.

Generate per-market features from a recording:
```powershell
.\.venv\Scripts\python.exe market_depth.py features --input data/depth/YOUR_MARKET_TICKER.jsonl
```
For model training, combine the generated per-market tables into one dataset containing 30+ distinct market tickers. Do not train on identical windows in both train and holdout.

**Next engineering stages**: authenticated incremental WebSocket books with sequence-gap recovery and market trades; external asset reference prices with timestamp alignment; a proper depth feature store; chronological cross-market evaluation; actual FIFO/latency-aware backtesting. Keep credentials in environment variables, never in Git.

Source references: https://docs.kalshi.com/ ; https://kalshieye.com/guides/kalshi-15-minute-crypto-markets/
