# Freqtrade: The Open-Source Crypto Trading Bot

**Automate your cryptocurrency trading strategies with Freqtrade, a powerful and versatile open-source bot.** ([Back to the Project](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade empowers both novice and experienced traders with robust tools for crypto trading. It supports various exchanges and offers features like backtesting, strategy optimization, and web/Telegram-based control.

## Key Features

*   **Open Source & Free:** Benefit from a community-driven project with transparent development.
*   **Exchange Support:** Compatible with major exchanges, including Binance, Kraken, OKX, and more.  See [exchange specific notes](docs/exchanges.md) for details.
*   **Futures Trading (Experimental):** Supports futures trading on selected exchanges.  See [trading with leverage](docs/leverage.md) for details.
*   **Backtesting & Optimization:** Test strategies with historical data and optimize them using machine learning.
*   **FreqAI Integration:** Leverage adaptive machine learning for smart, self-training trading strategies.  [Learn more](https://www.freqtrade.io/en/stable/freqai/)
*   **Web & Telegram Control:** Manage and monitor your bot through a user-friendly web UI or Telegram commands.
*   **Dry-Run Mode:** Test your strategies without risking real capital.
*   **Performance Monitoring:** Track your profit/loss in fiat currency and generate performance reports.
*   **Whitelisting/Blacklisting:** Control which cryptocurrencies the bot trades.
*   **Persistence** Persistence is achieved through sqlite.
*   **Builtin WebUI**: Builtin web UI to manage your bot.
*   **Manageable via Telegram**: Manage the bot with Telegram.

## Disclaimer

This software is for educational purposes only. Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

We strongly recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For other installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Supported Exchanges

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX
*   MyOKX
*   [and potentially many others](https://github.com/ccxt/ccxt/).

### Community Tested Exchanges

*   Bitvavo
*   Kucoin

## Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Documentation

Explore the comprehensive documentation on the [freqtrade website](https://www.freqtrade.io) to understand the bot's functionality.

## Basic Usage

### Bot commands

```
usage: freqtrade [-h] [-V]
                 {trade,create-userdir,new-config,show-config,new-strategy,download-data,convert-data,convert-trade-data,trades-to-ohlcv,list-data,backtesting,backtesting-show,backtesting-analysis,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-markets,list-pairs,list-strategies,list-hyperoptloss,list-freqaimodels,list-timeframes,show-trades,test-pairlist,convert-db,install-ui,plot-dataframe,plot-profit,webserver,strategy-updater,lookahead-analysis,recursive-analysis}
                 ...

Free, open source crypto trading bot

positional arguments:
  {trade,create-userdir,new-config,show-config,new-strategy,download-data,convert-data,convert-trade-data,trades-to-ohlcv,list-data,backtesting,backtesting-show,backtesting-analysis,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-markets,list-pairs,list-strategies,list-hyperoptloss,list-freqaimodels,list-timeframes,show-trades,test-pairlist,convert-db,install-ui,plot-dataframe,plot-profit,webserver,strategy-updater,lookahead-analysis,recursive-analysis}
    trade               Trade module.
    create-userdir      Create user-data directory.
    new-config          Create new config
    show-config         Show resolved config
    new-strategy        Create new strategy
    download-data       Download backtesting data.
    convert-data        Convert candle (OHLCV) data from one format to
                        another.
    convert-trade-data  Convert trade data from one format to another.
    trades-to-ohlcv     Convert trade data to OHLCV data.
    list-data           List downloaded data.
    backtesting         Backtesting module.
    backtesting-show    Show past Backtest results
    backtesting-analysis
                        Backtest Analysis module.
    hyperopt            Hyperopt module.
    hyperopt-list       List Hyperopt results
    hyperopt-show       Show details of Hyperopt results
    list-exchanges      Print available exchanges.
    list-markets        Print markets on exchange.
    list-pairs          Print pairs on exchange.
    list-strategies     Print available strategies.
    list-hyperoptloss   Print available hyperopt loss functions.
    list-freqaimodels   Print available freqAI models.
    list-timeframes     Print available timeframes for the exchange.
    show-trades         Show trades.
    test-pairlist       Test your pairlist configuration.
    convert-db          Migrate database to different system
    install-ui          Install FreqUI
    plot-dataframe      Plot candles with indicators.
    plot-profit         Generate plot showing profits.
    webserver           Webserver module.
    strategy-updater    updates outdated strategy files to the current version
    lookahead-analysis  Check for potential look ahead bias.
    recursive-analysis  Check for potential recursive formula issue.

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
```

### Telegram RPC commands

Utilize Telegram commands to control your bot remotely. Complete command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

- `/start`: Starts the trader.
- `/stop`: Stops the trader.
- `/stopentry`: Stop entering new trades.
- `/status <trade_id>|[table]`: Lists all or specific open trades.
- `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
- `/profit_long [<n>]`: Lists cumulative profit from all finished long trades, over the last n days.
- `/profit_short [<n>]`: Lists cumulative profit from all finished short trades, over the last n days.
- `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
- `/fx <trade_id>|all`: Alias to `/forceexit`
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency.
- `/daily <n>`: Shows profit or loss per day, over the last n days.
- `/help`: Show help message.
- `/version`: Show version.

## Development Branches

*   `develop`: For new features (may contain breaking changes).
*   `stable`: Latest stable release.

## Support

### Help / Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for community support and discussions.

### Issues

Report bugs and issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### Feature Requests

Suggest new features on the [feature request page](https://github.com/freqtrade/freqtrade/labels/enhancement).

### Pull Requests

Contribute to Freqtrade by submitting pull requests. See the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).

## Requirements

*   Up-to-date clock (synchronized with NTP server).
*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)
*   Minimum system requirements: 2GB RAM, 1GB disk space, 2vCPU