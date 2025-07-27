# Freqtrade: Your Free & Open-Source Crypto Trading Bot

Freqtrade is a powerful, free, and open-source crypto trading bot built in Python, offering advanced features for automated trading. [Check out the official repository](https://github.com/freqtrade/freqtrade)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is designed to automate your cryptocurrency trading on various exchanges, offering robust features for backtesting, optimization, and risk management.

## Key Features

*   **Automated Trading:** Execute trades based on your custom strategies.
*   **Backtesting:** Simulate trades to test and refine your strategies.
*   **Strategy Optimization:** Machine learning-based parameter optimization.
*   **Adaptive Prediction Modeling:** Smart strategy with FreqAI for market adaptation.
*   **Exchange Support:** Compatibility with many popular crypto exchanges.
*   **Dry-run Mode:** Test strategies without risking real capital.
*   **WebUI and Telegram Integration:** Manage and monitor your bot via a web interface or Telegram commands.
*   **Performance Monitoring:** Track and analyze trade performance.
*   **Customizable Whitelists/Blacklists:** Control which coins the bot trades.
*   **Open Source & Free:** Benefit from a community-driven, transparent, and free-to-use trading bot.
*   **Python 3.11+:** Runs on Windows, macOS, and Linux.

## Disclaimer

This software is for educational purposes only. Use the software at your own risk. The authors and all affiliates assume no responsibility for your trading results.

Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

We strongly recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchanges

Freqtrade supports a wide range of crypto exchanges. Please read the [exchange specific notes](docs/exchanges.md) to learn about eventual, special configurations needed for each exchange.

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX
*   MyOKX (OKX EEA)
*   ...and potentially many others via CCXT integration!

### Supported Futures Exchanges (experimental)
*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

### Community tested
*   Bitvavo
*   Kucoin

## Documentation

For comprehensive information on using Freqtrade, please consult the official documentation on the [freqtrade website](https://www.freqtrade.io).

## Quick Start

For a quick and easy setup, refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/). You can also find detailed installation instructions on the [installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

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

### Telegram RPC Commands

The bot can be managed via Telegram.  More details and the full command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)
*   `/start`: Starts the trader.
*   `/stop`: Stops the trader.
*   `/stopentry`: Stop entering new trades.
*   `/status <trade_id>|[table]`: Lists all or specific open trades.
*   `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
*   `/profit_long [<n>]`: Lists cumulative profit from all finished long trades, over the last n days.
*   `/profit_short [<n>]`: Lists cumulative profit from all finished short trades, over the last n days.
*   `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
*   `/fx <trade_id>|all`: Alias to `/forceexit`
*   `/performance`: Show performance of each finished trade grouped by pair
*   `/balance`: Show account balance per currency.
*   `/daily <n>`: Shows profit or loss per day, over the last n days.
*   `/help`: Show help message.
*   `/version`: Show version.

## Development Branches

*   `develop`: Contains new features and potential breaking changes.
*   `stable`: The latest stable release.
*   `feat/*`: Feature branches under active development.

## Support

### Help / Discord

Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support and community engagement.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs and issues via the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest new features via the [feature request section](https://github.com/freqtrade/freqtrade/labels/enhancement).

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to the project by submitting pull requests.  Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)

**Important:** Always create your PR against the `develop` branch, not `stable`.

## Requirements

### Up-to-date clock

The clock must be accurate, synchronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Minimum hardware required

*   2GB RAM
*   1GB disk space
*   2vCPU

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)