# Freqtrade: Your Open-Source Crypto Trading Bot for Automated Profits

Freqtrade is a powerful, free, and open-source crypto trading bot written in Python, designed to automate your trading strategies and help you navigate the volatile crypto market. [Visit the original repo](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

## Key Features

*   **Open Source & Free:** Leverage a community-driven bot, available to everyone.
*   **Multi-Exchange Support:** Trade on major exchanges, with community-tested support and future exchange compatibility.
*   **Backtesting & Strategy Optimization:** Simulate your strategies and use machine learning to optimize them.
*   **FreqAI Integration:** Build adaptive prediction models.
*   **WebUI & Telegram Control:** Manage your bot with a built-in web interface and Telegram commands.
*   **Dry-Run Mode:** Test your strategies without risking real capital.
*   **Performance Reporting:**  Gain insights into your trades with fiat profit/loss and trade status reports.
*   **Python 3.11+:** Enjoy full compatibility on any operating system.

## Disclaimer

This software is for educational purposes only. Do not risk money which
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS
AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not
hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchanges

Freqtrade supports a wide array of exchanges.  Refer to the [exchange specific notes](docs/exchanges.md) for special configurations.

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
*   [and potentially many others](https://github.com/ccxt/ccxt/)

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Read the [exchange specific notes](docs/exchanges.md) and the [trading with leverage](docs/leverage.md) documentation before starting.

### Community Tested

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).
For native installation, please refer to the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

Use Telegram to control your bot with these commands:

*   `/start`: Start the trader.
*   `/stop`: Stop the trader.
*   `/stopentry`: Stop entering new trades.
*   `/status <trade_id>|[table]`: List open trades.
*   `/profit [<n>]`: List cumulative profit.
*   `/profit_long [<n>]`: List cumulative profit from long trades.
*   `/profit_short [<n>]`: List cumulative profit from short trades.
*   `/forceexit <trade_id>|all`: Exit a trade.
*   `/fx <trade_id>|all`: Alias to `/forceexit`
*   `/performance`: Show performance.
*   `/balance`: Show account balance.
*   `/daily <n>`: Shows daily profit or loss.
*   `/help`: Show help message.
*   `/version`: Show version.

More details and the full command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

## Development Branches

*   `develop`: New features and potential breaking changes.
*   `stable`: Stable release.
*   `feat/*`: Feature branches.

## Support

### Help / Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community engagement.

### Issues

Report bugs or issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### Feature Requests

Suggest new features on the [enhancement requests](https://github.com/freqtrade/freqtrade/labels/enhancement).

### Pull Requests

Contribute by sending pull requests, and start with improving the documentation or issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue).  Always create PRs against the `develop` branch.

**Important:** Describe any new feature on an issue before working on it.

## Requirements

### Up-to-date Clock

Ensure accurate clock synchronization with an NTP server.

### Minimum Hardware Requirements

*   2GB RAM, 1GB disk space, 2vCPU (Recommended)

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)