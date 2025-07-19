# Freqtrade: The Open-Source Crypto Trading Bot for Automated Profits

**Automate your crypto trading strategy with Freqtrade, a powerful, open-source bot that's easy to use and customize!** ([See the original repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a free and open-source crypto trading bot written in Python designed to help you automate your trading strategies on various cryptocurrency exchanges.  It offers a range of features for both beginners and experienced traders.  With backtesting, strategy optimization, and a user-friendly interface, Freqtrade empowers you to take control of your crypto investments.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Supports Major Exchanges:** Integrates with popular exchanges like Binance, Bybit, OKX, and more (see the list below).
*   **Backtesting & Strategy Optimization:** Test your strategies with historical data and optimize them using machine learning.
*   **Machine Learning:** Uses machine learning to optimize your buy/sell strategy parameters with real exchange data.
*   **Adaptive Prediction Modeling:** Build a smart strategy with FreqAI that self-trains to the market via adaptive machine learning methods. [Learn more](https://www.freqtrade.io/en/stable/freqai/)
*   **Web UI & Telegram Control:** Manage your bot via a built-in web interface or through Telegram commands.
*   **Dry-Run Mode:**  Test your strategies safely without risking real money.
*   **Fiat Profit/Loss Reporting:**  Track your performance in your preferred fiat currency.
*   **Built-in Strategy Tools:**  Includes features like pair whitelisting, blacklisting, and performance reports.
*   **Python 3.11+ Based:** Runs on Windows, macOS, and Linux.

## Supported Exchanges

Please refer to the [exchange specific notes](docs/exchanges.md) for specific configuration details.

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
*   and potentially many others.

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

### Community Tested Exchanges

*   Bitvavo
*   Kucoin

## Documentation

Find comprehensive documentation and detailed guides on the [Freqtrade website](https://www.freqtrade.io).

## Disclaimer

This software is for educational purposes only.  Do not risk money you cannot afford to lose.  Use the software at your own risk.  The authors and affiliates are not responsible for your trading results. Always start with Dry-run mode and understand how the bot functions before trading with real funds. We recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.

## Quick Start

Get started quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).

For native installation, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

Control your bot remotely with Telegram.  See the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for a full command list.

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

*   `develop`:  The active development branch with the latest features.
*   `stable`: The latest stable release.
*   `feat/*`: Feature branches for specific feature development.

## Support

### Help / Discord

Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support and community discussions.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs in the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest new features in the [feature requests](https://github.com/freqtrade/freqtrade/labels/enhancement) section.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to Freqtrade through pull requests.  See the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.  Submit PRs against the `develop` branch.

## Requirements

### Up-to-date Clock

Ensure your system clock is accurate and synchronized to an NTP server to avoid exchange communication issues.

### Minimum Hardware

*   Minimal (advised): 2GB RAM, 1GB disk space, 2 vCPU

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)