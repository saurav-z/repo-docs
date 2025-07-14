# Freqtrade: Open-Source Crypto Trading Bot for Automated Trading

Automate your cryptocurrency trading strategies with Freqtrade, a powerful and open-source Python bot.  [Explore the original repo](https://github.com/freqtrade/freqtrade)!

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade empowers you to automate your crypto trading, offering backtesting, strategy optimization, and a user-friendly interface.  Written in Python, this bot supports a wide range of cryptocurrency exchanges.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Free and Open Source:** Leverage the power of a community-driven project.
*   **Supports Major Exchanges:**  Integrates with popular crypto exchanges, including Binance, Bybit, OKX, and more! (See [exchange specific notes](docs/exchanges.md) for details.)
*   **Backtesting:** Test your strategies with historical data to evaluate their performance.
*   **Strategy Optimization:** Utilize machine learning to optimize buy/sell parameters with real exchange data for maximum profitability.
*   **FreqAI Integration:** Build smart strategies that adapt to market conditions with adaptive machine learning. [Learn more](https://www.freqtrade.io/en/stable/freqai/)
*   **Dry-Run Mode:** Test your strategies without risking real funds.
*   **Telegram and WebUI Control:** Manage and monitor your bot via Telegram or a built-in web interface.
*   **Fiat Profit/Loss Reporting:** Easily track your profits and losses in your local currency.
*   **Detailed Performance Reports:** Gain insights with comprehensive performance statistics.
*   **Based on Python 3.11+:** For botting on any operating system - Windows, macOS and Linux.
*   **Persistence:** Persistence is achieved through sqlite.

## Disclaimer

*   This software is for educational purposes only.
*   Risk only money you can afford to lose.
*   The authors and affiliates are not responsible for trading results.
*   Always start in Dry-run mode.
*   Coding and Python knowledge is recommended.

## Getting Started

### Documentation

For complete instructions and in-depth information, please visit the official [freqtrade website](https://www.freqtrade.io).

### Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For native installation, consult the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

### Telegram Commands

Control your bot conveniently using Telegram.  More details and commands are available in the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/).

*   `/start`: Starts the trader.
*   `/stop`: Stops the trader.
*   `/stopentry`: Stop entering new trades.
*   `/status <trade_id>|[table]`: Lists all or specific open trades.
*   `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
*   `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
*   `/fx <trade_id>|all`: Alias to `/forceexit`
*   `/performance`: Show performance of each finished trade grouped by pair
*   `/balance`: Show account balance per currency.
*   `/daily <n>`: Shows profit or loss per day, over the last n days.
*   `/help`: Show help message.
*   `/version`: Show version.

## Development Branches

*   `develop`: Contains new features and may have breaking changes.
*   `stable`: Latest stable release, well-tested.
*   `feat/*`: Feature branches, for testing specific features.

## Support

### Help / Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support, questions, and community engagement.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs and issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest new features on the [feature requests](https://github.com/freqtrade/freqtrade/labels/enhancement).

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to Freqtrade by submitting pull requests.  See the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for details and best practices.  Always create PRs against the `develop` branch.

## Requirements

### Up-to-date clock

Ensure an accurate clock, synchronized frequently with a NTP server.

### Minimum hardware required

*   2GB RAM, 1GB disk space, 2vCPU (Recommended).

### Software requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)