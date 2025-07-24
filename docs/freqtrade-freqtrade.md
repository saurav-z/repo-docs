# Freqtrade: The Open Source Crypto Trading Bot

> Automate your cryptocurrency trading strategies with Freqtrade, a powerful and versatile open-source bot. ([Original Repository](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a free and open-source crypto trading bot written in Python, designed to automate your trading strategies across various cryptocurrency exchanges.  It offers a comprehensive suite of tools for backtesting, strategy optimization, and real-time trading, all manageable via Telegram or a web interface.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Cross-Platform Compatibility:** Runs on Windows, macOS, and Linux, supporting Python 3.11+.
*   **Exchange Integration:** Supports major exchanges, including Binance, Bybit, Gate.io, HTX, OKX, Kraken, and more.  See [exchange specific notes](docs/exchanges.md) for details.
*   **Backtesting & Strategy Optimization:**  Thoroughly test strategies with backtesting and optimize parameters using machine learning.
*   **Machine Learning with FreqAI:** Build smarter, adaptive strategies that self-train to market changes.
*   **Flexible Management:** Control the bot via a web UI and Telegram.
*   **Dry-run Mode:** Test strategies without risking real funds.
*   **Fiat Profit/Loss Display:** Monitor your profits and losses in your preferred fiat currency.
*   **Reporting:** Get performance reports on current trades.
*   **Data Handling:**  Utilize whitelists, blacklists, and dynamic pair selection.

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk, and you could lose money. Always use the software at your own risk and begin with "dry-run" mode before trading with real funds. Understand the software before using it. The authors and affiliates take no responsibility for your trading results.

## Quick Start

Get started quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For other installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

Control your bot easily with Telegram.  See the full command list in the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/).

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

*   `develop`: Active development branch, may contain new features and breaking changes.
*   `stable`: Latest stable release.
*   `feat/*`: Feature branches for specific features.

## Support

### Help / Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and to connect with other users.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs in the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue) and follow the provided template.

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest new features in the [issue tracker](https://github.com/freqtrade/freqtrade/issues/new/choose) following the template.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to Freqtrade by submitting pull requests.  See the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.  Create PRs against the `develop` branch.

## Requirements

### Up-to-date clock

Maintain an accurate clock, synchronized frequently with an NTP server, to avoid exchange communication issues.

### Minimum hardware required

- Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software requirements

*   [Python >= 3.11](http://docs.python-guide.org/en/latest/starting/installation/)
*   [pip](https://pip.pypa.io/en/stable/installing/)
*   [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
*   [TA-Lib](https://ta-lib.github.io/ta-lib-python/)
*   [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
*   [Docker](https://www.docker.com/products/docker) (Recommended)