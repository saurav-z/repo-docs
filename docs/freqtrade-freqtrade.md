# Freqtrade: Your Free & Open-Source Crypto Trading Bot

**Automate your crypto trading strategies with Freqtrade, a powerful and versatile bot that supports major exchanges.** ([View on GitHub](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a robust and free, open-source crypto trading bot written in Python, designed to empower both novice and experienced traders. It simplifies the process of creating and executing automated trading strategies on a wide range of cryptocurrency exchanges.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Multi-Exchange Support:** Trade on major exchanges, including Binance, Bybit, OKX, and more.  See supported exchanges listed below.
*   **Backtesting:** Test your trading strategies with historical data to optimize performance.
*   **Strategy Optimization:** Utilize machine learning for hyperparameter optimization.
*   **FreqAI:** Build a smart strategy with FreqAI that self-trains to the market via adaptive machine learning methods.
*   **WebUI and Telegram Integration:** Manage and monitor your bot through a built-in web interface or via Telegram commands.
*   **Dry-Run Mode:** Test your strategies without risking real funds.
*   **Advanced Analysis Tools:** Leverage plotting, profit/loss tracking, and performance reports.
*   **Based on Python 3.11+**: For botting on any operating system - Windows, macOS and Linux.
*   **Persistence**: Persistence is achieved through sqlite.

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
*   Potentially many others via [CCXT](https://github.com/ccxt/ccxt/)

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Please make sure to read the [exchange specific notes](docs/exchanges.md), as well as the [trading with leverage](docs/leverage.md) documentation before diving in.

### Community tested

Exchanges confirmed working by the community:

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [freqtrade website](https://www.freqtrade.io) to help you get started.

## Quickstart

The [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) provides the quickest way to get up and running. For other installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

Telegram is not mandatory. However, this is a great way to control your bot. More details and the full command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

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

*   `develop`: Active development branch, may contain new features and breaking changes.
*   `stable`: The latest stable release.
*   `feat/*`: Feature branches, used for specific feature development.

## Support

### Help / Discord

Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) to connect with the community and get support.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs and issues in the issue tracker, making sure to follow the template guide for details.

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest new features by creating a request in the issue tracker.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to the project by submitting pull requests. Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Requirements

### Up-to-date clock

The clock must be accurate, synchronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Minimum hardware required

To run this bot we recommend you a cloud instance with a minimum of:

- Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software requirements

-   [Python >= 3.11](http://docs.python-guide.org/en/latest/starting/installation/)
-   [pip](https://pip.pypa.io/en/stable/installing/)
-   [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
-   [TA-Lib](https://ta-lib.github.io/ta-lib-python/)
-   [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
-   [Docker](https://www.docker.com/products/docker) (Recommended)