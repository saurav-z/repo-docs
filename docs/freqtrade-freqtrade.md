# Freqtrade: Your Open-Source Crypto Trading Bot

**Automate your crypto trading strategies with Freqtrade, a powerful and customizable open-source bot.** ([Back to the original repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a free and open-source crypto trading bot written in Python, empowering you to automate your trading strategies across various cryptocurrency exchanges. Designed for flexibility and ease of use, it supports all major exchanges and offers control via Telegram or a webUI. Its features include backtesting, plotting, money management tools, and strategy optimization through machine learning.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features:

*   **Multi-Exchange Support:** Trade on major exchanges including Binance, Bybit, OKX, and more. ([See supported exchanges](docs/exchanges.md))
*   **Backtesting:** Simulate your trading strategies with historical data to refine your approach.
*   **Strategy Optimization:** Leverage machine learning for strategy parameter optimization using real-time exchange data.
*   **FreqAI Integration:** Build smart strategies with adaptive machine learning methods. [Learn more](https://www.freqtrade.io/en/stable/freqai/)
*   **WebUI and Telegram Control:** Manage your bot with a built-in web interface or via Telegram commands.
*   **Dry-Run Mode:** Test strategies without risking real funds.
*   **Profit/Loss Tracking:** Monitor your profit/loss in fiat currency.
*   **Community Support:** Benefit from a strong community and extensive documentation.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

We strongly recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchange Marketplaces

Please read the [exchange specific notes](docs/exchanges.md) to learn about eventual, special configurations needed for each exchange.

*   [X] [Binance](https://www.binance.com/)
*   [X] [Bitmart](https://bitmart.com/)
*   [X] [BingX](https://bingx.com/invite/0EM9RX)
*   [X] [Bybit](https://bybit.com/)
*   [X] [Gate.io](https://www.gate.io/ref/6266643)
*   [X] [HTX](https://www.htx.com/)
*   [X] [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [X] [Kraken](https://kraken.com/)
*   [X] [OKX](https://okx.com/)
*   [X] [MyOKX](https://okx.com/) (OKX EEA)
*   [ ] [potentially many others](https://github.com/ccxt/ccxt/). _(We cannot guarantee they will work)_

### Supported Futures Exchanges (experimental)

*   [X] [Binance](https://www.binance.com/)
*   [X] [Gate.io](https://www.gate.io/ref/6266643)
*   [X] [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [X] [OKX](https://okx.com/)
*   [X] [Bybit](https://bybit.com/)

Please make sure to read the [exchange specific notes](docs/exchanges.md), as well as the [trading with leverage](docs/leverage.md) documentation before diving in.

### Community Tested

Exchanges confirmed working by the community:

*   [X] [Bitvavo](https://bitvavo.com/)
*   [X] [Kucoin](https://www.kucoin.com/)

## Documentation

For comprehensive documentation, visit the [Freqtrade Website](https://www.freqtrade.io).

## Quick Start

Get started quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).

For alternative installation methods, refer to the [Installation documentation](https://www.freqtrade.io/en/stable/installation/).

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

Take control of your bot via Telegram (not mandatory):

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

*   `develop`: Latest features and potential breaking changes.
*   `stable`: The latest stable release.
*   `feat/*`: Feature branches currently under development.

## Support

### Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support, discussions, and community engagement.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs and issues to the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest new features via the [feature requests](https://github.com/freqtrade/freqtrade/labels/enhancement) section.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to Freqtrade by submitting pull requests.  Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) first. Create PRs against the `develop` branch.

## Requirements

### Up-to-Date Clock

Ensure an accurate clock synchronized to a NTP server.

### Minimum Hardware

*   Recommended: 2GB RAM, 1GB disk space, 2vCPU

### Software

*   [Python >= 3.11](http://docs.python-guide.org/en/latest/starting/installation/)
*   [pip](https://pip.pypa.io/en/stable/installing/)
*   [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
*   [TA-Lib](https://ta-lib.github.io/ta-lib-python/)
*   [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
*   [Docker](https://www.docker.com/products/docker) (Recommended)