# Freqtrade: Your Open-Source Crypto Trading Bot 🚀

**Automate your cryptocurrency trading strategies with Freqtrade, a powerful and customizable open-source bot.** ([Back to Original Repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade empowers you to build and automate your crypto trading strategies, offering a range of features from backtesting to machine learning-powered optimization. This bot provides a solid foundation for both beginners and experienced traders to explore the world of algorithmic trading.

## Key Features

*   **Python-Based:** Built on Python 3.11+, offering flexibility and cross-platform compatibility (Windows, macOS, Linux).
*   **Exchange Support:** Compatible with a wide array of major cryptocurrency exchanges (Binance, Bybit, OKX, and many more).
*   **Backtesting & Optimization:** Test your strategies with backtesting tools and optimize them using machine learning and FreqAI.
*   **Dry-Run Mode:** Simulate trading without risking real capital.
*   **WebUI & Telegram Integration:** Manage your bot through a built-in web interface and Telegram commands.
*   **Data Analysis:** Utilize tools for plotting, performance reporting, and profit/loss tracking in fiat currency.
*   **Adaptive Prediction Modeling:** Leverage FreqAI to build smart strategies that self-train to the market via adaptive machine learning methods.

## Disclaimer

This software is intended for educational and experimental purposes. Always start with dry-run mode and thoroughly understand the bot's functionality before using it with real funds.  The authors and affiliates are not responsible for your trading outcomes.
We strongly recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchanges

Freqtrade supports a variety of cryptocurrency exchanges. For exchange-specific configuration and notes, please consult the [exchange-specific notes](docs/exchanges.md).

*   [Binance](https://www.binance.com/)
*   [Bitmart](https://bitmart.com/)
*   [BingX](https://bingx.com/invite/0EM9RX)
*   [Bybit](https://bybit.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [HTX](https://www.htx.com/)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [Kraken](https://kraken.com/)
*   [OKX](https://okx.com/)
*   [MyOKX](https://okx.com/) (OKX EEA)
*   [potentially many others](https://github.com/ccxt/ccxt/). _(We cannot guarantee they will work)_

### Supported Futures Exchanges (experimental)

*   [Binance](https://www.binance.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [OKX](https://okx.com/)
*   [Bybit](https://bybit.com/)

Please make sure to read the [exchange specific notes](docs/exchanges.md), as well as the [trading with leverage](docs/leverage.md) documentation before diving in.

### Community-Tested Exchanges

Exchanges confirmed working by the community:

*   [Bitvavo](https://bitvavo.com/)
*   [Kucoin](https://www.kucoin.com/)

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io), guiding you through setup, usage, and advanced features.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/). For other installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

Control your bot seamlessly with Telegram. See the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for a full command list.

-   `/start`: Starts the trader.
-   `/stop`: Stops the trader.
-   `/stopentry`: Stop entering new trades.
-   `/status <trade_id>|[table]`: Lists all or specific open trades.
-   `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
-   `/profit_long [<n>]`: Lists cumulative profit from all finished long trades, over the last n days.
-   `/profit_short [<n>]`: Lists cumulative profit from all finished short trades, over the last n days.
-   `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
-   `/fx <trade_id>|all`: Alias to `/forceexit`
-   `/performance`: Show performance of each finished trade grouped by pair
-   `/balance`: Show account balance per currency.
-   `/daily <n>`: Shows profit or loss per day, over the last n days.
-   `/help`: Show help message.
-   `/version`: Show version.

## Development Branches

*   `develop`:  The main development branch, which may contain new features and breaking changes.
*   `stable`: The latest stable release branch.
*   `feat/*`: Feature branches, actively under development.

## Support

### Help / Discord

Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for community support and discussions.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs in the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).  If the issue hasn't been reported, please
[create a new issue](https://github.com/freqtrade/freqtrade/issues/new/choose) and
ensure you follow the template guide so that the team can assist you as
quickly as possible.

For every [issue](https://github.com/freqtrade/freqtrade/issues/new/choose) created, kindly follow up and mark satisfaction or reminder to close issue when equilibrium ground is reached.

--Maintain github's [community policy](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct)--

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Share your ideas for bot improvements. Search for existing requests first, then [create a new request](https://github.com/freqtrade/freqtrade/issues/new/choose), following the template.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to Freqtrade with your pull requests!

Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) before submitting your PRs.

Issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) can be good first contributions, and will help get you familiar with the codebase.

**Note:** Open an issue to discuss major new features on the [Discord](https://discord.gg/p7nuUNVfP7) (#dev channel) before you start working on them.

**Important:**  Always create your PR against the `develop` branch.

## Requirements

### Up-to-Date Clock

An accurate, frequently synchronized clock is critical for exchange communication.

### Minimum Hardware

*   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software

*   [Python >= 3.11](http://docs.python-guide.org/en/latest/starting/installation/)
*   [pip](https://pip.pypa.io/en/stable/installing/)
*   [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
*   [TA-Lib](https://ta-lib.github.io/ta-lib-python/)
*   [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
*   [Docker](https://www.docker.com/products/docker) (Recommended)