# Freqtrade: The Free and Open Source Crypto Trading Bot

**Automate your crypto trading with Freqtrade, a powerful Python-based bot designed for profitability and ease of use.  [Explore the Freqtrade Repo](https://github.com/freqtrade/freqtrade)**

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a versatile, open-source crypto trading bot built in Python, perfect for both beginners and experienced traders. It supports a wide range of major cryptocurrency exchanges and offers robust features for automated trading, backtesting, and strategy optimization.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

**Key Features:**

*   **Cross-Platform**: Runs on Windows, macOS, and Linux with Python 3.11+ support.
*   **Exchange Compatibility**: Supports major exchanges like Binance, Bybit, OKX, and more.
*   **Backtesting & Optimization**: Backtest your strategies and optimize them using machine learning.
*   **FreqAI Integration**: Leverage adaptive machine learning for smart strategy development.
*   **Dry-Run Mode**: Test your strategies risk-free with dry-run mode.
*   **Telegram & WebUI**: Manage your bot conveniently via Telegram commands or a built-in web interface.
*   **Data & Reporting**: Includes data downloaders, profit/loss display in fiat, and performance reports.
*   **Community Supported**: Benefit from a vibrant community and active development.

## Disclaimer

*   **For Educational Purposes Only**: Use this software at your own risk. The authors and affiliates are not responsible for your trading results.
*   **Start with Dry-Run**:  Always begin with the dry-run mode to understand the bot's functionality and expected outcomes before trading with real money.
*   **Coding Knowledge Recommended**: Understanding the codebase is encouraged.

## Supported Exchanges

Freqtrade supports many of the major crypto exchanges. Please refer to the [exchange specific notes](docs/exchanges.md) for any special configurations.

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
*   ...and potentially many others! (See [ccxt](https://github.com/ccxt/ccxt/))

### Supported Futures Exchanges (experimental)

*   [Binance](https://www.binance.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [OKX](https://okx.com/)
*   [Bybit](https://bybit.com/)

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io), providing detailed instructions and insights into the bot's functionality.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/). For other installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

Control your bot remotely using Telegram. See the full command list in the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/).

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

*   `develop`:  Contains new features and potential breaking changes.
*   `stable`: Contains the latest stable release.
*   `feat/*`: Feature branches for active development.

## Support

### Help / Discord

For support, questions, and community engagement, join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7).

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs through the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue). Before submitting, please search to ensure it hasn't already been reported.

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest new features through the [issue tracker](https://github.com/freqtrade/freqtrade/labels/enhancement). Search existing requests first.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to Freqtrade by submitting pull requests.  Refer to the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.  Always create PRs against the `develop` branch.

## Requirements

### Up-to-date clock

Ensure your system clock is accurate and synchronized to a NTP server.

### Minimum Hardware

*   2GB RAM (recommended)
*   1GB disk space
*   2 vCPU

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)