# Freqtrade: Your Open-Source Crypto Trading Bot

**Automate your crypto trading with Freqtrade, a powerful and versatile bot that empowers you to take control of your investments.** ([Learn more](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a free and open-source Python-based crypto trading bot designed to automate your trading strategies across various cryptocurrency exchanges. With its robust features and flexible design, Freqtrade allows both beginners and experienced traders to develop and implement their trading ideas efficiently.

![Freqtrade Screenshot](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Cross-Platform Compatibility:** Runs on Windows, macOS, and Linux, built on Python 3.11+.
*   **Exchange Support:** Integrates with major crypto exchanges like Binance, Bitmart, Bybit, Gate.io, Kraken, OKX and many more, with experimental Futures support.
*   **Backtesting & Optimization:** Utilize backtesting, machine learning, and adaptive prediction modeling with FreqAI to refine your trading strategies.
*   **WebUI & Telegram Integration:** Manage your bot easily through a built-in WebUI and Telegram commands.
*   **Data Analysis Tools:** Leverage plotting and reporting features for profit/loss and performance analysis.
*   **Dry-run Mode:** Test your strategies risk-free with dry-run functionality.

## Disclaimer

This software is for educational purposes only. Trade at your own risk. The authors and affiliates are not responsible for your trading outcomes. Always test in dry-run before live trading. Familiarity with coding and Python is recommended. Review the source code to understand the bot's mechanisms.

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
*   and potentially many others (see [CCXT](https://github.com/ccxt/ccxt/))

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Consult the [exchange specific notes](docs/exchanges.md) and [trading with leverage](docs/leverage.md) documentation for specific configurations.

### Community-tested Exchanges

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io), covering all aspects of the bot's functionality and usage.

## Quick Start

Get started quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/). For native installation, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

Manage your bot easily through Telegram. More details and the full command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

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
*   `feat/*`: Feature branches - use with caution.

## Support

### Help / Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.

### Bugs / Issues

Report bugs through the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue). Follow the issue template.

### Feature Requests

Submit feature requests in the [issue tracker](https://github.com/freqtrade/freqtrade/labels/enhancement), after checking for existing discussions.

### Pull Requests

Contribute by submitting pull requests.  Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md). Address [good first issues](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) to get familiar with the code.
Create PRs against the `develop` branch. Open an issue before starting any major new feature work.

## Requirements

### Up-to-date clock

Ensure your system clock is synchronized with an NTP server.

### Minimum Hardware Requirements

*   2GB RAM, 1GB disk space, 2vCPU.

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)