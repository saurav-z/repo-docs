# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, free, and open-source crypto trading bot that empowers you to automate your trading strategies and navigate the volatile world of cryptocurrencies, giving you an edge in the market. **[Check out the original repo](https://github.com/freqtrade/freqtrade).**

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

## Key Features

*   **Automated Trading:** Execute trading strategies 24/7, optimizing for profit.
*   **Backtesting & Optimization:** Test your strategies with backtesting, and refine them using machine learning.
*   **Machine Learning Integration:** Enhance your trading strategies with adaptive prediction modeling via FreqAI.
*   **Wide Exchange Support:** Compatible with major crypto exchanges, including Binance, OKX, and Bybit.
*   **User-Friendly Interface:** Manage your bot with a built-in WebUI or Telegram integration.
*   **Flexible Strategy Development:** Design and test your own trading strategies using Python.
*   **Risk Management:** Includes features like dry-run mode, profit/loss display, and performance reporting to help manage your trades.
*   **Community Driven:** Benefit from a vibrant community and comprehensive documentation.

## Disclaimer

This software is for educational purposes only. Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

## Supported Exchanges

Freqtrade supports a wide range of exchanges, with new integrations being continuously added.  Please refer to the [exchange specific notes](docs/exchanges.md) for setup details.

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
*   [and many others](https://github.com/ccxt/ccxt/)

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Documentation

The complete documentation can be found on the [freqtrade website](https://www.freqtrade.io).

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

### Telegram RPC commands

Telegram is not mandatory. However, this is a great way to control your bot. More details and the full command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

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

The project is structured with two main branches:

*   `develop`: The branch for new features, which may include breaking changes.
*   `stable`: The branch for the latest stable release.

## Support

### Help / Discord

For any questions or to engage with the community, join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7).

### Bugs / Issues

Report bugs and issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
Always follow the template guide and maintain github's [community policy](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct).

### Feature Requests

Suggest new features on the [feature requests](https://github.com/freqtrade/freqtrade/labels/enhancement).
If it hasn't been requested, please
[create a new request](https://github.com/freqtrade/freqtrade/issues/new/choose)
and ensure you follow the template guide so that it does not get lost
in the bug reports.

### Pull Requests

Contribute by submitting pull requests to the `develop` branch.
Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) to understand the requirements.
Explore issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) to get familiar with the codebase.

**Note:**  Before starting significant feature work, open an issue describing your plans or discuss them on the [discord](https://discord.gg/p7nuUNVfP7).

**Important:**  Always create your PR against the `develop` branch, not `stable`.

## Requirements

### Up-to-date Clock

Ensure your system clock is accurate and synchronized frequently to an NTP server.

### Minimum Hardware

*   Minimal (advised): 2GB RAM, 1GB disk space, 2vCPU

### Software

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)