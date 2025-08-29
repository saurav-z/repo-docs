# Freqtrade: Open-Source Crypto Trading Bot for Automated Profit

Freqtrade is a powerful, free, and open-source crypto trading bot that empowers users to automate their trading strategies across multiple exchanges. [View the original repo](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

## Key Features

*   **Supports Major Exchanges:** Trade on popular exchanges like Binance, Bybit, and OKX.
*   **Backtesting & Optimization:** Test strategies with historical data and optimize with machine learning.
*   **Automated Trading:** Execute trades automatically based on your predefined strategies.
*   **WebUI and Telegram Integration:** Manage your bot through a user-friendly web interface or Telegram commands.
*   **Adaptive Prediction Modeling (FreqAI):** Utilize machine learning for smart, self-training strategies.
*   **Dry-run Mode:** Test your strategies without risking real capital.
*   **Profit/Loss Tracking:** Monitor your trading performance with profit/loss displayed in fiat currency.

## Getting Started

Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for the quickest setup.  Alternatively, explore the [Installation documentation](https://www.freqtrade.io/en/stable/installation/) for other installation methods.

## Supported Exchanges

Freqtrade supports a wide range of crypto exchanges.  Please refer to the [exchange specific notes](docs/exchanges.md) for configuration details.

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid
*   Kraken
*   OKX / MyOKX
*   Bitvavo (Community Tested)
*   Kucoin (Community Tested)
*   And potentially many others - see [CCXT compatibility](https://github.com/ccxt/ccxt/)

## Disclaimer

This software is for educational purposes only.  Always use at your own risk. Start with Dry-run mode before using real money.

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).

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

## Development

*   `develop` - This branch has often new features, but might also contain breaking changes. We try hard to keep this branch as stable as possible.
*   `stable` - This branch contains the latest stable release. This branch is generally well tested.
*   `feat/*` - These are feature branches, which are being worked on heavily. Please don't use these unless you want to test a specific feature.

## Support & Community

*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and discussions.
*   **Issues:** Report bugs and issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Submit feature requests on the [enhancement label](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   **Pull Requests:** Contribute to the project by submitting pull requests. See the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).

## Requirements

*   Accurate Time: Ensure an accurate clock synchronized to an NTP server.
*   System Requirements:
    *   Minimal (advised): 2GB RAM, 1GB disk space, 2vCPU
*   Software Requirements:
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)