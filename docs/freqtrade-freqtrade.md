# Freqtrade: Your Open-Source Crypto Trading Bot for Automated Trading

Freqtrade is a powerful, open-source crypto trading bot designed to automate your trading strategies across major exchanges. Visit the [Freqtrade GitHub repository](https://github.com/freqtrade/freqtrade) to learn more.

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

**Key Features:**

*   **Automated Trading:** Automate your crypto trading strategies.
*   **Multi-Exchange Support:** Supports major cryptocurrency exchanges.
*   **Backtesting:** Test your strategies with historical data.
*   **Strategy Optimization:** Optimize strategies with machine learning.
*   **Dry-Run Mode:** Test your bot without risking real money.
*   **WebUI & Telegram Control:** Manage and monitor your bot via a web interface or Telegram.
*   **Built-in Reporting:** Monitor performance metrics, including profit and loss in fiat currency.
*   **FreqAI Integration:** Adaptive prediction modeling for building smart strategies.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not
hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchanges

Freqtrade supports a wide range of cryptocurrency exchanges. Please read the [exchange specific notes](docs/exchanges.md) for setup details.

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
*   ... and potentially many others (check [CCXT](https://github.com/ccxt/ccxt/) for a complete list).

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Please make sure to read the [exchange specific notes](docs/exchanges.md), as well as the [trading with leverage](docs/leverage.md) documentation before diving in.

### Community Tested Exchanges

Exchanges confirmed working by the community:

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io), offering detailed guidance on installation, configuration, and usage.

## Quick Start

Get up and running quickly with our [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/). Native installation instructions are available on the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Usage & Commands

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

Manage your bot via Telegram. See the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for the full command list.

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

*   `develop` - The branch for new features (may contain breaking changes).
*   `stable` - The latest stable release branch.
*   `feat/*` - Feature branches (for testing specific features).

## Support and Community

*   Join the Freqtrade [Discord Server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   Report bugs or issues in the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   Suggest new features in the [feature request section](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   Contribute with [Pull Requests](https://github.com/freqtrade/freqtrade/pulls) following the [Contributing guidelines](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).

## Requirements

### Time Synchronization

Ensure your system clock is accurate and synchronized with an NTP server.

### Minimum Hardware Requirements

*   2GB RAM, 1GB disk space, 2 vCPU (recommended).

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)