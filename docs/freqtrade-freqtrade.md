# Freqtrade: Your Open-Source Crypto Trading Bot for Automated Profits

Freqtrade is a free and open-source crypto trading bot that empowers you to automate your trading strategies.  [Check out the original repository](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade supports all major crypto exchanges and offers a comprehensive suite of tools for backtesting, strategy optimization, and automated trading. Manage your bot via Telegram or the built-in WebUI.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Automated Trading:** Execute trades 24/7 based on your custom strategies.
*   **Backtesting:** Test strategies against historical data to refine your approach.
*   **Strategy Optimization:** Leverage machine learning for parameter optimization.
*   **Adaptive Prediction Modeling:** Utilize FreqAI for self-training strategies.
*   **Multi-Exchange Support:** Trade on a variety of popular exchanges.
*   **WebUI and Telegram Control:** Easily manage and monitor your bot.
*   **Dry-Run Mode:** Simulate trading without risking real capital.
*   **Profit/Loss Reporting:** Track your performance in fiat currency.

## Disclaimer

**Risk Disclosure:**  Use this software at your own risk.  This software is for educational purposes and is not financial advice.  The authors and affiliates are not responsible for your trading outcomes. Always use "Dry-run" mode and understand the bot before trading with real funds.  Familiarity with Python and coding is recommended.

## Supported Exchanges

Freqtrade supports a wide range of cryptocurrency exchanges.  Refer to the [exchange specific notes](docs/exchanges.md) for any specific configurations.

*   [Binance](https://www.binance.com/)
*   [Bitmart](https://bitmart.com/)
*   [BingX](https://bingx.com/invite/0EM9RX)
*   [Bybit](https://bybit.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [HTX](https://www.htx.com/)
*   [Hyperliquid](https://hyperliquid.xyz/) (DEX)
*   [Kraken](https://kraken.com/)
*   [OKX](https://okx.com/)
*   [MyOKX](https://okx.com/) (OKX EEA)
*   [and many others](https://github.com/ccxt/ccxt/)

### Supported Futures Exchanges (experimental)

*   [Binance](https://www.binance.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [Hyperliquid](https://hyperliquid.xyz/) (DEX)
*   [OKX](https://okx.com/)
*   [Bybit](https://bybit.com/)

## Community-Tested Exchanges

The following exchanges are confirmed to be working by the community:

*   [Bitvavo](https://bitvavo.com/)
*   [Kucoin](https://www.kucoin.com/)

## Documentation

Comprehensive documentation is available to help you understand and use Freqtrade.
Find the complete documentation on the [freqtrade website](https://www.freqtrade.io).

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).

For native installation options, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

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

Use Telegram to manage your bot with these commands (see the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for full command list).

*   `/start`: Start trading.
*   `/stop`: Stop trading.
*   `/stopentry`: Stop new trades from entering.
*   `/status <trade_id>|[table]`: List trades.
*   `/profit [<n>]`: Show profit over the last *n* days.
*   `/profit_long [<n>]`: Show profit for long trades over the last *n* days.
*   `/profit_short [<n>]`: Show profit for short trades over the last *n* days.
*   `/forceexit <trade_id>|all`: Exit a trade.
*   `/fx <trade_id>|all`: Alias for `/forceexit`.
*   `/performance`: Show performance by pair.
*   `/balance`: Show account balance.
*   `/daily <n>`: Show daily profit/loss.
*   `/help`: Show help.
*   `/version`: Show version.

## Development Branches

*   `develop`: New features and potentially breaking changes.
*   `stable`: Latest stable release.
*   `feat/*`: Feature branches in development.

## Support

*   **Discord:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support.
*   **Issues:**  Report bugs and issues in the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:**  Suggest new features in the [feature request section](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   **Pull Requests:** Contribute to the project by submitting [pull requests](https://github.com/freqtrade/freqtrade/pulls).  Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) first. Create PRs against the `develop` branch.

## Requirements

*   **Accurate Clock:**  A synchronized and accurate clock (NTP).
*   **Hardware:**  Minimum of 2GB RAM, 1GB disk space, and 2 vCPU.
*   **Software:** Python 3.11+, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended).