# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, open-source crypto trading bot that empowers you to automate your trading strategies and navigate the volatile crypto market, easily accessible at [https://github.com/freqtrade/freqtrade](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

## Key Features

*   **Automated Trading:** Automate your trading strategies across multiple crypto exchanges.
*   **Backtesting:** Test strategies with historical data to optimize performance.
*   **Strategy Optimization:** Fine-tune buy/sell parameters using machine learning.
*   **FreqAI Integration:** Leverage FreqAI for adaptive prediction modeling to build smart trading strategies.
*   **Exchange Support:** Compatible with major exchanges like Binance, Bybit, OKX, and more.
*   **WebUI and Telegram Control:** Manage your bot through an intuitive web interface or Telegram commands.
*   **Dry-Run Mode:** Simulate trades without risking real money.
*   **Performance Reporting:** Monitor profits and losses in fiat currency.
*   **Python 3.11+:** Supports trading on any operating system (Windows, macOS, and Linux).

## Disclaimer

This software is intended for educational purposes only. Trade at your own risk and never invest more than you can afford to lose. Always run the bot in Dry-run mode first, and have a solid understanding of its operations before trading with real funds.

## Supported Exchanges

Freqtrade supports a wide range of exchanges. See the [exchange specific notes](docs/exchanges.md) for detailed configurations.

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
*   [Bitvavo](https://bitvavo.com/)
*   [Kucoin](https://www.kucoin.com/)

### Supported Futures Exchanges (experimental)

*   [Binance](https://www.binance.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [OKX](https://okx.com/)
*   [Bybit](https://bybit.com/)

Read the [exchange specific notes](docs/exchanges.md) and the [trading with leverage](docs/leverage.md) documentation before diving in.

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io) to guide you.

## Quick Start

Refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for a fast setup.

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

Use Telegram to control the bot. Full command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

*   `/start`: Start the trader.
*   `/stop`: Stop the trader.
*   `/stopentry`: Stop entering new trades.
*   `/status <trade_id>|[table]`: List open trades.
*   `/profit [<n>]`: List cumulative profit from finished trades.
*   `/forceexit <trade_id>|all`: Instantly exit a trade.
*   `/fx <trade_id>|all`: Alias to `/forceexit`
*   `/performance`: Show performance per pair.
*   `/balance`: Show account balance per currency.
*   `/daily <n>`: Shows profit or loss per day.
*   `/help`: Show help.
*   `/version`: Show version.

## Development Branches

*   `develop`: New features and potential breaking changes.
*   `stable`: Latest stable release.
*   `feat/*`: Feature branches.

## Support

### Help / Discord

Get support and connect with the community on the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7).

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs and search for existing issues.

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest improvements and new features.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute by submitting pull requests. Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) before submitting.