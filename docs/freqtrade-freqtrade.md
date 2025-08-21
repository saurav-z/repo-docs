# Freqtrade: Automate Your Crypto Trading Strategies with a Powerful Open-Source Bot

[Freqtrade](https://github.com/freqtrade/freqtrade) is a free and open-source cryptocurrency trading bot, enabling you to automate your trading strategies on various exchanges with powerful features and flexibility.

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

[![](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)](https://www.freqtrade.io)

## Key Features

*   **Supports Major Exchanges:** Trade on leading exchanges like Binance, Kraken, OKX, and many more via CCXT integration.
*   **Backtesting & Optimization:** Test strategies with historical data, and optimize them using machine learning for enhanced performance.
*   **Machine Learning Integration:** Leverage FreqAI for adaptive prediction modeling to build smart strategies.
*   **Dry-Run Mode:** Safely test your strategies without risking real capital.
*   **WebUI & Telegram Control:** Manage your bot via a built-in web interface and Telegram commands.
*   **Extensive Documentation:** Benefit from comprehensive documentation and a supportive community.
*   **Open Source & Customizable:** Modify and adapt the bot to your specific trading needs.

## Disclaimer

This software is for educational purposes only. **Use Freqtrade at your own risk; the authors and affiliates are not responsible for trading outcomes.** Always start with Dry-run mode and thoroughly understand the bot's functionality before using real funds. Coding and Python knowledge is recommended.

## Supported Exchanges

Freqtrade supports a wide range of exchanges. Refer to the [exchange-specific notes](docs/exchanges.md) for configurations.

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
*   ...and many others supported by CCXT.

**Community-Tested Exchanges:**

*   Bitvavo
*   Kucoin

**Experimental Futures Exchanges:**

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Please review the [exchange-specific notes](docs/exchanges.md) and [trading with leverage](docs/leverage.md) documentation.

## Documentation

For detailed information, visit the complete documentation on the [Freqtrade website](https://www.freqtrade.io).

## Quick Start

Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for rapid setup. Alternatively, refer to the [Installation documentation](https://www.freqtrade.io/en/stable/installation/) for native installation methods.

## Basic Usage

### Bot Commands

```bash
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

Use Telegram to control your bot:

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

Full command list: [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

## Development Branches

*   `develop`: Contains new features and potential breaking changes.
*   `stable`: Latest stable release.
*   `feat/*`: Feature branches for ongoing development.

## Support

### Help / Discord

Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support and community engagement.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs and issues via the issue tracker.

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Share your ideas for improving the bot.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to Freqtrade by submitting pull requests.
Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)
before contributing.
Issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) can be good first contributions.
**Important:** Always create your PR against the `develop` branch, not `stable`.

## Requirements

### Up-to-date clock

The clock must be accurate, synchronized to a NTP server frequently to avoid communication errors with exchanges.

### Minimum Hardware

*   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)