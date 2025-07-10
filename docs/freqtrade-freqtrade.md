# Freqtrade: Your Open-Source Crypto Trading Bot for Automated Profits

**Freqtrade** is a powerful, free, and open-source crypto trading bot built in Python that allows you to automate your trading strategies and optimize your cryptocurrency investments.  Take control of your crypto trading – visit the [Freqtrade GitHub Repository](https://github.com/freqtrade/freqtrade) to get started!

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Comprehensive Exchange Support:** Integrates with a wide range of major cryptocurrency exchanges, including Binance, OKX, Bybit, and more.
*   **Backtesting & Optimization:** Test your strategies with backtesting and optimize them using machine learning, ensuring data-driven decision-making.
*   **Strategy Flexibility:** Build custom strategies with Python, tailoring them to your specific trading goals.
*   **FreqAI Adaptive Prediction Modeling:** Leverage self-training machine learning to adapt to market dynamics.
*   **User-Friendly Interface:** Control and monitor your bot through a built-in web UI and Telegram integration.
*   **Risk Management:** Utilize dry-run mode, whitelists/blacklists, and performance reports to manage risk effectively.
*   **Extensive Documentation & Community:** Benefit from detailed documentation, community support, and active development.

## Getting Started

1.  **Installation:** Consult the [installation documentation](https://www.freqtrade.io/en/stable/installation/) for detailed instructions on how to install Freqtrade on your system.  Docker quickstart is also available.
2.  **Configuration:** Configure your bot by creating a configuration file, specifying your API keys, exchange settings, and trading strategies.
3.  **Dry-Run:** Test your strategies in dry-run mode to simulate trades without risking real capital.
4.  **Live Trading:** Once you're confident in your strategy, enable live trading to automate your cryptocurrency investments.

## Disclaimer

This software is for educational purposes only. Do not risk money which
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS
AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not
hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchanges

Freqtrade supports a wide range of cryptocurrency exchanges, including (but not limited to):

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

Please read the [exchange specific notes](docs/exchanges.md) for any specific configurations needed.

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Please make sure to read the [exchange specific notes](docs/exchanges.md), as well as the [trading with leverage](docs/leverage.md) documentation before diving in.

### Community tested

Exchanges confirmed working by the community:

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io) to guide you through every step.

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

Telegram integration provides convenient control over your bot:

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

## Development branches

*   `develop`:  Branch with new features and potential breaking changes.
*   `stable`:  Branch containing the latest stable release.
*   `feat/*`: Feature branches for specific development.

## Support & Community

### Help / Discord

Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support, discussions, and to connect with fellow users.

### Bugs / Issues

Report bugs or issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### Feature Requests

Suggest new features by submitting a [feature request](https://github.com/freqtrade/freqtrade/issues/new/choose).

### Pull Requests

Contribute to the project by submitting [pull requests](https://github.com/freqtrade/freqtrade/pulls).  Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidance.

## Requirements

### Up-to-date clock

An accurate clock synchronized to a NTP server is essential to avoid communication issues with exchanges.

### Minimum hardware required

- Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software requirements

-   Python >= 3.11
-   pip
-   git
-   TA-Lib
-   virtualenv (Recommended)
-   Docker (Recommended)