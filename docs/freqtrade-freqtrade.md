# Freqtrade: Your Free and Open Source Crypto Trading Bot

Freqtrade empowers you to automate your cryptocurrency trading strategies with a robust, open-source platform.  Visit the original repository for more information: [Freqtrade on GitHub](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

**Disclaimer:** *This software is for educational purposes only.  Trade at your own risk.  Start with Dry-run to understand the software before risking real capital.*

## Key Features

*   **Open Source & Free:**  Leverage a community-driven project with full control and transparency.
*   **Multi-Exchange Support:** Trade on major cryptocurrency exchanges.
*   **Backtesting & Optimization:** Test your strategies with historical data and optimize them using machine learning.
*   **Automated Trading:**  Automate your buy/sell strategies, freeing up your time.
*   **Dry-Run Mode:** Safely test your strategies without risking real funds.
*   **Telegram & WebUI Control:** Monitor and manage your bot via Telegram or the built-in web interface.
*   **Machine Learning:**  Use adaptive prediction modeling, via FreqAI.
*   **Advanced Analysis Tools:** Access performance reports, profit/loss calculations, and more.
*   **Python-Based:** Built on Python, making it easily customizable and extensible.

## Supported Exchanges

Freqtrade supports a wide range of cryptocurrency exchanges.  Check the [exchange specific notes](docs/exchanges.md) for details on configuration.

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX / MyOKX (OKX EEA)
*   And many others via CCXT - community tested on Bitvavo, Kucoin

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Getting Started

### Quick Start

Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for the easiest setup.

### Installation

For native installations, see the detailed [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io) to guide you through setup, configuration, and advanced features.

## Basic Usage & Commands

*   **Bot Commands:**
    *   `trade`: Trade module.
    *   `create-userdir`: Create user-data directory.
    *   `new-config`: Create new config
    *   `show-config`: Show resolved config
    *   `new-strategy`: Create new strategy
    *   `download-data`: Download backtesting data.
    *   `convert-data`: Convert candle (OHLCV) data from one format to another.
    *   `convert-trade-data`: Convert trade data from one format to another.
    *   `trades-to-ohlcv`: Convert trade data to OHLCV data.
    *   `list-data`: List downloaded data.
    *   `backtesting`: Backtesting module.
    *   `backtesting-show`: Show past Backtest results
    *   `backtesting-analysis`: Backtest Analysis module.
    *   `hyperopt`: Hyperopt module.
    *   `hyperopt-list`: List Hyperopt results
    *   `hyperopt-show`: Show details of Hyperopt results
    *   `list-exchanges`: Print available exchanges.
    *   `list-markets`: Print markets on exchange.
    *   `list-pairs`: Print pairs on exchange.
    *   `list-strategies`: Print available strategies.
    *   `list-hyperoptloss`: Print available hyperopt loss functions.
    *   `list-freqaimodels`: Print available freqAI models.
    *   `list-timeframes`: Print available timeframes for the exchange.
    *   `show-trades`: Show trades.
    *   `test-pairlist`: Test your pairlist configuration.
    *   `convert-db`: Migrate database to different system
    *   `install-ui`: Install FreqUI
    *   `plot-dataframe`: Plot candles with indicators.
    *   `plot-profit`: Generate plot showing profits.
    *   `webserver`: Webserver module.
    *   `strategy-updater`: updates outdated strategy files to the current version
    *   `lookahead-analysis`: Check for potential look ahead bias.
    *   `recursive-analysis`: Check for potential recursive formula issue.

*   **Telegram Commands:** Control the bot via Telegram (setup required - see the documentation).
    *   `/start`: Starts trading.
    *   `/stop`: Stops trading.
    *   `/stopentry`: Stops entering new trades.
    *   `/status <trade_id>|[table]`: Lists open trades.
    *   `/profit [<n>]`: Lists cumulative profit.
    *   `/profit_long [<n>]`: Lists cumulative profit from all finished long trades, over the last n days.
    *   `/profit_short [<n>]`: Lists cumulative profit from all finished short trades, over the last n days.
    *   `/forceexit <trade_id>|all`: Instantly exits a trade.
    *   `/fx <trade_id>|all`: Alias to `/forceexit`
    *   `/performance`: Show performance of each finished trade grouped by pair
    *   `/balance`: Show account balance.
    *   `/daily <n>`: Shows profit/loss per day.
    *   `/help`: Shows help.
    *   `/version`: Shows version.

## Development Branches

*   `develop`:  The branch with new features.  May contain breaking changes.
*   `stable`:  The latest stable release.
*   `feat/*`:  Feature branches - test specific features here.

## Support & Community

*   **Discord:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support and community discussions.
*   **Issues:** Report bugs and suggest features on the [GitHub Issues page](https://github.com/freqtrade/freqtrade/issues).
*   **Pull Requests:** Contribute to the project by submitting [Pull Requests](https://github.com/freqtrade/freqtrade/pulls).  Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) before submitting.
*   **Good First Issues:** Consider contributing to issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) to get familiar with the codebase.

## Requirements

### System Requirements

*   **Clock Accuracy:** Ensure your system clock is synchronized with a NTP server.
*   **Minimal Hardware:**
    *   2GB RAM (recommended)
    *   1GB disk space
    *   2 vCPUs

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)