# Freqtrade: Your Open-Source Crypto Trading Bot

Automate your crypto trading strategies with Freqtrade, a powerful and open-source Python-based bot. [Learn more at the original repository](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade offers a comprehensive suite of tools for automated crypto trading, including backtesting, strategy optimization, and support for major exchanges, all managed through Telegram or a web UI.

## Key Features

*   **Supports Major Exchanges:** Binance, Bitmart, BingX, Bybit, Gate.io, HTX, Kraken, OKX, and more (see [exchange specific notes](docs/exchanges.md)).
*   **Backtesting & Optimization:** Evaluate and refine your trading strategies with backtesting, and leverage machine learning for optimization.
*   **FreqAI Integration:** Build adaptive strategies with machine learning that learns from the market (self-trains).
*   **Telegram & WebUI Control:** Easily manage your bot through Telegram commands or the built-in web interface.
*   **Dry-run Mode:** Test your strategies without risking real capital.
*   **Profit/Loss Reporting:** View profit and loss in fiat currency.
*   **Detailed Performance Reports:** Get insights into your trades with performance status reports.
*   **Based on Python 3.11+**: For botting on any operating system - Windows, macOS and Linux.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

We strongly recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).

For other installation methods, please refer to the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

*   `trade`: Initiate trading.
*   `create-userdir`: Create user directory.
*   `new-config`: Create new configuration file.
*   `show-config`: Display the active configuration.
*   `new-strategy`: Create a new trading strategy.
*   `download-data`: Download historical trading data.
*   `convert-data`: Convert candle data formats.
*   `convert-trade-data`: Convert trade data formats.
*   `trades-to-ohlcv`: Convert trade data to OHLCV format.
*   `list-data`: List downloaded data files.
*   `backtesting`: Run backtesting.
*   `backtesting-show`: View past backtesting results.
*   `backtesting-analysis`: Perform backtesting analysis.
*   `hyperopt`: Utilize the hyperopt module.
*   `hyperopt-list`: List hyperopt results.
*   `hyperopt-show`: Display details of hyperopt results.
*   `list-exchanges`: Print available exchanges.
*   `list-markets`: List markets on exchanges.
*   `list-pairs`: List trading pairs.
*   `list-strategies`: List available trading strategies.
*   `list-hyperoptloss`: List available hyperopt loss functions.
*   `list-freqaimodels`: List available freqAI models.
*   `list-timeframes`: List available timeframes.
*   `show-trades`: Display current trades.
*   `test-pairlist`: Test your pairlist configuration.
*   `convert-db`: Convert database to a different system.
*   `install-ui`: Install FreqUI.
*   `plot-dataframe`: Plot candles with indicators.
*   `plot-profit`: Generate profit plots.
*   `webserver`: Run the webserver module.
*   `strategy-updater`: Updates outdated strategy files.
*   `lookahead-analysis`: Run lookahead analysis.
*   `recursive-analysis`: Run recursive analysis.

### Telegram RPC Commands

Manage your bot directly from Telegram. See the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for a full command list.

*   `/start`: Start the trader.
*   `/stop`: Stop the trader.
*   `/stopentry`: Prevent new trade entries.
*   `/status <trade_id>|[table]`: List open trades.
*   `/profit [<n>]`: List cumulative profit from all finished trades.
*   `/profit_long [<n>]`: List cumulative profit from all finished long trades.
*   `/profit_short [<n>]`: List cumulative profit from all finished short trades.
*   `/forceexit <trade_id>|all`: Force an exit of a trade.
*   `/fx <trade_id>|all`: Alias to `/forceexit`
*   `/performance`: Show performance of finished trades.
*   `/balance`: Show account balance.
*   `/daily <n>`: Shows profit or loss per day.
*   `/help`: Display help information.
*   `/version`: Show the bot version.

## Development Branches

*   `develop`: The branch for new features, but it may contain breaking changes.
*   `stable`: The latest stable release branch.
*   `feat/*`: Feature branches in active development.

## Support

### Help / Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community engagement.

### Issues

Report bugs and issues in the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### Feature Requests

Suggest new features in the [enhancement section](https://github.com/freqtrade/freqtrade/labels/enhancement).

### Pull Requests

Contribute to Freqtrade by submitting pull requests to the `develop` branch.  See the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Requirements

### Up-to-date Clock

Ensure your system clock is accurate and synchronized to a NTP server.

### Minimum Hardware

*   2GB RAM, 1GB disk space, 2vCPU (Recommended)

### Software

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)