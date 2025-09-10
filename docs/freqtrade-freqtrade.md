# Freqtrade: Open-Source Crypto Trading Bot

**Automate your crypto trading with Freqtrade, the free and open-source bot offering powerful features for maximizing your trading potential.** ([Visit the original repository](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade is a versatile and customizable crypto trading bot written in Python, designed to help you navigate the volatile crypto markets. Offering a wide array of features, it empowers both beginners and experienced traders to automate their strategies and optimize their trading performance.

## Key Features

*   **Automated Trading:** Execute trades automatically based on pre-defined strategies.
*   **Backtesting & Optimization:** Backtest strategies with historical data and optimize them using machine learning for improved performance.
*   **Exchange Support:** Supports major cryptocurrency exchanges, including Binance, Bybit, and OKX.
*   **WebUI & Telegram Integration:** Manage and monitor your bot through a built-in web interface or via Telegram commands.
*   **Strategy Flexibility:** Create custom strategies using Python and a range of technical indicators.
*   **Machine Learning for Adaptive Predictions:** Utilizes FreqAI for self-training market adaptation.
*   **Risk Management:** Features like dry-run mode and blacklists help mitigate risks.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not
hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchanges

Freqtrade supports a wide range of cryptocurrency exchanges. Please read the [exchange specific notes](docs/exchanges.md) for important configuration details.

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
*   [and potentially many others](https://github.com/ccxt/ccxt/)

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

### Community Tested Exchanges

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [freqtrade website](https://www.freqtrade.io), providing detailed information on installation, configuration, and usage.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/). For other installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

Freqtrade offers a variety of commands for managing your bot, including trading, backtesting, and optimization. See the detailed list in the original README.

### Telegram RPC Commands

Control your bot with ease using Telegram commands. A full list of commands is available in the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/).

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

*   `develop`: For features and potential breaking changes.
*   `stable`: Contains the latest stable release.
*   `feat/*`: Feature branches actively being worked on.

## Support

### Help / Discord

Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support and community discussions.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs through the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Suggest new features using the [feature request section](https://github.com/freqtrade/freqtrade/labels/enhancement).

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute to the project by submitting pull requests. Please read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) before contributing.