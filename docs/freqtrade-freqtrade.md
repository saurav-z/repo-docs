# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, free, and open-source crypto trading bot that empowers you to automate your trading strategies. ([See the original repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

## Key Features

*   **Supports Major Exchanges:** Trade on popular exchanges like Binance, Kraken, OKX, and more.
*   **Backtesting & Optimization:** Test your strategies with backtesting and optimize them using machine learning.
*   **Dry-Run Mode:** Safely test strategies without risking real funds.
*   **WebUI and Telegram Integration:** Manage and monitor your bot through a built-in web interface or Telegram commands.
*   **Machine Learning:** Leverage freqAI for adaptive prediction modeling and smart strategy development.
*   **Open Source:** Free and open-source software, with community support and contributions.

## Supported Exchanges

Freqtrade supports a wide array of exchanges:

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

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

For more details, please read the [exchange specific notes](docs/exchanges.md), and the [trading with leverage](docs/leverage.md) documentation

### Community Tested Exchanges:

*   Bitvavo
*   Kucoin

## Disclaimer

This software is for educational purposes only.  Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

We strongly recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.

## Documentation

Comprehensive documentation is available at the [freqtrade website](https://www.freqtrade.io).

## Quick Start

Refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for a fast setup.
For native installation, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage & Commands

Explore available commands for the bot to trade, backtest, and optimize strategies.

### Bot commands
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

*   `develop`: Latest features, may have breaking changes.
*   `stable`: Latest stable release.
*   `feat/*`: Feature branches.

## Support & Community

*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   **Issues:** Report bugs and issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:**  Suggest improvements via the [feature request](https://github.com/freqtrade/freqtrade/labels/enhancement) section.
*   **Pull Requests:** Contribute to the project by submitting pull requests.
    Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)

## Requirements

*   **Up-to-date clock:** Essential for accurate exchange communication (NTP server).
*   **Minimum Hardware:** 2GB RAM, 1GB disk space, 2vCPU (recommended).
*   **Software:** Python >= 3.11, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended)