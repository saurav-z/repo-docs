# Freqtrade: Your Free and Open-Source Crypto Trading Bot

[Freqtrade](https://github.com/freqtrade/freqtrade) is a powerful and versatile crypto trading bot designed to automate your trading strategies on various exchanges.

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

## Key Features

*   **Automated Trading:** Execute trades based on your custom strategies.
*   **Backtesting:** Evaluate strategy performance using historical data.
*   **Strategy Optimization:** Utilize machine learning for parameter tuning.
*   **Adaptive Prediction Modeling**: Build smart strategies with FreqAI via adaptive machine learning.
*   **Multi-Exchange Support:** Compatible with major cryptocurrency exchanges.
*   **Web UI & Telegram Integration:** Monitor and control your bot through a web interface or Telegram.
*   **Risk Management Tools:**  Includes features like dry-run mode and blacklist/whitelist options.
*   **Python 3.11+:** Runs on Windows, macOS, and Linux.

## Disclaimer

This software is for educational purposes only. **Use Freqtrade at your own risk; the developers are not responsible for your trading results.** Always start in dry-run mode and understand the bot's functionality before using real funds.

## Supported Exchanges

Freqtrade supports a wide range of exchanges; see the [exchange-specific notes](docs/exchanges.md) for details.

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX & MyOKX (OKX EEA)
*   Potentially many others (check [CCXT](https://github.com/ccxt/ccxt/))

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Also community tested exchanges confirmed to be working include:
*   Bitvavo
*   Kucoin

## Documentation

Consult the comprehensive [Freqtrade documentation](https://www.freqtrade.io) for detailed information.

## Quick Start

Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for a fast setup.  You can also find native installation instructions on the [installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

See the [Freqtrade documentation](https://www.freqtrade.io/en/stable/usage/) for the full command list.

### Telegram RPC Commands

Manage your bot via Telegram; see the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for a list of commands.

*   `/start`, `/stop`, `/stopentry`: Control trade execution.
*   `/status <trade_id>|[table]`: View trade status.
*   `/profit [<n>]`, `/profit_long [<n>]`, `/profit_short [<n>]`: Review profit/loss.
*   `/forceexit <trade_id>|all`: Exit trades.
*   `/balance`: Show account balances.
*   `/daily <n>`: View daily profit/loss.
*   `/help`, `/version`:  Get assistance and version information.

## Development Branches

*   `develop`:  Active development, may contain new features and breaking changes.
*   `stable`:  Latest stable release, well-tested.
*   `feat/*`:  Feature branches for specific feature development.

## Support

### Help / Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

Report bugs by searching the issue tracker or [creating a new issue](https://github.com/freqtrade/freqtrade/issues/new/choose).

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Submit feature requests by checking the issue tracker or [creating a new request](https://github.com/freqtrade/freqtrade/issues/new/choose).

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Contribute by submitting pull requests; review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).  Contribute by improving documentation, or contributing to [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue).

**Important:**  Create PRs against the `develop` branch.

## Requirements

### Up-to-date clock

The clock must be accurate, synchronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Minimum Hardware

*   2GB RAM, 1GB disk space, 2vCPU

### Software

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)