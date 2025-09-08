# Freqtrade: Your Open-Source Crypto Trading Bot

**Automate your crypto trading with Freqtrade, a powerful, free, and open-source bot designed for all major exchanges.** ([Visit the Repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade empowers you to automate your cryptocurrency trading strategies. It's built on Python and offers a comprehensive suite of tools for backtesting, optimization, and real-time trading.  Control your bot through Telegram or a user-friendly webUI.

## Key Features

*   **Open Source & Free:**  Use and customize Freqtrade without any cost.
*   **Multi-Exchange Support:** Works with major exchanges including Binance, Bybit, OKX, and more (see supported exchanges below).
*   **Backtesting & Strategy Optimization:**  Test your strategies and optimize parameters using machine learning.
*   **FreqAI Integration:** Build advanced strategies using FreqAI's adaptive prediction modeling.
*   **Built-in Web UI:**  Monitor and manage your bot via a convenient web interface.
*   **Telegram Integration:**  Control and receive updates through Telegram.
*   **Dry-Run Mode:** Test your strategies without risking real funds.
*   **Profit/Loss Tracking:**  View profits and losses in your preferred fiat currency.

## Supported Exchanges

Freqtrade supports a wide range of exchanges, with more being added regularly.

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
*   ...and more.  See the [exchange specific notes](docs/exchanges.md) for details.

### Supported Futures Exchanges (experimental)
*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Important Considerations

*   **Disclaimer:**  This software is for educational purposes.  Trade at your own risk and understand the bot's functionality before using real funds.
*   **Python & Coding Knowledge Recommended:**  Familiarity with Python and coding is beneficial for customization and troubleshooting.
*   **Documentation is Key:**  Refer to the [Freqtrade documentation](https://www.freqtrade.io) for detailed information.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For native installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

*   **Bot commands** See the detailed list in the original README.
*   **Telegram RPC commands:**  Control your bot using Telegram.  See the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for a complete list of commands.

## Development Branches

*   `develop`:  Branch with new features; may have breaking changes.
*   `stable`:  Latest stable release.

## Contribute

We welcome contributions!

*   [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue) - Report bugs and issues.
*   [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement) - Suggest new features.
*   [Pull Requests](https://github.com/freqtrade/freqtrade/pulls) - Contribute code, improve documentation, and more!  See the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).

## Support

*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for help and discussions.

## Requirements

*   **Up-to-date clock:** Accurate time synchronization is crucial.
*   **Minimum hardware:** 2GB RAM, 1GB disk space, 2vCPU (advised).
*   **Software:** Python 3.11+, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended).