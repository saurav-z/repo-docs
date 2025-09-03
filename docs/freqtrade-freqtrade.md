# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, open-source crypto trading bot written in Python that empowers you to automate your trading strategies.  [Explore the Freqtrade project on GitHub](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

## Key Features

*   **Supports Major Exchanges**: Trade on a wide range of crypto exchanges, including Binance, Bybit, Kraken, and OKX.  [See supported exchanges](#supported-exchange-marketplaces).
*   **Backtesting and Optimization**: Test your strategies with backtesting and optimize them using machine learning with real exchange data.
*   **Automated Trading**: Automate your trading strategies and manage the bot using Telegram or a webUI.
*   **Dry-Run Mode**:  Test your strategies without risking real capital with the dry-run mode.
*   **Machine Learning**:  Take advantage of adaptive prediction modeling with FreqAI.
*   **Comprehensive Tools**: Includes plotting, money management, and performance reporting.
*   **Open Source and Community Driven**: Benefit from a vibrant community and actively developed project.

## Disclaimer

This software is for educational purposes only.  Trade with caution and only risk funds you can afford to lose.  Understand how the bot works before trading with real money.  The authors and affiliates are not responsible for your trading results.

## Supported Exchange Marketplaces

Freqtrade supports a wide variety of exchanges.  See the [exchange-specific notes](docs/exchanges.md) for special configurations needed.

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
*   Community-tested exchanges (Bitvavo, Kucoin)

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Documentation

Refer to the official [Freqtrade documentation](https://www.freqtrade.io) to understand how the bot works.

## Quick Start

Get started quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For native installation, see the [Installation documentation](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

Freqtrade offers a comprehensive set of commands.

**Core Commands:** `/start`, `/stop`, `/stopentry`, `/status`, `/profit`, `/forceexit`, `/performance`, `/balance`, `/daily`, `/help`, `/version`.

Explore the full command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/).

## Development and Contribution

The project is structured in two main branches:

*   `develop`: The branch with new features, potentially breaking changes.
*   `stable`: The latest stable release.

If you would like to contribute, please read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) and the [community policy](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct) before opening a Pull Request.

## Requirements

*   **Python 3.11+**
*   [pip](https://pip.pypa.io/en/stable/installing/)
*   [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
*   [TA-Lib](https://ta-lib.github.io/ta-lib-python/)
*   [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
*   [Docker](https://www.docker.com/products/docker) (Recommended)

### Minimum hardware requirements:

*   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU