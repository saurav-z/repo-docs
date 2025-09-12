# Freqtrade: Your Open-Source Crypto Trading Bot

**Freqtrade is a powerful, open-source crypto trading bot that helps you automate your trading strategies with advanced features and comprehensive exchange support. [Get started on Github!](https://github.com/freqtrade/freqtrade)**

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade empowers you to automate your crypto trading strategies with a versatile and customizable bot. Designed to work with various exchanges and offering a range of features including backtesting, strategy optimization, and convenient control via Telegram and webUI.

## Key Features:

*   **Free and Open Source:** Leverage a community-driven project.
*   **Python-Based:** Built on Python 3.11+, ensuring cross-platform compatibility (Windows, macOS, Linux).
*   **Comprehensive Exchange Support:** Integrates with major exchanges, including Binance, Bybit, OKX, and more.
*   **Backtesting:** Simulate your strategies before deploying real capital.
*   **Strategy Optimization:** Use machine learning to optimize your trading parameters.
*   **Adaptive Prediction Modeling:** Leverage FreqAI for self-training strategies.
*   **WebUI & Telegram Control:** Manage and monitor your bot through a built-in web interface or Telegram commands.
*   **Risk Management:** Includes dry-run mode, whitelist/blacklist functionality, and fiat profit/loss display.

## Supported Exchanges:

Freqtrade supports a wide array of exchanges, with notes for each exchange [here](docs/exchanges.md)

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
*   And potentially many others via [CCXT](https://github.com/ccxt/ccxt/).

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

### Community Verified Exchanges

*   Bitvavo
*   Kucoin

## Getting Started:

*   **Quickstart:** Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for rapid setup.
*   **Installation:** Consult the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/) for native installation methods.
*   **Documentation:** Ensure you understand the bot before use, see the complete documentation on the [freqtrade website](https://www.freqtrade.io).

## Basic Usage:

*   **Bot Commands:** Use a wide range of commands for backtesting, hyperopt, and strategy management. Detailed [commands](#basic-usage) listed below.
*   **Telegram RPC Commands:** Control your bot and view key information directly through Telegram. See the full list in the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

## Development and Contribution:

*   **Development Branches:** Follow the `develop` branch for the newest features (may have breaking changes), and `stable` for the latest stable release.
*   **Support:** Join the [Discord server](https://discord.gg/p7nuUNVfP7) for support.
*   **Issues/Bugs:** Report issues via the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Submit enhancement ideas through the [feature request](https://github.com/freqtrade/freqtrade/labels/enhancement) section.
*   **Contributing:** Contribute to the project by submitting [Pull Requests](https://github.com/freqtrade/freqtrade/pulls). Read the [CONTRIBUTING document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Requirements

*   **Up-to-date clock:** Essential for exchange communication, synchronize with a NTP server.
*   **Minimum Hardware:**
    *   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU
*   **Software Requirements:**
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)

**Disclaimer:** *This software is for educational purposes only. Trade at your own risk.*