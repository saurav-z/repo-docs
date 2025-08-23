# Freqtrade: Your Free and Open-Source Crypto Trading Bot

Freqtrade is a powerful, free, and open-source crypto trading bot written in Python, designed to automate your trading strategies and maximize your profits. [Explore the original repo](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

## Key Features

*   **Automated Trading:** Automate your crypto trading strategies 24/7.
*   **Broad Exchange Support:** Compatible with a wide range of major cryptocurrency exchanges, including Binance, Kraken, and OKX.
*   **Backtesting & Optimization:** Backtest strategies with historical data and optimize them using machine learning.
*   **Strategy Development:** Create and customize your own trading strategies using Python.
*   **Dry-Run Mode:** Test strategies in a risk-free dry-run mode before deploying real funds.
*   **Telegram & WebUI Integration:** Monitor and manage your bot via Telegram commands or the built-in web interface.
*   **FreqAI Integration:** Leverage adaptive machine learning methods for smart strategy building and market adaptation.
*   **Community Driven:** Benefit from a vibrant and active community for support and collaboration.

## Supported Exchanges

Freqtrade supports a growing list of cryptocurrency exchanges.  See the [exchange specific notes](docs/exchanges.md) for configuration details.

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
*   Bitvavo (Community Tested)
*   Kucoin (Community Tested)

**Experimental Futures Exchanges:**

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Disclaimer

This software is for educational purposes only.  **Trade at your own risk.** Always start with Dry-run and thoroughly understand the bot before using real money.  The developers and affiliates are not responsible for your trading results.

## Getting Started

### Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io), including setup guides, strategy development tutorials, and API references.

### Quickstart

The [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) provides the fastest way to get up and running.

### Installation

For other installation methods, see the [installation documentation](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

Freqtrade offers a variety of commands for managing your bot.

For example:

*   `/start`
*   `/stop`
*   `/status`
*   `/profit`

Explore the full command list in the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/).

## Development Branches

*   `develop`: Branch with new features (may contain breaking changes)
*   `stable`: Latest stable release

## Support and Community

*   **Discord:** Join the [Freqtrade Discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   **Issues:** Report bugs or issues in the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Suggest new features in the [feature requests](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   **Pull Requests:** Contribute to the project by submitting pull requests. Review the [CONTRIBUTING document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Requirements

*   **Accurate Clock:** Requires an accurate, NTP-synchronized clock.
*   **Minimum Hardware:**
    *   2GB RAM
    *   1GB disk space
    *   2 vCPU
*   **Software:**
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)