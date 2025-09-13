# Freqtrade: Open-Source Crypto Trading Bot 🤖

**Automate your cryptocurrency trading strategies with Freqtrade, a powerful, free, and open-source bot designed for both beginners and experienced traders.** ([Back to the original repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade empowers you to develop and execute automated trading strategies on various cryptocurrency exchanges, providing robust backtesting, optimization tools, and real-time monitoring.

## Key Features

*   **Multi-Exchange Support:** Trade on major exchanges including Binance, Bybit, OKX, and more. (See [Supported Exchanges](#supported-exchange-marketplaces) below).
*   **Backtesting & Optimization:** Test and refine your strategies with historical data and optimize parameters using machine learning.
*   **FreqAI Integration:** Leverage adaptive machine learning for smart strategy building.
*   **User-Friendly Interface:** Manage your bot via a built-in web UI or Telegram integration.
*   **Risk Management:** Employ dry-run mode for safe testing, profit/loss tracking in fiat currency, and performance reporting.
*   **Community-Driven:** Benefit from an active community, extensive documentation, and ongoing development.
*   **Open Source & Customizable:** Tailor the bot to your specific trading style and needs.

## Disclaimer

*   This software is for educational purposes only.
*   Use this software at your own risk, and don't invest more than you can afford to lose.
*   Prior knowledge of coding and Python is advised.
*   Always start in Dry-Run mode.

## Supported Exchange Marketplaces

Freqtrade supports a wide range of cryptocurrency exchanges. Please consult the [exchange-specific notes](docs/exchanges.md) for configuration details.

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid
*   Kraken
*   OKX
*   MyOKX
*   ...and many others.

### Supported Futures Exchanges (Experimental)

*   Binance
*   Gate.io
*   Hyperliquid
*   OKX
*   Bybit

### Community-Tested Exchanges

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io), covering installation, configuration, strategy development, and advanced features.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).

For other installation methods, refer to the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

(See the original README for bot commands)

### Telegram RPC Commands

(See the original README for Telegram command list)

## Development Branches

*   `develop`:  Branch with new features and potential breaking changes.
*   `stable`:  Latest stable release.

## Support

*   **Discord:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   **Issues:** Report bugs and suggest features on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Pull Requests:** Contribute to the project by submitting [pull requests](https://github.com/freqtrade/freqtrade/pulls). Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Requirements

*   **Up-to-date Clock:** Synchronize your clock to an NTP server.
*   **Minimum Hardware:**
    *   2GB RAM (advised)
    *   1GB disk space
    *   2 vCPU
*   **Software:**
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)