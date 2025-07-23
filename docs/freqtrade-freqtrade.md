# Freqtrade: Automate Your Crypto Trading with Python

Freqtrade is a free and open-source crypto trading bot built in Python, designed to help you automate your trading strategies and maximize your profits.  Check out the original repository [here](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

## Key Features

*   **Cross-Platform Compatibility:** Works on Windows, macOS, and Linux.
*   **Exchange Support:**  Integrates with major cryptocurrency exchanges like Binance, Bybit, OKX, and more (see [exchange specific notes](docs/exchanges.md) for details).
*   **Backtesting:** Test your trading strategies using historical data.
*   **Dry-Run Mode:** Simulate trades without risking real funds.
*   **Strategy Optimization:** Enhance strategies using machine learning.
*   **FreqAI Integration:** Utilize adaptive machine learning for smart strategy development.
*   **Customization:**  Whitelist/blacklist cryptocurrencies for targeted trading.
*   **User Interface:** Built-in web UI for convenient bot management.
*   **Telegram Integration:** Control and monitor your bot via Telegram commands.
*   **Fiat Currency Display:**  Track your profit/loss in your preferred fiat currency.
*   **Performance Reporting:**  Generate performance reports to analyze your trading results.

## Important Disclaimers

*   **Educational Purposes Only:** This software is for educational and experimental purposes. Use at your own risk.
*   **Risk Management:**  Start with dry-run mode and thoroughly understand the bot's functionality before using real funds.
*   **Coding Knowledge Recommended:** Basic understanding of Python is beneficial for customizing and troubleshooting.

## Supported Exchanges

Freqtrade supports a wide array of cryptocurrency exchanges. Please consult the [exchange specific notes](docs/exchanges.md) for configurations.

*   [Binance](https://www.binance.com/)
*   [Bitmart](https://bitmart.com/)
*   [BingX](https://bingx.com/invite/0EM9RX)
*   [Bybit](https://bybit.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [HTX](https://www.htx.com/)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [Kraken](https://kraken.com/)
*   [OKX](https://okx.com/)
*   [MyOKX](https://okx.com/) (OKX EEA)
*   [and many others](https://github.com/ccxt/ccxt/)

### Supported Futures Exchanges (experimental)

*   [Binance](https://www.binance.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [OKX](https://okx.com/)
*   [Bybit](https://bybit.com/)

## Community Tested Exchanges

*   [Bitvavo](https://bitvavo.com/)
*   [Kucoin](https://www.kucoin.com/)

## Getting Started

*   **Quick Start:** Refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for an easy setup.
*   **Installation:**  Detailed installation instructions can be found in the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Documentation and Support

*   **Documentation:** Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).
*   **Community:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for discussions and support.
*   **Issues:** Report bugs or issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:**  Submit new feature ideas using the [feature request](https://github.com/freqtrade/freqtrade/labels/enhancement) label.
*   **Contributions:**  Contribute to the project by submitting pull requests.  Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Requirements

### System

*   **Up-to-date clock:** Requires an accurate time using an NTP server

### Minimum Hardware

*   **Recommended:** 2GB RAM, 1GB disk space, 2vCPU

### Software

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)