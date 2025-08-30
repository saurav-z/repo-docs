# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, open-source crypto trading bot that empowers you to automate your trading strategies and optimize your crypto portfolio.  [View the original repository](https://github.com/freqtrade/freqtrade)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

## Key Features

*   **Automated Trading:** Automate your crypto trading strategies 24/7.
*   **Extensive Exchange Support:** Supports a wide range of major crypto exchanges.
*   **Backtesting:** Test your strategies with historical data to optimize performance.
*   **Machine Learning Optimization:** Leverage machine learning for strategy optimization.
*   **Telegram & WebUI Control:** Manage your bot conveniently through Telegram and a built-in WebUI.
*   **Dry-Run Mode:** Safely test your strategies without risking real funds.
*   **Adaptive Prediction Modeling (FreqAI):** Build smart strategies that self-train to market changes.
*   **Open-Source & Free:** Benefit from a community-driven project with no associated costs.
*   **Fiat Profit/Loss Display:** Track profits and losses in your preferred fiat currency.

## Supported Exchanges

Freqtrade supports a wide range of exchanges, including:

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX / MyOKX
*   ... and potentially many others via CCXT integration.

**Experimental Futures Exchanges:**

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

**Community Tested:**

*   Bitvavo
*   Kucoin

Refer to the [exchange specific notes](docs/exchanges.md) for any special configurations.

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).

For native installation, consult the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

**(Refer to original README for commands)**

## Contributing

Freqtrade thrives on community contributions.

*   **Report Bugs/Issues:** Help improve the bot by reporting bugs through the [issue tracker](https://github.com/freqtrade/freqtrade/issues).
*   **Request Features:** Suggest new features via the [feature request](https://github.com/freqtrade/freqtrade/labels/enhancement) to enhance the bot.
*   **Submit Pull Requests:** Contribute code, documentation, or improvements via [pull requests](https://github.com/freqtrade/freqtrade/pulls).  Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Support

*   **Discord:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for community support and discussions.

## Requirements

*   **Up-to-date Clock:** Accurate time synchronization is crucial for exchange communication.
*   **Minimum Hardware:** 2GB RAM, 1GB disk space, 2vCPU (Recommended).
*   **Software Requirements:** Python >= 3.11, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended).

## Disclaimer

*   This software is for educational purposes only. Use at your own risk.
*   Start with Dry-run mode before using real money.
*   Coding and Python knowledge are recommended.