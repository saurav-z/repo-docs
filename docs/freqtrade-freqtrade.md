# Freqtrade: The Free and Open-Source Crypto Trading Bot

Freqtrade is a powerful, open-source crypto trading bot that empowers you to automate your trading strategies.  [Explore the Freqtrade project on GitHub](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

## Key Features

*   **Automated Trading:** Execute your crypto trading strategies automatically, 24/7.
*   **Multi-Exchange Support:** Compatible with major crypto exchanges, with a growing list of supported platforms.
*   **Backtesting & Optimization:** Test your strategies with historical data and optimize them using machine learning.
*   **Machine Learning:**  Use machine learning to optimize your buy/sell strategy parameters with real exchange data.
*   **FreqAI:** Build a smart strategy with FreqAI that self-trains to the market via adaptive machine learning methods.
*   **User-Friendly Interface:** Manage and monitor your bot through a built-in WebUI and Telegram integration.
*   **Dry-Run Mode:** Test your strategies without risking real funds.
*   **Open Source & Community Driven:** Benefit from a vibrant community and contribute to the project.
*   **Fiat Profit/Loss:** Display your profit/loss in fiat currency.
*   **Performance Reporting:** Review the performance of your current trades.

## Supported Exchanges

Freqtrade supports a wide range of cryptocurrency exchanges, including:

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX and MyOKX
*   Bitvavo (Community Tested)
*   Kucoin (Community Tested)

**Note:** Check the [exchange-specific notes](docs/exchanges.md) for configuration details.

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Getting Started

### Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).

### Quick Start

Refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for the fastest setup.  Alternatively, explore the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/) for other installation methods.

## Basic Usage

### Bot Commands

(See Original README for command details. Removed redundant listing.)

### Telegram RPC Commands

(See Original README for command details. Removed redundant listing.)

## Development Branches

*   `develop`:  Branch for new features and potential breaking changes.
*   `stable`:  Branch for the latest stable releases.
*   `feat/*`: Feature branches.

## Contributing

We welcome contributions!  Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for details on how to contribute.

## Support and Community

*   **Discord:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for help and community discussions.
*   **Issues:** Report bugs and suggest features on the [issue tracker](https://github.com/freqtrade/freqtrade/issues).
*   **Pull Requests:**  Submit your improvements via [pull requests](https://github.com/freqtrade/freqtrade/pulls).

## Requirements

### Up-to-date clock
The clock must be accurate, synchronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Minimum hardware required

To run this bot we recommend you a cloud instance with a minimum of:

*   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software requirements

*   [Python >= 3.11](http://docs.python-guide.org/en/latest/starting/installation/)
*   [pip](https://pip.pypa.io/en/stable/installing/)
*   [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
*   [TA-Lib](https://ta-lib.github.io/ta-lib-python/)
*   [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
*   [Docker](https://www.docker.com/products/docker) (Recommended)

## Disclaimer

This software is for educational purposes only. Do not risk money which
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS
AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not
hesitate to read the source code and understand the mechanism of this bot.