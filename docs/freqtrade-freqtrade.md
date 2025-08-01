# Freqtrade: Your Open-Source Crypto Trading Bot for Automated Profits

Freqtrade is a powerful, open-source Python-based crypto trading bot, designed to automate your trading strategies and maximize your profits.  [Learn more about Freqtrade on Github](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

## Key Features

*   **Automated Trading:** Automate trading on various cryptocurrency exchanges.
*   **Backtesting:** Test and refine strategies using historical data.
*   **Machine Learning:** Optimize strategies with machine learning.
*   **WebUI & Telegram Control:** Manage your bot via a web interface or Telegram.
*   **Multiple Exchanges:** Supports major exchanges, with ongoing community testing and expansion.
*   **Dry-Run Mode:** Test strategies without risking real funds.
*   **Performance Analysis:** Gain valuable insights into your trading results.
*   **FreqAI:** Build smart strategies with FreqAI that self-trains to the market via adaptive machine learning methods. [Learn more](https://www.freqtrade.io/en/stable/freqai/)

## Disclaimer

This software is for educational purposes only. Trade with caution. Always backtest and use dry-run modes before using real money.  The authors and contributors are not responsible for your trading outcomes.

## Supported Exchanges

Freqtrade supports a wide range of cryptocurrency exchanges.  Please see [exchange specific notes](docs/exchanges.md) for any special configurations.

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX / MyOKX (OKX EEA)

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

### Community tested

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io)

## Quick Start

Get started quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For other installation methods, refer to the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Development Branches

*   `develop`:  The branch for new features (may contain breaking changes).
*   `stable`:  The latest stable release branch.
*   `feat/*`: Feature branches (use with caution).

## Support

### Help / Discord

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.

### Issues

Report bugs and issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).

### Feature Requests

Suggest new features via the [feature request](https://github.com/freqtrade/freqtrade/labels/enhancement) section.

### Pull Requests

Contribute to Freqtrade by submitting pull requests.  Refer to the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for details.

## Requirements

### Up-to-date clock

*   An accurate and synchronized clock is essential for reliable exchange communication.

### Minimum hardware

*   2GB RAM
*   1GB disk space
*   2 vCPU

### Software

*   Python 3.11+
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)