# Freqtrade: Your Open-Source Crypto Trading Bot 🚀

**Automate your crypto trading strategies with Freqtrade, a powerful, free, and open-source Python trading bot.** Learn more and get started with Freqtrade on the [original repository](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a versatile crypto trading bot written in Python, designed to empower traders of all levels. It supports a wide range of exchanges and offers a comprehensive suite of features for automated trading.

## Key Features:

*   **Open Source & Free:** Leverage the power of a community-driven, open-source project.
*   **Multi-Exchange Support:** Trade on major exchanges like Binance, Bybit, and Kraken, with support for numerous others.
*   **Backtesting & Strategy Optimization:** Test your strategies and optimize them using machine learning.
*   **Dry-Run Mode:** Simulate trades without risking real capital.
*   **Telegram & WebUI Integration:** Control and monitor your bot through Telegram and a built-in web interface.
*   **Adaptive Prediction Modeling (FreqAI):** Build a smart strategy with FreqAI that self-trains to the market via adaptive machine learning methods.
*   **Python-Based:** Built on Python 3.11+ for cross-platform compatibility.
*   **Built-in WebUI:** Built-in web UI to manage your bot.
*   **Manageable via Telegram:** Manage the bot with Telegram.
*   **Performance status report:** Provide a performance status of your current trades.

## Disclaimer

***Trading cryptocurrencies involves substantial risk, and you could lose money.*** The information provided here is for educational purposes only. Always start with dry-run mode, and only trade with funds you can afford to lose. We recommend having Python knowledge and understanding the bot's mechanisms. The authors and contributors are not responsible for your trading outcomes.

## Supported Exchanges

Freqtrade supports a wide array of exchanges. Please refer to the [exchange-specific notes](docs/exchanges.md) for any special configurations.

*   [X] [Binance](https://www.binance.com/)
*   [X] [Bitmart](https://bitmart.com/)
*   [X] [BingX](https://bingx.com/invite/0EM9RX)
*   [X] [Bybit](https://bybit.com/)
*   [X] [Gate.io](https://www.gate.io/ref/6266643)
*   [X] [HTX](https://www.htx.com/)
*   [X] [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [X] [Kraken](https://kraken.com/)
*   [X] [OKX](https://okx.com/)
*   [X] [MyOKX](https://okx.com/) (OKX EEA)
*   [ ] [potentially many others](https://github.com/ccxt/ccxt/). _(We cannot guarantee they will work)_

### Supported Futures Exchanges (experimental)

*   [X] [Binance](https://www.binance.com/)
*   [X] [Gate.io](https://www.gate.io/ref/6266643)
*   [X] [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [X] [OKX](https://okx.com/)
*   [X] [Bybit](https://bybit.com/)

## Community Tested Exchanges

*   [X] [Bitvavo](https://bitvavo.com/)
*   [X] [Kucoin](https://www.kucoin.com/)

## Documentation

Explore the comprehensive documentation for detailed information on installation, configuration, and usage: [Freqtrade Documentation](https://www.freqtrade.io).

## Quick Start

Get up and running quickly with Freqtrade using the [Docker Quickstart](https://www.freqtrade.io/en/stable/docker_quickstart/). For native installation methods, see the [Installation documentation](https://www.freqtrade.io/en/stable/installation/).

## Development branches

*   `develop` - This branch has often new features, but might also contain breaking changes. We try hard to keep this branch as stable as possible.
*   `stable` - This branch contains the latest stable release. This branch is generally well tested.
*   `feat/*` - These are feature branches, which are being worked on heavily. Please don't use these unless you want to test a specific feature.

## Support & Community

*   **Discord:** Join the Freqtrade community on [Discord](https://discord.gg/p7nuUNVfP7) for support and discussions.
*   **Issues:** Report bugs or issues via the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Suggest new features or improvements using the [feature request](https://github.com/freqtrade/freqtrade/labels/enhancement) label.
*   **Pull Requests:** Contribute to the project by submitting pull requests. Learn more in the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).

## Requirements

*   **Accurate Clock:** A synchronized clock with a NTP server is essential.
*   **Minimum Hardware:**
    *   Recommended: 2GB RAM, 1GB disk space, 2vCPU
*   **Software Requirements:**
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)