# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, open-source crypto trading bot designed to automate your trading strategies. ([Original Repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade empowers both novice and experienced traders with robust tools for automated cryptocurrency trading. It supports major exchanges and offers a user-friendly interface, backtesting, and strategy optimization.

## Key Features

*   **Automated Trading:** Execute trades 24/7 based on your defined strategies.
*   **Multi-Exchange Support:** Compatible with leading crypto exchanges, including Binance, Bybit, OKX and many more.
*   **Backtesting & Optimization:** Test strategies with historical data and optimize parameters using machine learning.
*   **WebUI & Telegram Integration:** Monitor and manage your bot through a built-in web interface and Telegram commands.
*   **Dry-Run Mode:** Simulate trades without real money to test and refine your strategies.
*   **FreqAI:** Build smart strategies with FreqAI that self-trains to the market via adaptive machine learning methods.
*   **Python-Based:** Written in Python, offering flexibility and extensibility.

## Getting Started

### Quick Start

Get up and running quickly using Docker with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).

### Installation

For native installation methods, refer to the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Supported Exchanges

Freqtrade supports a wide range of exchanges, with more being added regularly.  Please consult the [exchange specific notes](docs/exchanges.md) for specific configuration requirements.

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
*   And potentially many others via [ccxt](https://github.com/ccxt/ccxt/).

**Supported Futures Exchanges (experimental):**

*   [Binance](https://www.binance.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [OKX](https://okx.com/)
*   [Bybit](https://bybit.com/)

**Community Tested Exchanges:**

*   [Bitvavo](https://bitvavo.com/)
*   [Kucoin](https://www.kucoin.com/)

## Documentation

Comprehensive documentation is available on the [freqtrade website](https://www.freqtrade.io) to guide you through setup, usage, and advanced features.

## Basic Usage

*   **Bot Commands:** Use the command line interface to manage your bot.
*   **Telegram RPC Commands:** Control your bot via Telegram with commands like `/start`, `/stop`, `/status`, and many more. (See the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for a full list).

## Development

Freqtrade is an open-source project and welcomes contributions from the community.

*   **Development Branches:**  `develop` (latest features), `stable` (stable releases), and `feat/*` (feature branches).  Always create pull requests against the `develop` branch.
*   **Contributing:**  Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.
*   **Good First Issues:**  Explore issues tagged as [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) to get involved.

## Support & Community

*   **Discord:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   **Issues:** Report bugs and request features on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:**  Suggest new features via the [enhancement](https://github.com/freqtrade/freqtrade/labels/enhancement) label.

## Requirements

*   **Up-to-Date Clock:** Accurate time synchronization is crucial for exchange communication.
*   **Minimum Hardware:** 2GB RAM, 1GB disk space, 2vCPU (recommended).
*   **Software:** Python 3.11+, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended).

## Disclaimer

*   This software is for educational purposes only.
*   Use at your own risk.
*   Start in Dry-run mode.
*   Knowledge of Python and trading concepts is recommended.