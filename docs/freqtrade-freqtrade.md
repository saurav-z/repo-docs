# Freqtrade: Your Free & Open-Source Crypto Trading Bot

Freqtrade empowers you to automate your crypto trading strategies with a powerful, open-source bot.  [Visit the original repository](https://github.com/freqtrade/freqtrade) for more information and to get started.

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a free and open-source crypto trading bot written in Python, offering a comprehensive suite of features for both novice and experienced traders.

[Image of Freqtrade Screenshot]

## Key Features

*   **Automated Trading:** Execute strategies automatically on supported exchanges.
*   **Backtesting:**  Simulate your strategies using historical data.
*   **Strategy Optimization:** Utilize machine learning for parameter optimization.
*   **FreqAI Integration:** Build adaptive prediction models using adaptive machine learning methods.
*   **Supported Exchanges:** Binance, Bybit, Gate.io, HTX, Kraken, OKX, Kucoin, and many more. (See the [exchange specific notes](docs/exchanges.md) for details)
*   **WebUI & Telegram Control:** Manage and monitor your bot through a built-in web interface or Telegram.
*   **Dry-Run Mode:** Test your strategies without risking real capital.
*   **Profit/Loss Tracking:** Easily monitor your gains and losses in fiat currency.

## Disclaimer

This software is for educational purposes only. Trade at your own risk.

## Supported Exchanges

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
*   [Bitvavo](https://bitvavo.com/)
*   [Kucoin](https://www.kucoin.com/)

**Supported Futures Exchanges (experimental)**

*   [Binance](https://www.binance.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [OKX](https://okx.com/)
*   [Bybit](https://bybit.com/)

Please read the [exchange specific notes](docs/exchanges.md), as well as the [trading with leverage](docs/leverage.md) documentation before diving in.

## Documentation

Consult the comprehensive [freqtrade website](https://www.freqtrade.io) for detailed documentation and guides.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For native installation, refer to the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands (CLI)
See the below for bot commands.

### Telegram Commands
Control your bot easily with Telegram. See documentation for full command list.

*   `/start`: Starts the trader.
*   `/stop`: Stops the trader.
*   `/status`: Lists all or specific open trades.
*   `/profit`: Lists cumulative profit from all finished trades.
*   `/balance`: Show account balance per currency.

## Development Branches

*   `develop`:  For the latest features (may contain breaking changes).
*   `stable`:  The latest stable, tested release.
*   `feat/*`: Feature branches, for active development.

## Support

### Discord
Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for community support.

### Issues
Report bugs and contribute to the project by following the process outlined in the guidelines.  Please [search the issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue) first.

### Feature Requests
Share your ideas and suggestions for new features. See [feature requests](https://github.com/freqtrade/freqtrade/labels/enhancement).

### Pull Requests
We welcome contributions! Please read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) before submitting a pull request.

## Requirements

### Essential Software

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)

### Hardware Recommendations

*   Minimal (advised): 2GB RAM, 1GB disk space, 2vCPU