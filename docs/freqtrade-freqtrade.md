# Freqtrade: Your Free and Open-Source Crypto Trading Bot

**Automate your crypto trading strategies with Freqtrade, a powerful, free, and open-source trading bot for major cryptocurrency exchanges. Explore the possibilities at the original repository: [Freqtrade on GitHub](https://github.com/freqtrade/freqtrade)**

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade empowers you to develop and automate your crypto trading strategies, offering advanced features and comprehensive tools for effective trading.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Multi-Exchange Support:** Trade on a wide range of major crypto exchanges.
*   **Backtesting & Strategy Optimization:** Test your strategies with historical data and optimize them using machine learning.
*   **Automated Trading:**  Configure the bot to execute trades automatically based on your defined strategies.
*   **Built-in WebUI & Telegram Integration:** Manage and monitor your bot easily through a web interface or Telegram.
*   **Dry-Run Mode:** Test your strategies without risking real capital.
*   **Adaptive prediction modeling:** Build a smart strategy with FreqAI that self-trains to the market via adaptive machine learning methods. [Learn more](https://www.freqtrade.io/en/stable/freqai/)

## Supported Exchanges

Freqtrade supports numerous exchanges. For exchange-specific configurations, please refer to the [exchange specific notes](docs/exchanges.md).

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
*   ...and many more through CCXT integration.

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For native installation options, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

A list of basic bot commands and a description can be found in the original README.

### Telegram RPC commands

The original README provides information on telegram bot commands and their functionality.

## Development Branches

*   `develop`:  Features new features and potential breaking changes; Aiming for stability.
*   `stable`:  The latest stable release branch, well-tested.
*   `feat/*`: Feature branches for active development.

## Support

*   **Discord:** Join the [Freqtrade Discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   **Issues:** Report bugs and issues [here](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Submit feature requests [here](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   **Pull Requests:** Contribute to Freqtrade development by submitting pull requests [here](https://github.com/freqtrade/freqtrade/pulls).

## Requirements

*   Up-to-date and synchronized clock with a NTP server.
*   **Minimum hardware:** 2GB RAM, 1GB disk space, 2vCPU
*   **Software:** Python 3.11+, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended)