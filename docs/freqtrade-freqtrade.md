# Freqtrade: Open-Source Crypto Trading Bot

**Automate your crypto trading strategies with Freqtrade, a free and open-source bot designed for profitability and ease of use.** ([Original Repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade empowers you to trade cryptocurrencies automatically. It supports various exchanges, and offers robust features for backtesting, strategy optimization, and automated trading via Telegram or web UI.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Open Source & Free:** Benefit from a community-driven, open-source platform.
*   **Cross-Platform:** Runs on Windows, macOS, and Linux with Python 3.11+.
*   **Exchange Support:** Compatible with major exchanges including Binance, Bybit, OKX, and more.
*   **Backtesting & Optimization:** Backtest strategies with historical data and optimize them using machine learning.
*   **FreqAI Integration:** Build intelligent strategies with self-training, adaptive machine learning capabilities.
*   **Dry-Run Mode:** Test strategies without risking real money.
*   **Web UI & Telegram Control:** Manage and monitor your bot through a built-in web interface and Telegram integration.
*   **Risk Management:** Includes whitelisting/blacklisting of cryptocurrencies and performance reporting.

## Getting Started

Refer to the official [documentation](https://www.freqtrade.io) to understand how the bot works.

### Quickstart

Get started quickly using the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).

### Installation

For native installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Disclaimer

*   **Educational Purposes Only:** This software is for educational use only.
*   **Risk Awareness:** Use at your own risk and only with money you can afford to lose.
*   **Due Diligence:** Start in Dry-run mode and thoroughly understand the bot before trading with real funds.
*   **Knowledge:** Familiarity with coding and Python is recommended.

## Exchanges

Freqtrade supports numerous exchanges. Always refer to the [exchange specific notes](docs/exchanges.md) for special configurations.

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
*   ...and potentially others.

### Supported Futures Exchanges (Experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Community Tested Exchanges

*   Bitvavo
*   Kucoin

## Basic Usage

### Bot Commands

Basic commands are available in the README. More details are found on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

*   `/start` - Start trading.
*   `/stop` - Stop trading.
*   `/status` - List trades
*   `/profit` - Show profit.
*   `/help` - Show help message.
*   `/version` - Show version.
  ...and other commands to manage your trades.

## Development

The project follows two main branches:

*   `develop` - Latest features, but may have breaking changes.
*   `stable` - Latest stable release.

### Contributing

Contributions are welcome! Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

*   **Good first issues** are available for beginners.
*   **Create an issue** before starting any major new feature.
*   **PRs** should be created against the `develop` branch.

### Support

*   **Discord:** Join the [Discord server](https://discord.gg/p7nuUNVfP7) for help and discussions.
*   **Issues:** Report bugs and request features on the [issue tracker](https://github.com/freqtrade/freqtrade/issues).

## Requirements

*   **Up-to-date clock:** Requires accurate time synchronization.
*   **Minimal hardware:** 2GB RAM, 1GB disk space, 2vCPU (advised).
*   **Software:** Python >= 3.11, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended).