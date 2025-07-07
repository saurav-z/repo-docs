# Freqtrade: The Free & Open Source Crypto Trading Bot

Freqtrade is a powerful, open-source crypto trading bot written in Python, allowing you to automate your trading strategies across various cryptocurrency exchanges. ([Original Repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

## Key Features

*   **Automated Trading:** Executes trading strategies automatically on supported exchanges.
*   **Backtesting:** Test your strategies against historical market data.
*   **Strategy Optimization:** Optimize your strategies using machine learning.
*   **Machine Learning with FreqAI:** Adaptive prediction modeling to build smart, self-training strategies.
*   **Web UI:** Manage and monitor your bot through a built-in web interface.
*   **Telegram Integration:** Control and receive notifications via Telegram.
*   **Supported Exchanges:** Supports major exchanges. ([See list](#supported-exchange-marketplaces))
*   **Dry-run mode:** Run the bot without real money.

## Disclaimer

This software is for educational purposes only. Do not risk money which
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS
AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not
hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchange Marketplaces

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
*   [potentially many others](https://github.com/ccxt/ccxt/). _(We cannot guarantee they will work)_

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid
*   OKX
*   Bybit

Please make sure to read the [exchange specific notes](docs/exchanges.md), as well as the [trading with leverage](docs/leverage.md) documentation before diving in.

### Community tested

Exchanges confirmed working by the community:

*   Bitvavo
*   Kucoin

## Documentation

Complete documentation can be found on the [freqtrade website](https://www.freqtrade.io).

## Quickstart

Refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for a fast setup. For native installation, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot commands
For detailed usage commands, please refer to the original README.

### Telegram RPC commands

Telegram integration is a great way to control the bot. See the documentation for a full command list: [freqtrade telegram-usage](https://www.freqtrade.io/en/latest/telegram-usage/)

- `/start`: Starts the trader.
- `/stop`: Stops the trader.
- `/stopentry`: Stop entering new trades.
- `/status <trade_id>|[table]`: Lists all or specific open trades.
- `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
- `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
- `/fx <trade_id>|all`: Alias to `/forceexit`
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency.
- `/daily <n>`: Shows profit or loss per day, over the last n days.
- `/help`: Show help message.
- `/version`: Show version.

## Development Branches

*   `develop`: Contains new features, may have breaking changes.
*   `stable`: Contains the latest stable release.
*   `feat/*`: Feature branches.

## Support

*   **Discord:** Join the [Discord server](https://discord.gg/p7nuUNVfP7) for help and community interaction.
*   **Issues:** Report bugs or suggest features on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Pull Requests:** Contribute code improvements via [pull requests](https://github.com/freqtrade/freqtrade/pulls).  Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)

## Requirements

### Up-to-date clock
The clock must be accurate, synchronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Minimum hardware required
*   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)