# Freqtrade: Your Open-Source Crypto Trading Bot

**Automate your crypto trading strategies with Freqtrade, a free and open-source bot with backtesting, optimization, and webUI support.** ([View on GitHub](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade is a powerful and versatile crypto trading bot that allows you to automate your trading strategies across various cryptocurrency exchanges. Written in Python, Freqtrade offers a comprehensive suite of tools for backtesting, optimization, and real-time trading management.

## Key Features

*   **Open Source & Free:** Freedom to use, modify, and distribute.
*   **Multi-Exchange Support:** Compatible with numerous major exchanges.
*   **Backtesting:** Test your strategies with historical data.
*   **Strategy Optimization:** Enhance performance with machine learning.
*   **FreqAI Support**: Build a smart strategy that self-trains to the market.
*   **WebUI & Telegram Integration:** Manage your bot through a web interface or Telegram.
*   **Dry-run Mode:** Safely test strategies without risking real funds.
*   **Profit/Loss Display:** Monitor your trading results in fiat currency.
*   **Performance Reporting:** Detailed insights into your trades.

## Supported Exchanges

Freqtrade supports a wide array of cryptocurrency exchanges. Please refer to the [exchange specific notes](docs/exchanges.md) for configurations.

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX & MyOKX (OKX EEA)

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

*Community Tested:* Bitvavo, Kucoin

## Getting Started

*   **Quickstart:** Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for a fast setup.
*   **Installation:** For native installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage & Commands

### Bot Commands

For more detailed usage instructions, see the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

*   `/start`: Starts the trader.
*   `/stop`: Stops the trader.
*   `/stopentry`: Stop entering new trades.
*   `/status <trade_id>|[table]`: Lists all or specific open trades.
*   `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
*   `/profit_long [<n>]`: Lists cumulative profit from all finished long trades, over the last n days.
*   `/profit_short [<n>]`: Lists cumulative profit from all finished short trades, over the last n days.
*   `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
*   `/fx <trade_id>|all`: Alias to `/forceexit`
*   `/performance`: Show performance of each finished trade grouped by pair
*   `/balance`: Show account balance per currency.
*   `/daily <n>`: Shows profit or loss per day, over the last n days.
*   `/help`: Show help message.
*   `/version`: Show version.

## Development Branches

*   `develop`: Active development branch with the latest features.
*   `stable`: Stable, well-tested releases.
*   `feat/*`: Feature branches for specific development tasks.

## Contributing

We welcome contributions! Please read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) to understand the requirements before sending your pull-requests.

*   **Issues:** [Report bugs](https://github.com/freqtrade/freqtrade/issues/new/choose) and [suggest features](https://github.com/freqtrade/freqtrade/issues/new/choose).
*   **Pull Requests:** Contribute code to improve the bot.
*   **Discord:** Join the [discord server](https://discord.gg/p7nuUNVfP7)

## Requirements

*   **Up-to-date clock:** Accurate NTP synchronization is essential.
*   **Hardware:** Minimum recommended: 2GB RAM, 1GB disk space, 2vCPU.
*   **Software:** Python 3.11+, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended).

## Disclaimer

*   This software is for educational purposes only.
*   Use at your own risk; the authors and affiliates are not responsible for trading results.
*   Start with Dry-run mode before using real funds.
*   Familiarity with coding and Python is recommended.