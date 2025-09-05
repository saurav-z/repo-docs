# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, free, and open-source crypto trading bot that empowers you to automate your trading strategies and navigate the volatile world of cryptocurrencies. [**Get started with Freqtrade on GitHub**](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

## Key Features

*   **Automated Trading:** Execute trading strategies 24/7 on supported exchanges.
*   **Backtesting:** Simulate strategies to evaluate performance with historical data.
*   **Strategy Optimization:** Use machine learning to fine-tune your trading parameters.
*   **Adaptive Prediction Modeling (FreqAI):** Build smarter strategies that learn from the market.
*   **Multi-Exchange Support:** Trade on various major crypto exchanges.
*   **WebUI & Telegram Integration:** Manage and monitor your bot via a user-friendly web interface and Telegram commands.
*   **Dry-Run Mode:** Test your strategies without risking real capital.

## Important Disclaimer

This software is for educational purposes only. Trade at your own risk. Always start in Dry-run mode and thoroughly understand the bot's functionality before risking real money. Familiarity with coding and Python is recommended.

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
*   OKX
*   OKX (EEA)
*   Bitvavo
*   Kucoin

*(See [exchange-specific notes](docs/exchanges.md) for potential special configurations.)*

**Experimental Futures Exchanges:**

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Read the [exchange-specific notes](docs/exchanges.md) and the [trading with leverage documentation](docs/leverage.md) before diving in.

## Getting Started

For a quick start, refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).
For other installation methods, please refer to the [installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage & Commands

### Bot Commands

Access a list of commands using `freqtrade -h`.

### Telegram RPC Commands

Manage your bot conveniently through Telegram with commands like:

*   `/start`: Start the trader.
*   `/stop`: Stop the trader.
*   `/status`: View open trades.
*   `/profit`: Check cumulative profit.
*   `/forceexit`: Instantly exit a trade.
*   `/balance`: View account balances.

[Learn more about Telegram usage](https://www.freqtrade.io/en/latest/telegram-usage/)

## Development

Freqtrade is an actively developed open-source project.
*   Use the `develop` branch for the latest features (may contain breaking changes).
*   Use the `stable` branch for the latest stable release.
*   Contribute with [Pull Requests](https://github.com/freqtrade/freqtrade/pulls) against the `develop` branch.

## Support and Community

*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and discussions.
*   **Issues:** Report bugs or issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Submit feature requests on the [issue tracker](https://github.com/freqtrade/freqtrade/issues/new/choose).
*   **Contributing:** Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for information about submitting pull requests.

## Requirements

*   **Up-to-date clock:** Essential for accurate trading; synchronize with an NTP server.
*   **Minimum Hardware:**
    *   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU
*   **Software:**
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)