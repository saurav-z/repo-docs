# Freqtrade: Your Free and Open-Source Crypto Trading Bot

**Automate your crypto trading strategies with Freqtrade, a powerful, free, and open-source trading bot written in Python.** ([Original Repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade empowers you to automate your cryptocurrency trading strategies. Designed with flexibility in mind, it seamlessly integrates with major exchanges and offers robust features for backtesting, optimization, and real-time trading.  Manage your bot via Telegram or the built-in webUI.

## Key Features

*   **Multi-Exchange Support:**  Supports major exchanges including Binance, Bybit, OKX, Gate.io, Kraken, and more.
*   **Backtesting & Simulation:** Test your strategies with historical data using backtesting and dry-run modes.
*   **Strategy Optimization:** Leverage machine learning for hyperparameter optimization, to fine-tune your trading strategies.
*   **FreqAI Adaptive Prediction Modeling:** Create intelligent, self-training strategies that adapt to market changes.
*   **Built-in WebUI & Telegram Integration:** Monitor and control your bot through a user-friendly web interface or via Telegram commands.
*   **Profit/Loss Display:** Track your performance with profit/loss displayed in fiat currency.
*   **Performance Reporting:**  Gain insights with comprehensive performance reports of your current trades.
*   **Python 3.11+:**  Compatible with Windows, macOS, and Linux.

## Disclaimer

This software is for educational purposes only. Trade at your own risk. The developers and contributors assume no responsibility for your trading outcomes. Always begin with dry-run mode before trading with real funds.

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
*   And potentially many others via [CCXT](https://github.com/ccxt/ccxt/).

## Supported Futures Exchanges (experimental)

*   [Binance](https://www.binance.com/)
*   [Gate.io](https://www.gate.io/ref/6266643)
*   [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
*   [OKX](https://okx.com/)
*   [Bybit](https://bybit.com/)

Please read the [exchange specific notes](docs/exchanges.md) and the [trading with leverage](docs/leverage.md) documentation.

## Documentation

Comprehensive documentation is available on the [freqtrade website](https://www.freqtrade.io), providing details on installation, configuration, and advanced features.

## Quick Start

Get started quickly using the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/). For other installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

A quick overview of the bot's command-line interface is shown. For more detailed command information, please consult the documentation.

### Telegram RPC Commands

Manage your bot remotely with Telegram commands.  More details on [documentation](https://www.freqtrade.io/en/latest/telegram-usage/). Some example commands are:

*   `/start`: Starts the trader.
*   `/stop`: Stops the trader.
*   `/status <trade_id>|[table]`: Lists all or specific open trades.
*   `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
*   `/help`: Show help message.

## Development Branches

*   `develop`: The branch with new features, and potentially breaking changes.
*   `stable`: Contains the latest stable release.
*   `feat/*`: Feature branches.

## Support

*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   **Issues:**  Report bugs or request features via the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Pull Requests:** Contribute to the project by submitting pull requests. Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Requirements

### Clock

Ensure an accurate clock synchronized with an NTP server.

### Minimum Hardware

*   2GB RAM, 1GB disk space, 2vCPU

### Software

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)