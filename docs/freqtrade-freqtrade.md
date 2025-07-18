# Freqtrade: Open-Source Crypto Trading Bot

**Automate your crypto trading strategies with Freqtrade, the free and open-source bot designed for all major exchanges.**  ([Visit the original repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade empowers you to create and backtest automated trading strategies across various cryptocurrency exchanges.  It provides a comprehensive suite of tools for backtesting, optimization, and managing your trading activities.

## Key Features

*   **Open Source & Free:** Leverage a community-driven, open-source platform.
*   **Exchange Support:** Compatible with a wide range of major exchanges (Binance, Bybit, OKX, Gate.io, Kraken, and more).
*   **Backtesting:** Simulate your strategies with historical data to assess performance.
*   **Strategy Optimization:** Utilize machine learning to optimize your strategy parameters.
*   **FreqAI Integration:** Build smart strategies using adaptive machine learning.
*   **Web UI & Telegram Control:** Manage your bot through a built-in web interface or Telegram commands.
*   **Dry-Run Mode:** Test your strategies without risking real capital.
*   **Profit/Loss Reporting:** Track your profits in your preferred fiat currency.
*   **Data Analysis & Plotting:** Visualize market data, backtesting results, and performance reports.

## Disclaimer

This software is for educational purposes only. Use it at your own risk. Always start in dry-run mode before using real funds. The authors and contributors are not responsible for your trading outcomes.

## Supported Exchanges

A wide range of exchanges are supported, including:

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX / MyOKX (OKX EEA)

See [exchange specific notes](docs/exchanges.md) for specific exchange requirements.

## Getting Started

Refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) or the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/) for detailed instructions on how to get started.

## Support & Community

*   **Documentation:**  Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).
*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   **Issues:** Report bugs or suggest features via the [issue tracker](https://github.com/freqtrade/freqtrade/issues).
*   **Contributing:**  Contribute to the project by submitting [pull requests](https://github.com/freqtrade/freqtrade/pulls).  See the [CONTRIBUTING document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for more information.

## Requirements

*   **Python:**  >= 3.11
*   **Hardware:** Minimum 2GB RAM, 1GB disk space, 2 vCPU (recommended)
*   **Dependencies:** `pip`, `git`, `TA-Lib`, `virtualenv` (Recommended), `Docker` (Recommended)
*   **Time Synchronization:** An accurate and synchronized system clock is essential.