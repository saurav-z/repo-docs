# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, open-source crypto trading bot that allows you to automate your trading strategies across multiple exchanges.  **Automate your crypto trading strategies with Freqtrade, a free and open-source bot with backtesting, machine learning, and more!**  [Get started with Freqtrade](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade empowers you to take control of your crypto trading with a flexible and customizable bot, offering features like backtesting, machine learning, and a user-friendly interface.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Supports Major Exchanges:** Trade on a wide range of popular crypto exchanges.
*   **Backtesting:**  Test your strategies with historical data to evaluate performance.
*   **Strategy Optimization:** Enhance your trading strategies using machine learning techniques.
*   **FreqAI Integration:** Leverage adaptive prediction modeling for smart strategy development.
*   **WebUI and Telegram Control:** Manage your bot through a web interface or Telegram commands.
*   **Dry-Run Mode:** Safely test your strategies without risking real funds.
*   **Fiat Profit/Loss Display:** Monitor your profits and losses in your local currency.
*   **Performance Reporting:** Get a clear view of your trades with detailed performance metrics.

## Supported Exchanges

Freqtrade supports numerous exchanges.  See the [exchange specific notes](docs/exchanges.md) for setup details.

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
*   ...and many more via CCXT integration.

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

## Documentation & Quick Start

Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).  For a quick start, explore the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  Detailed installation instructions are provided on the [installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Disclaimer

This software is for educational purposes only. Use at your own risk.  The authors and affiliates are not responsible for your trading results.  Always start in Dry-run mode and understand the bot's functionality before trading with real money.

## Development and Community

### Get Help

Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and to connect with other users.

### Contribute

We welcome contributions!  Please review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) and open a new issue or pull request.

### Contact

*   [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)
*   [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)
*   [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

## Requirements

*   **Python >= 3.11**
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)
*   Up-to-date clock synchronized to a NTP server.