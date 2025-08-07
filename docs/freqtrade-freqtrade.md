# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, free, and open-source crypto trading bot designed to automate your trading strategies across various cryptocurrency exchanges.  [Explore the Freqtrade project on GitHub](https://github.com/freqtrade/freqtrade).

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

**Key Features:**

*   **Automated Trading:**  Execute trading strategies 24/7 on supported exchanges.
*   **Backtesting:**  Test strategies using historical data to evaluate performance.
*   **Strategy Optimization:**  Leverage machine learning for hyperparameter tuning.
*   **Adaptive prediction modeling:** Build a smart strategy with FreqAI that self-trains to the market via adaptive machine learning methods. [Learn more](https://www.freqtrade.io/en/stable/freqai/)
*   **WebUI & Telegram Integration:**  Monitor and manage your bot via a user-friendly web interface and Telegram commands.
*   **Extensive Exchange Support:**  Works with many popular exchanges like Binance, OKX, and Bybit, with more being added frequently (See list below).
*   **Dry-run Mode:**  Test strategies safely without risking real funds.
*   **Community Driven:** Benefit from a large and active community.

**Supported Exchanges:**

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
*   Bitvavo
*   Kucoin

*And potentially many others!*

**Important Disclaimers:**

*   **Educational Purposes Only:** This software is for educational purposes and should not be considered financial advice.
*   **Risk Management:** Always use dry-run mode initially and never risk money you cannot afford to lose.
*   **Coding Knowledge Recommended:** Familiarity with Python is recommended to understand and customize the bot effectively.

**Getting Started:**

*   **Quick Start:**  Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for a rapid setup.
*   **Installation:**  Refer to the detailed [Installation documentation](https://www.freqtrade.io/en/stable/installation/) for native installation methods.

**Documentation and Support:**

*   **Official Documentation:**  Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).
*   **Community Support:**  Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for discussions and help.
*   **Bug Reports & Feature Requests:**  Report bugs and request features on the [GitHub Issues page](https://github.com/freqtrade/freqtrade/issues).
*   **Contributing:**  Contribute to the project by submitting [pull requests](https://github.com/freqtrade/freqtrade/pulls).  See the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for more details.

**Requirements:**

*   **Up-to-date clock**: Synchronize with a NTP server.
*   **Minimum Hardware**: 2GB RAM, 1GB disk space, 2vCPU.
*   **Software**: Python >= 3.11, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended).