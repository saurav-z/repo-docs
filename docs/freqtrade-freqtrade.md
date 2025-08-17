# Freqtrade: Your Open-Source Crypto Trading Bot

**Automate your crypto trading strategies and take control of your investments with Freqtrade - a powerful, free, and open-source trading bot!** ([Original Repo](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a versatile crypto trading bot written in Python, designed for both beginners and experienced traders. It supports a wide range of major exchanges and offers features for backtesting, strategy optimization, and flexible control.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features

*   **Open Source & Free:** Utilize a community-driven tool with no licensing fees.
*   **Multi-Exchange Support:** Trade on leading crypto exchanges, including Binance, Kraken, and OKX.
*   **Automated Trading:** Execute trading strategies 24/7, even while you sleep.
*   **Backtesting:** Simulate trading strategies using historical data to evaluate performance.
*   **Strategy Optimization:**  Refine your trading strategies with machine learning for optimal results.
*   **FreqAI Integration:** Build smart strategies that adapt to market changes with adaptive machine learning methods.
*   **WebUI & Telegram Control:** Manage your bot via a built-in web interface or through Telegram commands.
*   **Dry-Run Mode:** Test strategies without risking real capital.

## Supported Exchanges

Freqtrade supports a growing list of exchanges, including:

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

**Note:** Please consult the [exchange specific notes](docs/exchanges.md) for detailed configuration instructions for each exchange.

## Getting Started

*   **Documentation:** Access comprehensive documentation on the [Freqtrade website](https://www.freqtrade.io) to understand the bot's functionality.
*   **Quickstart:**  Get up and running quickly with our [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).
*   **Installation:** Find detailed installation instructions on the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Disclaimer

*   This software is for educational purposes only.
*   Do not risk money you cannot afford to lose.
*   Use the software at your own risk. The authors and affiliates are not responsible for your trading outcomes.
*   We strongly recommend having coding and Python knowledge to understand the bot's mechanism.
*   Always start with Dry-run before using real funds.

## Community & Support

*   **Discord:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support, discussions, and community engagement.
*   **Issues:**  Report bugs and issues on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:**  Suggest new features on the [feature request section](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   **Pull Requests:** Contribute to the project by submitting pull requests. Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.

## Requirements

*   **Clock Accuracy:**  Ensure your system clock is synchronized with an NTP server.
*   **Minimum Hardware:**
    *   2GB RAM (advised)
    *   1GB disk space
    *   2 vCPU
*   **Software:**
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)