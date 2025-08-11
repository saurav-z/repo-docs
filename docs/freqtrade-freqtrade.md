# Freqtrade: Your Open-Source Crypto Trading Bot

**Automate your crypto trading strategies and optimize your profits with Freqtrade, a powerful and versatile open-source trading bot.**  [Explore the Freqtrade Repo](https://github.com/freqtrade/freqtrade)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade empowers you to automate your trading strategies across various cryptocurrency exchanges. It's a free, open-source solution written in Python, providing extensive tools for backtesting, optimization, and real-time trading.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

**Key Features:**

*   **Open Source & Free:** Leverage a community-driven project without licensing fees.
*   **Cross-Platform:** Compatible with Windows, macOS, and Linux, built on Python 3.11+.
*   **Exchange Support:**  Integrates with major crypto exchanges like Binance, Kraken, and OKX, and is expanding, with experimental support for futures exchanges. See the supported exchanges [here](docs/exchanges.md).
*   **Backtesting & Optimization:**  Test strategies and optimize them using machine learning.
*   **Machine Learning Integration:** Benefit from adaptive prediction modeling using FreqAI.
*   **Dry-Run Mode:** Simulate trades without risking real capital.
*   **WebUI & Telegram Control:** Manage your bot effortlessly through a built-in WebUI and Telegram integration.
*   **Fiat Profit Display:** Easily track your profit/loss in your preferred fiat currency.
*   **Comprehensive Reporting:** Generate performance reports to monitor trade performance and trends.

## Disclaimer

This software is for educational purposes only.  **Trade responsibly. Only risk capital you can afford to lose.  The authors and affiliates assume no responsibility for your trading results.**

## Getting Started

*   **Quick Start:**  Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for a fast setup.
*   **Installation:** Explore native installation methods in the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).
*   **Documentation:** Familiarize yourself with the bot's operation via the [freqtrade website](https://www.freqtrade.io).

## Important Considerations:

*   **Accurate Clock:** Ensure your system clock is synchronized to a reliable NTP server to avoid exchange communication issues.
*   **Hardware Requirements:** We recommend a cloud instance with a minimum of 2GB RAM, 1GB disk space, and 2vCPU for optimal performance.
*   **Requirements:** Install Python >= 3.11, pip, git, TA-Lib, and virtualenv (recommended). Docker is also recommended.

## Community & Support

*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for discussions, support, and community engagement.
*   **Issues:** Report bugs and contribute via the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:**  Suggest improvements using the [feature request](https://github.com/freqtrade/freqtrade/labels/enhancement) label.
*   **Contributions:**  We welcome [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)! Review the [CONTRIBUTING document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines.