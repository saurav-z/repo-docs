# Freqtrade: Your Open-Source Crypto Trading Bot

**Freqtrade is a powerful and free open-source crypto trading bot designed to automate your trading strategies across various exchanges.**  Visit the original repository for more information: [Freqtrade GitHub](https://github.com/freqtrade/freqtrade)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

Freqtrade empowers you to develop and execute automated trading strategies on cryptocurrency exchanges.  With features like backtesting, machine learning-based strategy optimization, and a user-friendly interface, you can refine your trading approach and potentially increase your profitability.

## Key Features

*   **Automated Trading:** Execute trades automatically based on your defined strategies.
*   **Backtesting:** Test your strategies using historical market data.
*   **Strategy Optimization:** Leverage machine learning to optimize your trading parameters.
*   **FreqAI Adaptive Prediction Modeling**: Build a smart strategy that self-trains to the market via adaptive machine learning methods. [Learn more](https://www.freqtrade.io/en/stable/freqai/)
*   **Exchange Support:** Supports many major exchanges (Binance, Bybit, Kraken, OKX, and more).
*   **WebUI and Telegram Integration:** Control and monitor your bot via a web interface or Telegram commands.
*   **Dry-Run Mode:** Simulate trades without risking real money.
*   **Profit Reporting:** Track profits in fiat currency.

## Supported Exchanges

Freqtrade supports a wide range of exchanges, including:

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX
*   and more (see the [exchange-specific notes](docs/exchanges.md))

## Getting Started

*   **Docker Quickstart:**  The easiest way to get started is with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).
*   **Installation:** For native installation, refer to the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Important Notes & Disclaimer

*   **Educational Purposes:** This software is for educational purposes only.
*   **Risk Management:** Do not risk money you cannot afford to lose. Always start with dry-run mode.
*   **Python Knowledge:**  Familiarity with Python and coding is recommended.
*   **Accuracy of clock:** The clock must be accurate, synchronized to a NTP server very frequently to avoid problems with communication to the exchanges.

## Documentation and Community

*   **Documentation:** Comprehensive documentation is available on the [Freqtrade website](https://www.freqtrade.io).
*   **Discord:**  Join the [Freqtrade Discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.

## Contribution

Freqtrade is an open-source project, and contributions are welcome!  Read the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) to learn how to contribute.

## Requirements

*   **Python:** >= 3.11
*   **Other requirements:** pip, git, TA-Lib, and virtualenv (recommended)
*   **Hardware:** Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU