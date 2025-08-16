# Freqtrade: Your Open-Source Crypto Trading Bot

**Automate your cryptocurrency trading strategies with Freqtrade, a powerful and flexible open-source bot.**  [Visit the original repo](https://github.com/freqtrade/freqtrade)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a free, open-source crypto trading bot written in Python, designed for both beginners and experienced traders. It supports a wide array of exchanges and offers advanced features to help you automate and optimize your trading strategies.

**Key Features:**

*   **Extensive Exchange Support:** Compatible with major cryptocurrency exchanges, including Binance, Bybit, OKX, and more.
*   **Backtesting and Strategy Optimization:** Backtest your strategies with historical data and optimize them using machine learning techniques, including adaptive prediction modeling with FreqAI.
*   **Automated Trading:** Automate your buy and sell strategies.
*   **Dry-Run Mode:** Test your strategies without risking real funds.
*   **User-Friendly Interface:**  Built-in WebUI for managing your bot and Telegram integration for convenient control.
*   **Advanced Analysis Tools:** Includes tools for plotting, performance reporting, and more.
*   **Flexible Configuration:** Configure your bot with a whitelist and blacklist for crypto currencies.

### Disclaimer

*This software is for educational purposes only. Trade at your own risk.*

### Supported Exchanges

Freqtrade supports a variety of exchanges, with ongoing community testing and improvements.

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
*   And more (check exchange specific notes).

### Getting Started

1.  **Documentation:** Explore the [Freqtrade documentation](https://www.freqtrade.io) to understand the bot's functionality and configuration.
2.  **Installation:** Follow the [installation instructions](https://www.freqtrade.io/en/stable/installation/) to get the bot set up on your system.  The [Docker Quickstart](https://www.freqtrade.io/en/stable/docker_quickstart/) is recommended for a quick setup.
3.  **Configuration:** Configure your bot with your desired trading strategies, exchange settings, and other preferences.
4.  **Dry Run:** Start your bot in dry-run mode to test your strategies without real funds.
5.  **Live Trading:** Once you're comfortable, enable live trading to start automating your trades.

### Useful Commands

Here are a few of the most common commands to get you started.

*   `/start`: Starts the trader.
*   `/stop`: Stops the trader.
*   `/status <trade_id>|[table]`: Lists all or specific open trades.
*   `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
*   `/help`: Show help message.
*   `/version`: Show version.

### Development Branches

The project is structured into main branches:
*   `develop`
*   `stable`
*   `feat/*` (Feature branches)

### Contributing

Freqtrade is an open-source project, and contributions are welcome!  Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for guidelines. For general questions and support, join the [Freqtrade Discord server](https://discord.gg/p7nuUNVfP7).

### Support

*   **Discord:** Join the [Freqtrade Discord server](https://discord.gg/p7nuUNVfP7) for support and community discussions.
*   **Issues:** Report bugs or request features via the [issue tracker](https://github.com/freqtrade/freqtrade/issues).
*   **Pull Requests:** Contribute to the project by submitting [pull requests](https://github.com/freqtrade/freqtrade/pulls).

### Requirements

**Essential**

*   Up-to-date clock (synchronized with NTP).
*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)

**Minimum Hardware**

*   2GB RAM
*   1GB disk space
*   2vCPU