# Freqtrade: The Open-Source Crypto Trading Bot You've Been Waiting For

Freqtrade is a powerful, free, and open-source crypto trading bot designed to automate your trading strategies across various cryptocurrency exchanges.  [Check out the original repo here](https://github.com/freqtrade/freqtrade)

## Key Features

*   **Python-Based:** Built on Python 3.11+ for compatibility across Windows, macOS, and Linux.
*   **Exchange Support:** Supports a wide array of major cryptocurrency exchanges, including Binance, Kraken, and OKX. (See [exchange specific notes](docs/exchanges.md) for details)
*   **Backtesting & Optimization:** Includes robust backtesting tools and machine learning-based strategy optimization.
*   **FreqAI Adaptive Prediction:**  Leverage self-training adaptive machine learning. [Learn more](https://www.freqtrade.io/en/stable/freqai/)
*   **WebUI & Telegram Control:** Monitor and manage your bot through a built-in web interface and Telegram integration.
*   **Dry-Run Mode:** Test your strategies without risking real capital.
*   **Automated Data Handling:**  Download, convert, and manage historical market data for analysis and backtesting.
*   **Performance Reporting:**  Track your profit/loss in fiat currency and generate performance reports.

## Disclaimer

_**Important:  This software is for educational purposes only.**_  Always use caution and start with Dry-run mode to understand the bot before using real funds. The developers are not responsible for your trading results.  It is recommended that you have basic knowledge of Python.

## Supported Exchanges

Freqtrade supports the following exchanges:

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
*   And potentially many others through CCXT.

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

See the [exchange specific notes](docs/exchanges.md), and [trading with leverage](docs/leverage.md) documentation.

### Community Tested

Exchanges confirmed working by the community:

*   Bitvavo
*   Kucoin

## Documentation

Comprehensive documentation is available on the [freqtrade website](https://www.freqtrade.io) to help you understand how the bot works.

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/). For native installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot Commands

(See `freqtrade --help` for details, or see the documentation)

### Telegram RPC Commands

Easily manage your bot via Telegram with commands like:

*   `/start`, `/stop`, `/status`, `/profit`, `/forceexit`, `/performance`, `/balance`, `/daily`, `/help`, `/version`

## Development Branches

*   `develop`: Active development branch (may contain breaking changes).
*   `stable`: Latest stable release branch.
*   `feat/*`: Feature branches (for testing specific features).

## Support & Community

*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and discussions.
*   **Issues:** Report bugs and suggest improvements on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Submit enhancement requests on the [feature requests](https://github.com/freqtrade/freqtrade/labels/enhancement) page.
*   **Pull Requests:** Contribute to the project by submitting pull requests; see the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).

## Requirements

*   **Up-to-date Clock:** Accurate time synchronization is crucial (NTP recommended).
*   **Hardware:**
    *   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU
*   **Software:**
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)