# Freqtrade: The Open-Source Crypto Trading Bot

**Automate your crypto trading strategy with Freqtrade, a powerful, free, and open-source bot designed for both beginners and experienced traders!**  ([Back to the Freqtrade Repository](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a versatile crypto trading bot written in Python, offering a wide array of features to empower your trading journey.

## Key Features:

*   **Open Source & Free:** Benefit from a community-driven project with no cost.
*   **Multi-Exchange Support:** Trade on major exchanges like Binance, Bybit, and OKX, with experimental support for futures.
*   **Backtesting & Optimization:**  Test strategies with historical data and optimize them using machine learning.
*   **AI-Powered Strategies:**  Utilize FreqAI for adaptive prediction modeling, which learns and adapts to market conditions.
*   **Dry-Run Mode:** Practice your strategies without risking real capital.
*   **WebUI and Telegram Integration:** Manage your bot effortlessly through a web interface or Telegram commands.
*   **Customizable Strategies:**  Create and refine your own trading strategies.
*   **Detailed Reporting:**  Monitor your profit/loss in fiat currency and access performance reports.
*   **Community Support:** Get help, share ideas, and connect with other traders on Discord.

## Important Considerations

*   **Disclaimer:** This software is for educational purposes only. Use at your own risk.
*   **Knowledge is Key:**  Familiarize yourself with the bot's functionality and consider Python and coding knowledge.
*   **Start with Dry-Run:**  Always test your strategies in dry-run mode before using real funds.

## Supported Exchanges

Freqtrade supports numerous exchanges, with specific notes and configurations documented [here](docs/exchanges.md).

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX / MyOKX (OKX EEA)

### Supported Futures Exchanges (experimental)

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

Additionally, the community has confirmed successful trading on Bitvavo and Kucoin.

## Documentation

Comprehensive documentation is available on the [Freqtrade Website](https://www.freqtrade.io), providing detailed guides and information.

## Quick Start

Get started quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).
For other installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage & Commands
(See the original README for commands).

## Development Branches

*   `develop`:  For new features (may have breaking changes).
*   `stable`: The latest stable release.
*   `feat/*`: Feature branches (for testing specific features).

## Support & Community

*   **Discord:**  Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for support and discussions.
*   **Issues:**  Report bugs and problems on the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:**  Suggest improvements or new features [here](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   **Pull Requests:**  Contribute to the project through [pull requests](https://github.com/freqtrade/freqtrade/pulls).

## Requirements

*   **Accurate Clock:** Ensure your system clock is synchronized to a NTP server.
*   **Minimum Hardware:**
    *   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU
*   **Software:** Python 3.11+, pip, git, TA-Lib, virtualenv (recommended), Docker (recommended).