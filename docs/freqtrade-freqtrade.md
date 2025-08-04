# Freqtrade: Your Free & Open Source Crypto Trading Bot

**Take control of your crypto trading with Freqtrade, the advanced bot offering backtesting, strategy optimization, and automated trading capabilities. Check out the original repo [here](https://github.com/freqtrade/freqtrade).**

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Freqtrade is a powerful, open-source crypto trading bot written in Python, designed to automate your trading strategies across major cryptocurrency exchanges. This versatile bot offers a comprehensive suite of features, including backtesting, strategy optimization, and real-time trading.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Key Features:

*   **Automated Trading:** Execute trades based on your defined strategies.
*   **Backtesting:** Test and refine your strategies using historical data.
*   **Strategy Optimization:**  Use machine learning to optimize trading parameters.
*   **Adaptive prediction modeling** Use FreqAI for self-training strategies via adaptive machine learning.
*   **Supported Exchanges:** Wide range of supported exchanges, including Binance, OKX, and many more (see list below).
*   **Telegram Integration:**  Manage your bot and monitor trades directly through Telegram.
*   **WebUI:** User-friendly web interface for easy management.
*   **Dry-run Mode:** Safely test strategies without risking real funds.
*   **Fiat Profit/Loss Display:** Track your profits and losses in your preferred fiat currency.

## Supported Exchanges

Freqtrade supports a wide variety of exchanges:

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX / MyOKX (OKX EEA)

### Experimental Futures Exchanges

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

### Community-Tested Exchanges

*   Bitvavo
*   Kucoin

## Documentation

Explore the complete documentation on the [Freqtrade website](https://www.freqtrade.io).

## Quick Start

Get up and running quickly with the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/).  For native installation, see the [Installation documentation](https://www.freqtrade.io/en/stable/installation/).

## Disclaimer

*This software is for educational purposes only. Trade at your own risk. The authors and affiliates are not responsible for your trading results.*  It's strongly recommended to have Python and coding knowledge, test in Dry-run mode first, and thoroughly understand the bot's mechanics.

## Support

*   **Discord:** Join the Freqtrade [Discord server](https://discord.gg/p7nuUNVfP7) for community support and discussions.
*   **Issues:** Report bugs and issues via the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Suggest new features via the [enhancement label](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   **Pull Requests:** Contribute to the project by submitting [pull requests](https://github.com/freqtrade/freqtrade/pulls).

## Requirements

### Up-to-date clock

The clock must be accurate, synchronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Minimum hardware required

To run this bot we recommend you a cloud instance with a minimum of:

*   Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software Requirements

*   Python >= 3.11
*   pip
*   git
*   TA-Lib
*   virtualenv (Recommended)
*   Docker (Recommended)