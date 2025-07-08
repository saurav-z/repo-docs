# Freqtrade: Your Open-Source Crypto Trading Bot

Tired of manually trading crypto? **Freqtrade is a free, open-source crypto trading bot built in Python that automates your trading strategies, saving you time and maximizing your potential profits.** ([Original Repository](https://github.com/freqtrade/freqtrade))

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

<br/>
<img src="https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png" alt="Freqtrade Screenshot">

## Key Features

*   **Automated Trading:** Execute your trading strategies automatically on supported exchanges.
*   **Multi-Exchange Support:** Compatible with major crypto exchanges.
*   **Backtesting:** Test your strategies with historical data.
*   **Strategy Optimization:** Utilize machine learning for optimized strategy parameters.
*   **FreqAI:** Build smart strategies with adaptive machine learning via FreqAI.
*   **WebUI & Telegram Control:** Manage and monitor your bot via a web interface or Telegram commands.
*   **Dry-Run Mode:** Test your strategies without risking real money.
*   **Comprehensive Analysis Tools:** Detailed reporting and analysis of your trading performance.
*   **Community-Driven:** Benefit from a vibrant and supportive community.

## Supported Exchanges

Freqtrade supports a wide array of exchanges, including:

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
*   ... and more (see [exchange specific notes](docs/exchanges.md)).

**Experimental Futures Exchanges:**

*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

**Community Confirmed:**

*   Bitvavo
*   Kucoin

## Getting Started

### Quick Start

Follow the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) for a rapid setup.

### Installation

For native installation methods, see the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Important Considerations

*   **Disclaimer:** Use this software at your own risk. The developers are not responsible for your trading results.
*   **Education is Key:** Familiarize yourself with how the bot works before deploying it with real funds.
*   **Coding Knowledge:** Basic Python and coding knowledge are recommended.

## Documentation

Access the complete documentation on the [Freqtrade website](https://www.freqtrade.io).

## Basic Usage & Commands

*   **Bot Commands:** `trade`, `create-userdir`, `new-config`, `show-config`, `new-strategy`, `download-data`, `convert-data`, `convert-trade-data`, `trades-to-ohlcv`, `list-data`, `backtesting`, `backtesting-show`, `backtesting-analysis`, `edge`, `hyperopt`, `hyperopt-list`, `hyperopt-show`, `list-exchanges`, `list-markets`, `list-pairs`, `list-strategies`, `list-hyperoptloss`, `list-freqaimodels`, `list-timeframes`, `show-trades`, `test-pairlist`, `convert-db`, `install-ui`, `plot-dataframe`, `plot-profit`, `webserver`, `strategy-updater`, `lookahead-analysis`, `recursive-analysis`
*   **Telegram Commands:** `/start`, `/stop`, `/stopentry`, `/status`, `/profit`, `/forceexit`, `/fx`, `/performance`, `/balance`, `/daily`, `/help`, `/version` (See [documentation](https://www.freqtrade.io/en/latest/telegram-usage/) for full command list.)

## Development

*   **Branches:** `develop` (new features, potentially breaking), `stable` (latest stable release), `feat/*` (feature branches)
*   **Contributing:** Contribute to the project by improving documentation, submitting bug fixes, or implementing new features. See the [CONTRIBUTING document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).

## Requirements

*   **Accurate Clock:** The clock must be synchronized to an NTP server.
*   **Minimum Hardware:** 2GB RAM, 1GB disk space, 2vCPU (recommended)
*   **Software:** Python >= 3.11, pip, git, TA-Lib, virtualenv (Recommended), Docker (Recommended)

## Support

*   **Discord:** Join the [Freqtrade Discord server](https://discord.gg/p7nuUNVfP7) for help and community support.
*   **Issues:** Report bugs or request features in the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).