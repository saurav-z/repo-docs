# Freqtrade: Your Open-Source Crypto Trading Bot

Freqtrade is a powerful, free, and open-source crypto trading bot that empowers you to automate your trading strategies.  [Check out the original repository](https://github.com/freqtrade/freqtrade)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)

## Key Features

*   **Automated Trading:** Execute your crypto trading strategies automatically 24/7.
*   **Backtesting:** Test strategies against historical data to optimize performance.
*   **Strategy Optimization:** Utilize machine learning to fine-tune buy/sell strategies.
*   **Exchange Support:** Compatible with a wide range of major crypto exchanges, including Binance, Bybit, and OKX. (See list below)
*   **Web UI and Telegram Integration:** Manage your bot through a built-in web UI or Telegram commands.
*   **Dry-Run Mode:** Simulate trading without risking real money.
*   **Risk Management:** Display profit/loss in fiat and performance reports to understand current trades.
*   **FreqAI:** Adaptive prediction modeling to build smarter strategies using machine learning.

## Supported Exchanges

Freqtrade supports a wide range of exchanges:

*   Binance
*   Bitmart
*   BingX
*   Bybit
*   Gate.io
*   HTX
*   Hyperliquid (DEX)
*   Kraken
*   OKX / MyOKX (OKX EEA)

And many others through CCXT.
*See [exchange specific notes](docs/exchanges.md) to learn about special configurations needed for each exchange.*

### Supported Futures Exchanges (experimental)
*   Binance
*   Gate.io
*   Hyperliquid (DEX)
*   OKX
*   Bybit

*Please see [exchange specific notes](docs/exchanges.md), and the [trading with leverage](docs/leverage.md) documentation before using Futures exchanges.*

### Community Tested Exchanges
*   Bitvavo
*   Kucoin

## Getting Started

### Quickstart
The [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) offers a great way to get started quickly.

### Installation
See the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/) for native installation methods.

## Basic Usage & Commands

### Bot Commands

*   `/start`: Starts the trader.
*   `/stop`: Stops the trader.
*   `/stopentry`: Stop entering new trades.
*   `/status <trade_id>|[table]`: Lists all or specific open trades.
*   `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
*   `/profit_long [<n>]`: Lists cumulative profit from all finished long trades, over the last n days.
*   `/profit_short [<n>]`: Lists cumulative profit from all finished short trades, over the last n days.
*   `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
*   `/fx <trade_id>|all`: Alias to `/forceexit`
*   `/performance`: Show performance of each finished trade grouped by pair
*   `/balance`: Show account balance per currency.
*   `/daily <n>`: Shows profit or loss per day, over the last n days.
*   `/help`: Show help message.
*   `/version`: Show version.

### Full Command List
More details and the full command list can be found in the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

## Development Branches
*   `develop`: Contains new features and potentially breaking changes.
*   `stable`: Latest stable release.
*   `feat/*`: Feature branches, for testing new specific features.

## Support and Contribution

*   **Discord:** Join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7) for support and community interaction.
*   **Issues:** Report bugs or suggest improvements via the [issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue).
*   **Feature Requests:** Submit feature requests [here](https://github.com/freqtrade/freqtrade/labels/enhancement).
*   **Pull Requests:** Contribute to the project by submitting pull requests [here](https://github.com/freqtrade/freqtrade/pulls).  Review the [Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md) for details.

## Disclaimer

*This software is for educational purposes only. Use at your own risk. The authors and affiliates are not responsible for trading results.*

Always start in Dry-run mode and understand how the bot works before trading with real money.

## Requirements
*   **Up-to-date Clock:** Accurate time synchronization is crucial.
*   **Hardware:** Minimum hardware requirements are 2GB RAM, 1GB disk space, 2vCPU.
*   **Software:**
    *   Python >= 3.11
    *   pip
    *   git
    *   TA-Lib
    *   virtualenv (Recommended)
    *   Docker (Recommended)