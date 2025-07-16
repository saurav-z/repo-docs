# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

NautilusTrader is an open-source, AI-first algorithmic trading platform designed for performance, reliability, and seamless integration, allowing you to backtest and deploy strategies with ease. ([View on GitHub](https://github.com/nautechsystems/nautilus_trader))

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

| Branch    | Version                                                                                                                                                                                                                     | Status                                                                                                                                                                                            |
| :-------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `master`  | [![version](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fnautechsystems%2Fnautilus_trader%2Fmaster%2Fversion.json)](https://packages.nautechsystems.io/simple/nautilus-trader/index.html)  | [![build](https://github.com/nautechsystems/nautilus_trader/actions/workflows/build.yml/badge.svg?branch=nightly)](https://github.com/nautechsystems/nautilus_trader/actions/workflows/build.yml) |
| `nightly` | [![version](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fnautechsystems%2Fnautilus_trader%2Fnightly%2Fversion.json)](https://packages.nautechsystems.io/simple/nautilus-trader/index.html) | [![build](https://github.com/nautechsystems/nautilus_trader/actions/workflows/build.yml/badge.svg?branch=nightly)](https://github.com/nautechsystems/nautilus_trader/actions/workflows/build.yml) |
| `develop` | [![version](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fnautechsystems%2Fnautilus_trader%2Fdevelop%2Fversion.json)](https://packages.nautechsystems.io/simple/nautilus-trader/index.html) | [![build](https://github.com/nautechsystems/nautilus_trader/actions/workflows/build.yml/badge.svg?branch=develop)](https://github.com/nautechsystems/nautilus_trader/actions/workflows/build.yml) |

| Platform           | Rust   | Python     |
| :----------------- | :----- | :--------- |
| `Linux (x86_64)`   | 1.88.0 | 3.11-3.13  |
| `Linux (ARM64)`    | 1.88.0 | 3.11-3.13  |
| `macOS (ARM64)`    | 1.88.0 | 3.11-3.13  |
| `Windows (x86_64)` | 1.88.0 | 3.11-3.13* |

\* Windows builds are currently pinned to CPython 3.13.2, see [installation guide](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/getting_started/installation.md).

- **Docs**: <https://nautilustrader.io/docs/>
- **Website**: <https://nautilustrader.io>
- **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Built with Rust for speed and efficiency, utilizing asynchronous networking with Tokio.
*   **Reliable & Safe:**  Leverages Rust's type and thread safety, with optional Redis-backed state persistence.
*   **Cross-Platform:** Runs seamlessly on Linux, macOS, and Windows; deploy using Docker.
*   **Modular & Flexible:** Supports easy integration with any REST API or WebSocket feed via modular adapters.
*   **Advanced Order Types:** Includes  `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`, execution instructions and contingency orders.
*   **Customizable & Extensible:**  Add user-defined components, or assemble entire systems from scratch leveraging the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Backtesting Capabilities:** Run simulations with multiple venues, instruments, and strategies using historical data with nanosecond resolution.
*   **Live Deployment:** Use the same strategy code for backtesting and live trading with no code changes.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   **AI Training:** Backtest engine designed for training AI trading agents (RL/ES).

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png" alt="nautilus-trader">

## Why Choose NautilusTrader?

NautilusTrader provides a robust, Python-native environment for quantitative traders, offering:

*   **Exceptional Performance:** Native binary core components written in Rust and Cython.
*   **Code Reusability:** Identical strategy code for backtesting and live trading, reducing development time.
*   **Enhanced Security:**  Improved risk management and type safety, which lowers operational risk.
*   **Extensibility:** Highly extendable through message bus, custom components, custom data, and custom adapters.

## Technologies & Architecture

NautilusTrader is designed with a focus on speed, safety, and ease of use. The platform uses:

*   **Rust:** For core performance-critical components, ensuring speed and memory efficiency.
*   **Python:** Utilizes Python for strategy development and backtesting within a performant and reliable native environment.
*   **Cython:**  Bridging the gap between Python and Rust for efficient integration.

## Integrations

NautilusTrader offers modular *adapters* for easy integration with trading venues and data providers.

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

| Name                                                                         | ID                    | Type                    | Status                                                  | Docs                                        |
| :--------------------------------------------------------------------------- | :-------------------- | :---------------------- | :------------------------------------------------------ | :------------------------------------------ |
| [Betfair](https://betfair.com)                                               | `BETFAIR`             | Sports Betting Exchange | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/betfair.md)       |
| [Binance](https://binance.com)                                               | `BINANCE`             | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/binance.md)       |
| [Binance US](https://binance.us)                                             | `BINANCE`             | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/binance.md)       |
| [Binance Futures](https://www.binance.com/en/futures)                        | `BINANCE`             | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/binance.md)       |
| [Bybit](https://www.bybit.com)                                               | `BYBIT`               | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/bybit.md)         |
| [Coinbase International](https://www.coinbase.com/en/international-exchange) | `COINBASE_INTX`       | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/coinbase_intx.md) |
| [Databento](https://databento.com)                                           | `DATABENTO`           | Data Provider           | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/databento.md)     |
| [dYdX](https://dydx.exchange/)                                               | `DYDX`                | Crypto Exchange (DEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/dydx.md)          |
| [Hyperliquid](https://hyperliquid.xyz)                                       | `HYPERLIQUID`         | Crypto Exchange (DEX)   | ![status](https://img.shields.io/badge/building-orange) | [Guide](docs/integrations/hyperliquid.md)   |
| [Interactive Brokers](https://www.interactivebrokers.com)                    | `INTERACTIVE_BROKERS` | Brokerage (multi-venue) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/ib.md)            |
| [OKX](https://okx.com)                                                       | `OKX`                 | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/building-orange) | [Guide](docs/integrations/okx.md)           |
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

## Installation

Follow the steps below to install NautilusTrader:

### Prerequisites

*   Python 3.11 or higher
*   [uv](https://docs.astral.sh/uv/) - Recommended package manager

### Installation Steps

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Or .venv\Scripts\activate.ps1 for PowerShell on Windows
    ```
2.  **Install NautilusTrader using pip**:
    ```bash
    pip install -U nautilus_trader
    ```

### Alternative Installation Methods

*   **Install from the Nautech Systems Package Index** for stable or development wheels.
*   **Install from source** if you want to compile from the git repository, after installing `rustup`, `clang` and `uv`. See the detailed [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation).

## Community & Resources

*   **Documentation:** [https://nautilustrader.io/docs/](https://nautilustrader.io/docs/)
*   **Discord:** [https://discord.gg/NautilusTrader](https://discord.gg/NautilusTrader)
*   **Website:** [https://nautilustrader.io](https://nautilustrader.io)
*   **Support:** [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Contributing

We welcome contributions!  Please review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file and open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) before submitting pull requests.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).