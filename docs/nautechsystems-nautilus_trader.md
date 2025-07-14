# NautilusTrader: High-Performance Algorithmic Trading Platform

**NautilusTrader empowers quantitative traders with a Python-native, AI-first platform for backtesting and live deployment of high-frequency trading strategies.**

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

*   **[Documentation](https://nautilustrader.io/docs/)**
*   **[Website](https://nautilustrader.io)**
*   **Support: [support@nautilustrader.io](mailto:support@nautilustrader.io)**
*   **[GitHub Repository](https://github.com/nautechsystems/nautilus_trader)**

## Key Features

*   **High Performance:** Core components written in Rust for speed and efficiency.
*   **Reliable and Safe:** Utilizes Rust's type and thread safety features for stability, with optional Redis-backed persistence.
*   **Cross-Platform:** Runs seamlessly on Linux, macOS, and Windows; deployable using Docker.
*   **Modular Design:** Integrates with various trading venues and data providers through modular adapters.
*   **Advanced Order Types:** Supports a range of order types, including `IOC`, `FOK`, `GTC`, and conditional triggers like `OCO` and `OUO`.
*   **Customizable:** Allows for the addition of user-defined components and system customization using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Comprehensive Backtesting:** Backtest with nanosecond resolution using historical quote tick, trade tick, bar, order book, and custom data, supporting multiple venues and strategies.
*   **Unified Strategy Deployment:** Enables identical strategy implementations for backtesting and live deployments.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies through multi-venue capabilities.
*   **AI-Ready:** Backtesting engine is fast enough to train AI trading agents (RL/ES).

## Why NautilusTrader?

NautilusTrader bridges the gap between research and production, offering a Python-native environment for quantitative traders.  It provides:

*   **High-Performance Event-Driven Execution:** Leverages native binary core components.
*   **Code Parity:** Allows the use of the same strategy code for both backtesting and live trading.
*   **Enhanced Reliability:** Risk management functionality, logical accuracy, and type safety are prioritized.
*   **Extensibility:** Custom components, adapters, data, and a message bus can be integrated.

## Core Technologies: Python, Rust, and Cython

NautilusTrader's design employs a combination of technologies for optimal performance and ease of use:

*   **Python:** Provides a straightforward syntax and is the lingua franca of data science, machine learning, and artificial intelligence, while also allowing for a rich ecosystem of libraries and communities.
*   **Rust:** Offers speed and safety, especially safe concurrency. Rust eliminates bugs at compile-time with a rich type system and memory-safety and thread-safety.
*   **Cython:** Addresses performance limitations in Python by introducing static typing, improving the speed of key components.

## Integrations

NautilusTrader features a modular design, allowing for easy integration with trading venues and data providers through adapters.

### Supported Integrations

The following integrations are currently supported. See [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details:

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
| [Interactive Brokers](https://www.interactivebrokers.com)                    | `INTERACTIVE_BROKERS` | Brokerage (multi-venue) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/ib.md)            |
| [OKX](https://okx.com)                                                       | `OKX`                 | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/building-orange) | [Guide](docs/integrations/okx.md)           |
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

### Integration Status

*   `building`: Under construction.
*   `beta`: Functioning, but in beta testing.
*   `stable`: Stabilized with a tested API.

## Versioning and Releases

NautilusTrader is under active development with a bi-weekly release schedule.

### Branches

*   `master`: Reflects the latest released version, recommended for production.
*   `nightly`: Daily snapshots of the `develop` branch, for early testing.
*   `develop`: The active development branch for contributions.

## Precision Mode

NautilusTrader supports:

*   **High-precision:** 128-bit integers with up to 16 decimals of precision.
*   **Standard-precision:** 64-bit integers with up to 9 decimals of precision.

## Installation

NautilusTrader is installable through PyPI or from source.

### Installing from PyPI

Install the latest binary wheel from PyPI using pip:

```bash
pip install -U nautilus_trader
```

### Installing from the Nautech Systems Package Index

Install the latest stable release:

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

Install pre-release versions:

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

### Installing from Source

Detailed instructions for installing from source are included in the original [README](https://github.com/nautechsystems/nautilus_trader).

## Redis

Redis is optional and is only needed for configuring a [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

The Makefile automates installation and build tasks.

## Examples

Examples of indicators, strategies, and backtests are available in the repository, including:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
*   [indicator](/nautilus_trader/indicators/) examples written in Cython.
*   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
*   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Docker images are provided, including the latest release, nightly builds, and JupyterLab examples.

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for developer information.

### Testing

Cargo-nextest is the standard Rust test runner for NautilusTrader. Run Rust tests with `make cargo-test`.

## Contributing

Contributions are welcome; please review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and roadmap for more information.  Contributions should target the `develop` branch.

## Community

Join the community on [Discord](https://discord.gg/NautilusTrader) to discuss NautilusTrader.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).