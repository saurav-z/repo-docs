# NautilusTrader: High-Performance Algorithmic Trading Platform

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

**NautilusTrader is a high-performance, open-source platform designed to empower quantitative traders with robust backtesting and live deployment capabilities.** ([Original Repo](https://github.com/nautechsystems/nautilus_trader))

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

*   **High Performance:** Core components written in Rust for speed and efficiency.
*   **Reliable & Safe:** Rust-powered type and thread safety, with optional Redis-backed state persistence.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows. Deploy using Docker.
*   **Modular Design:** Supports integrations with various exchanges and data providers via modular adapters.
*   **Advanced Order Types:** Supports a wide range of order types, including advanced and conditional triggers.
*   **Customizable:** Add user-defined components or assemble entire systems from scratch.
*   **Backtesting:** Backtest strategies with high-resolution historical data.
*   **Live Deployment:** Deploy strategies live with identical code used in backtesting.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage.
*   **AI Training Ready:** Fast backtest engine suitable for training AI trading agents.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

*   **Event-Driven Python with Native Performance:** Benefit from the power of Python with the speed of Rust.
*   **Code Parity:** Seamlessly transition between backtesting and live trading using the same strategy code.
*   **Reduced Operational Risk:** Enhanced risk management features, logical accuracy, and type safety.
*   **Extensible Architecture:** Utilize a message bus, custom components, and data adapters for tailored solutions.

## Why Python & Rust?

NautilusTrader leverages the strengths of both Python and Rust:

*   **Python:** Offers a clean syntax and vast ecosystem for data science, machine learning, and AI.
*   **Rust:** Provides performance, safety, and reliability for mission-critical components.

## Integrations

NautilusTrader offers modular integrations, enabling connection to trading venues and data providers:

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
| [Interactive Brokers](https://www.interactivebrokers.com)                    | `INTERACTIVE_BROKERS` | Brokerage (multi-venue) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/ib.md)            |
| [OKX](https://okx.com)                                                       | `OKX`                 | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/building-orange) | [Guide](docs/integrations/okx.md)           |
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule.

### Branches

-   `master`: Latest released version.
-   `nightly`: Daily snapshots of the `develop` branch.
-   `develop`: Active development branch.

## Precision Mode

NautilusTrader supports High-precision (128-bit) and Standard-precision (64-bit) for value types.

## Installation

Install NautilusTrader using `pip` or from source. We highly recommend using the [uv](https://docs.astral.sh/uv) package manager.

### Installation Options:

*   **From PyPI:** `pip install -U nautilus_trader`
*   **From Nautech Systems Package Index:**  `pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple`
*   **From Source:** Requires Rust and build tools (see the original README for detailed instructions).

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

## Redis

Redis is optional and only required if used as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

The `Makefile` automates common development tasks. See the original README for a list of targets.

## Examples

Examples of indicators, strategies, and backtesting are available in both Python and Cython.

## Docker

Docker images are provided for easy deployment.

## Development

For development guidance, see the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html).

### Testing

Use `cargo-nextest` (installed via `cargo install cargo-nextest`) for Rust tests and the command `make cargo-test`.

## Contributing

Contributions are welcome! Follow the guidelines in [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and sign the CLA.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader).

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

*NautilusTrader™ is developed and maintained by Nautech Systems. © 2015-2025 Nautech Systems Pty Ltd. All rights reserved.*