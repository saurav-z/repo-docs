# NautilusTrader: High-Performance Algorithmic Trading Platform 🚀

NautilusTrader is an open-source, AI-first trading platform built for high-performance, reliable, and production-grade algorithmic trading, enabling quantitative traders to backtest, deploy, and scale strategies. [Explore the NautilusTrader Repository](https://github.com/nautechsystems/nautilus_trader).

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

*   **High Performance:** Built with Rust for speed and efficiency.
*   **Reliable & Safe:** Rust-powered with optional Redis persistence for state management.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows with Docker support.
*   **Modular & Extensible:** Integrates with any REST API or WebSocket feed via modular adapters and supports custom components.
*   **Advanced Order Types:** Includes `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`, and conditional triggers.
*   **Backtesting & Live Deployment:** Use the same strategy code for backtesting and live trading.
*   **Multi-Venue Capabilities:** Facilitates market-making and statistical arbitrage strategies across multiple venues.
*   **AI Training Ready:** Backtesting engine suitable for training AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

NautilusTrader bridges the gap between research and production, offering a high-performance, robust, and Python-native environment. It empowers quantitative traders with:

*   **Performance:** Core components written in Rust.
*   **Code Parity:** Identical strategy code for backtesting and live trading.
*   **Reduced Risk:** Enhanced risk management and type safety.
*   **Extensibility:** Leverage message bus, custom components, and adapters.

NautilusTrader addresses the traditional challenges of reimplementing strategies in different languages by leveraging Rust and Cython to provide a performant and safe environment directly within the Python ecosystem.

## Technology Stack

### Python and Cython

Python, the leading language for data science and AI, is used extensively with Cython to achieve performance.

### Rust

Rust, known for its performance, safety, and concurrency, is used for NautilusTrader's core components, ensuring efficiency and reliability.

## Integrations

NautilusTrader supports a variety of trading venues and data providers.

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

-   **ID**: The default client ID for the integrations adapter clients.
-   **Type**: The type of integration (often the venue type).

### Status

-   `building`: Under construction and likely not in a usable state.
-   `beta`: Completed to a minimally working state and in a beta testing phase.
-   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

NautilusTrader is under active development. We aim for a bi-weekly release schedule with releases on:

*   `master`: Latest released version, recommended for production.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch.

## Precision Mode

NautilusTrader offers two precision modes:

*   **High-Precision:** 128-bit integers with up to 16 decimals.
*   **Standard-Precision:** 64-bit integers with up to 9 decimals.

The official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install NautilusTrader using pip.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems Package Index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

### Development Wheels

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

### From Source

Install from source by following these steps:

1.  Install Rust and Clang (or equivalent C compiler).
2.  Clone the repository.
3.  Navigate into the directory.
4.  Use `uv sync --all-extras` to manage dependencies.
5.  Set environment variables.
6.  See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

## Redis

Redis is optional, used for the cache and message bus.

## Makefile

The Makefile simplifies development tasks.

*   `make install`: Installs in `release` build mode with all dependency groups and extras.
*   `make install-debug`: Same as `make install` but with `debug` build mode.
*   `make install-just-deps`: Installs just the `main`, `dev` and `test` dependencies (does not install package).
*   `make build`: Runs the build script in `release` build mode (default).
*   `make build-debug`: Runs the build script in `debug` build mode.
*   `make build-wheel`: Runs uv build with a wheel format in `release` mode.
*   `make build-wheel-debug`: Runs uv build with a wheel format in `debug` mode.
*   `make cargo-test`: Runs all Rust crate tests using `cargo-nextest`.
*   `make clean`: Deletes all build results, such as `.so` or `.dll` files.
*   `make distclean`: **CAUTION** Removes all artifacts not in the git index from the repository. This includes source files which have not been `git add`ed.
*   `make docs`: Builds the documentation HTML using Sphinx.
*   `make pre-commit`: Runs the pre-commit checks over all files.
*   `make ruff`: Runs ruff over all files using the `pyproject.toml` config (with autofix).
*   `make pytest`: Runs all tests with `pytest`.
*   `make test-performance`: Runs performance tests with [codspeed](https://codspeed.io).

Run `make help` for documentation on all available make targets.

## Examples

Explore indicator and strategy examples in Python and Cython.

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py)
*   [indicator](/nautilus_trader/indicators/)
*   [strategy](/nautilus_trader/examples/strategies/)
*   [backtest](/examples/backtest/)

## Docker

Docker images are available for easy setup.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

## Development

*   Refer to the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development information.
*   Run `make build-debug` for efficient development.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for Rust testing.

```bash
cargo install cargo-nextest
```

Run Rust tests with `make cargo-test`.

## Contributing

Contribute to NautilusTrader by opening an [issue](https://github.com/nautechsystems/nautilus_trader/issues) and reviewing the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader).

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens.

## License

Licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology
company specializing in the development of high-performance trading systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">