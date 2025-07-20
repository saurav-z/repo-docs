# NautilusTrader: High-Performance Algorithmic Trading Platform

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

**NautilusTrader is an open-source, AI-first algorithmic trading platform designed for speed, reliability, and flexibility, empowering quantitative traders to build and deploy high-performance trading strategies.** Explore the power of [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) and elevate your trading strategies.

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

- **Docs:** <https://nautilustrader.io/docs/>
- **Website:** <https://nautilustrader.io>
- **Support:** [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Core written in Rust for speed and efficiency.
*   **Reliable & Safe:** Rust-powered type and thread safety, optional Redis persistence.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows; deploy with Docker.
*   **Modular & Adaptable:** Integrate with any REST API or WebSocket feed.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, and more.
*   **Customizable:** Add custom components and build systems from scratch.
*   **Backtesting & Live Deployment:** Use the same strategy code for both.
*   **Multi-Venue Support:** Facilitates market-making and arbitrage strategies.
*   **AI-Ready:** Backtesting engine fast enough to train AI trading agents.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why NautilusTrader?

*   **Performance:** Fast, event-driven Python with native binary components.
*   **Consistency:** Identical strategy code for backtesting and live trading.
*   **Risk Management:** Enhanced risk management, logical accuracy, and type safety.
*   **Extensibility:** Leverage message bus, custom components, and adapters.

## Why Python?

Python, the *lingua franca* of data science and AI, offers a clean and straightforward syntax. NautilusTrader leverages Python's vast ecosystem while addressing its performance limitations through core components written in Rust and Cython.

## Why Rust?

Rust delivers performance, safety, and memory efficiency. Its rich type system guarantees memory-safety and thread-safety, eliminating bugs at compile-time, making it ideal for mission-critical trading systems.

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader uses modular *adapters* to integrate with various trading venues and data providers.

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

### Status

- `building`: Under construction and likely not in a usable state.
- `beta`: Completed to a minimally working state and in a beta testing phase.
- `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

NautilusTrader is under active development with a bi-weekly release schedule.

### Branches

*   `master`: Latest released version; recommended for production.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x**.

## Precision Mode

NautilusTrader supports high-precision (128-bit) and standard-precision (64-bit) modes for core value types.  The default is high-precision on Linux and macOS, and standard-precision on Windows.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install using pip or build from source.  Using [uv](https://docs.astral.sh/uv) is highly recommended.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems Package Index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple  # For pre-releases
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple # Specific version
```

### From Source

1.  Install rustup and other build tools as described in the original documentation.
2.  Clone the repository:
    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```
3.  Set environment variables (Linux/macOS):

    ```bash
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"  # Adjust Python version
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```
4.  Install from source.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for other options and further details.

## Redis

Using Redis is optional, and used as a backend for cache or message bus functionality.

## Makefile

The `Makefile` simplifies common tasks:

-   `make install`: Installs in `release` build mode.
-   `make build`: Runs the build script in `release` build mode (default).
-   `make test-performance`: Runs performance tests with [codspeed](https://codspeed.io).
-   `make help`: Shows all available make targets.

## Examples

Find examples in Python and Cython for indicators, strategies, and backtesting.

## Docker

Use Docker containers for easy deployment, with `latest`, `nightly`, `jupyterlab:latest`, and `jupyterlab:nightly` tags.

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development information.

> [!TIP]
>
> Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

### Testing with Rust

NautilusTrader uses [cargo-nextest](https://nexte.st) for reliable Rust testing.

```bash
cargo install cargo-nextest
make cargo-test # Runs Rust tests
```

## Contributing

Contribute to NautilusTrader by opening an [issue](https://github.com/nautechsystems/nautilus_trader/issues) and following the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines.

> [!NOTE]
>
> Pull requests should target the `develop` branch.

## Community

Join the [Discord](https://discord.gg/NautilusTrader) community.

> [!WARNING]
>
> NautilusTrader does not promote cryptocurrency tokens. Report suspicious activity at <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems. For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">