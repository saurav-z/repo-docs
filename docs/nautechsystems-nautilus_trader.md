# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

**NautilusTrader is an open-source platform empowering quantitative traders to build, backtest, and deploy high-performance algorithmic trading strategies.** Dive deeper into the platform's capabilities at the [NautilusTrader GitHub Repository](https://github.com/nautechsystems/nautilus_trader).

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

## Key Features of NautilusTrader

*   **High Performance:** Core components written in Rust with asynchronous networking using Tokio for speed and efficiency.
*   **Reliable and Safe:**  Benefit from Rust's memory and thread safety features, including optional Redis-backed state persistence.
*   **Cross-Platform Compatibility:** Deploy strategies on Linux, macOS, and Windows, plus Docker support.
*   **Modular and Flexible:** Integrate with any data source or trading venue through modular adapters (REST/WebSocket).
*   **Advanced Order Types:** Supports sophisticated order types such as `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`, and contingency orders like `OCO`, `OUO`, `OTO`.
*   **Customization:**  Extend the platform with user-defined components, the cache, and the message bus.
*   **Comprehensive Backtesting:** Test strategies with historical data (tick, trade, bar, order book) with nanosecond resolution.
*   **Seamless Live Deployment:**  Use identical strategy code for both backtesting and live trading.
*   **Multi-Venue Support:** Enables market making and statistical arbitrage strategies.
*   **AI Training Ready:** The backtesting engine is fast enough to train AI trading agents.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

*   **Blazing-Fast Python:** Leverage a high-performance, event-driven environment with core components written in Rust.
*   **Code Reusability:** Ensure consistency between backtesting and live trading with identical strategy code.
*   **Enhanced Risk Management:** Improve operational safety with enhanced risk management functionality, logical accuracy, and type safety.
*   **Extensible Architecture:**  Easily integrate custom components, custom data, custom adapters, and utilize the message bus.

## Technology Stack

NautilusTrader leverages the strengths of both Python and Rust:

*   **Python:**  Provides a clean and accessible syntax, is a de facto standard in data science and machine learning, and offers a rich ecosystem of libraries.  Cython extends Python with static typing.
*   **Rust:** Delivers high performance and memory safety, crucial for latency-sensitive trading systems. The platform uses Rust for core performance-critical components.

## Integrations

NautilusTrader features a modular design based on *adapters* to connect to a range of trading venues and data providers.

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

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

NautilusTrader is under active development and follows a bi-weekly release schedule, with updates documented in release notes.

*   `master`:  Reflects the latest released version.
*   `nightly`:  Daily snapshots from the `develop` branch for early testing.
*   `develop`:  The active development branch.

## Installation

Install NautilusTrader using the latest supported Python version inside a virtual environment.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From Nautech Systems Package Index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

## Installation from Source

1.  Install [rustup](https://rustup.rs/).
2.  Enable `cargo` in the current shell.
3.  Install [clang](https://clang.llvm.org/).
4.  Install [uv](https://docs.astral.sh/uv/getting-started/installation).
5.  Clone the source and install from the project's root directory:
    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```
6.  Set environment variables for PyO3 compilation (Linux and macOS):
    ```bash
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

## Redis

Redis is optional for use with NautilusTrader and is required when configured as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Make Targets

Use `make help` for a complete list. Some common targets include:

*   `make install`: Installs with all dependencies in release mode.
*   `make install-debug`: Installs with all dependencies in debug mode.
*   `make cargo-test`: Runs all Rust crate tests using cargo-nextest.
*   `make build-wheel`: Builds a wheel in release mode.
*   `make test-performance`: Runs performance tests with codspeed.

## Examples

Explore examples of indicators, strategies, and backtesting in Python and Cython within the `/examples` and `/nautilus_trader/examples` directories.

## Docker

Pre-built Docker images are available on GitHub Container Registry (`ghcr.io/nautechsystems/`).  Pull and run the `jupyterlab:nightly` image for a backtest example notebook:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

The [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) provides useful information for developers.

## Contributing

Contribute to NautilusTrader by opening an [issue](https://github.com/nautechsystems/nautilus_trader/issues) and discussing enhancements or bug fixes, then following the guidelines in [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md).

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader).

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">