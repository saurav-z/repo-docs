# NautilusTrader: High-Performance Algorithmic Trading Platform

**NautilusTrader is an open-source, AI-first algorithmic trading platform built for speed, reliability, and ease of use.** ([See the original repo](https://github.com/nautechsystems/nautilus_trader))

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

NautilusTrader is designed for quantitative traders and trading firms looking to backtest and deploy algorithmic trading strategies efficiently. This platform addresses the challenge of maintaining parity between Python research/backtesting and live trading environments.

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

*   **High Performance:** Core components written in Rust with asynchronous networking using [tokio](https://crates.io/crates/tokio).
*   **Reliable and Safe:** Type- and thread-safety powered by Rust, with optional Redis-backed state persistence.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows. Deploy using Docker.
*   **Modular and Flexible:** Modular adapters for easy integration with any REST API or WebSocket feed.
*   **Advanced Order Types:** Supports advanced order types (IOC, FOK, GTC, etc.) and conditional triggers.
*   **Customizable and Extensible:** Add custom components or assemble systems from scratch using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Comprehensive Backtesting:** Backtest strategies with high-resolution data, including tick, trade, bar, and order book data. Supports multiple venues and instruments.
*   **Seamless Live Deployment:** Identical strategy implementations for backtesting and live trading.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies.
*   **AI Training:** Backtest engine designed for training AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

*   **Event-Driven Performance:** Native binary core components for high-speed Python execution.
*   **Code Consistency:**  Identical strategy code for both backtesting and live trading, streamlining development.
*   **Enhanced Security:** Enhanced risk management and type safety reduce operational risk.
*   **Extensible Platform:** Easily extendable via a message bus, custom components and actors.

## Technology Stack

NautilusTrader leverages a powerful combination of technologies:

*   **Rust:** For performance-critical components, ensuring speed and reliability.
*   **Python:** For strategy development, backtesting, and integration.
*   **Cython:** To introduce static typing for the rich Python ecosystem.

## Integrations

NautilusTrader is designed for modularity, working with adapters to connect to various trading venues and data providers.

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

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

*   **Bi-weekly release schedule** with breaking changes documented in release notes.
*   **Branches:** `master` (latest release), `nightly` (daily snapshots), and `develop` (active development).
*   **Stable API Roadmap**:  Aiming for a stable API for version 2.x, with a formal deprecation process for changes.

## Precision Mode

NautilusTrader supports two precision modes:

*   **High-precision**: 128-bit integers with up to 16 decimals.
*   **Standard-precision**: 64-bit integers with up to 9 decimals.

The default is high-precision on Linux and macOS.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Choose from these options to install:

1.  Pre-built binary wheels from PyPI *or* the Nautech Systems package index.
2.  Build from source.

> [!TIP]
>
> Install using the [uv](https://docs.astral.sh/uv) package manager with a "vanilla" CPython.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

#### Stable wheels

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development wheels

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

### From Source

1.  Install [rustup](https://rustup.rs/).
2.  Enable `cargo` in the current shell.
3.  Install [clang](https://clang.llvm.org/).
4.  Install uv.
5.  Clone the source and install from the project's root directory:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

6.  Set environment variables (Linux and macOS only):

    ```bash
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

## Redis

Redis is optional and used as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus). See the **Redis** section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for details.

## Makefile

The `Makefile` streamlines installation, build, and development tasks. Use `make help` to list the available targets.

## Examples

Explore indicator and strategy examples in Python and Cython:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
*   [indicator](/nautilus_trader/indicators/) examples written in Cython.
*   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
*   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Pre-built Docker images are available:

*   `nautilus_trader:latest` (latest release)
*   `nautilus_trader:nightly` (nightly branch)
*   `jupyterlab:latest` (with JupyterLab)
*   `jupyterlab:nightly` (with JupyterLab)

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

For a smooth development experience, see the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html).

> [!TIP]
>
> Run `make build-debug` to compile after changes to Rust or Cython code.

### Testing with Rust

NautilusTrader uses [cargo-nextest](https://nexte.st) for efficient Rust testing.  Install it via:

```bash
cargo install cargo-nextest
```

## Contributing

We welcome contributions!  Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss enhancements or fixes.  Review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file for guidelines and CLA details. Pull requests should target the `develop` branch.

## Community

Join our [Discord](https://discord.gg/NautilusTrader) community for discussions and updates.

> [!WARNING]
>
> NautilusTrader does not issue any tokens.  Official communications are through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader), or X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).  Report suspicious activity to the platform and to <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology
company specializing in the development of high-performance trading systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">