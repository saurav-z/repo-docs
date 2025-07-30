# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

NautilusTrader is an open-source, AI-first algorithmic trading platform designed for high-performance backtesting and live deployment.  Explore the original repo: [https://github.com/nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader).

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

*   **High Performance:**  Core components written in Rust, with asynchronous networking for speed and efficiency.
*   **Reliability:** Rust-powered type and thread safety, plus optional Redis-backed state persistence.
*   **Cross-Platform:** Compatible with Linux, macOS, and Windows; deployable using Docker.
*   **Modular Design:** Integrates with any REST API or WebSocket feed via customizable adapters.
*   **Advanced Order Types:** Supports advanced order types (IOC, FOK, GTC, GTD, etc.), execution instructions (post-only, reduce-only), and contingency orders (OCO, OUO, OTO).
*   **Customization:** Add user-defined components and assemble systems from scratch using the cache and message bus.
*   **Backtesting & Live Trading Parity:** Identical strategy implementations for backtesting and live deployment.
*   **Multi-Venue:** Enables market-making and statistical arbitrage strategies across multiple venues.
*   **AI Training Ready:** Backtest engine is fast enough for training AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

NautilusTrader bridges the gap between Python research and production trading environments, offering:

*   **High-Performance Event-Driven Python:** Leverage the power of Python with native binary core components.
*   **Code Parity:** Use the same strategy code for both backtesting and live trading.
*   **Reduced Operational Risk:** Benefit from enhanced risk management, logical accuracy, and type safety.
*   **Extensibility:**  Customize your trading system with the message bus, custom components, and adapters.

## Technical Overview

NautilusTrader addresses the limitations of traditional Python-based trading systems by incorporating Rust and Cython. This allows for high-performance, type-safe core components while providing a Python-native environment suitable for professional quantitative traders.

### Why Python?

Python is a widely-used, versatile language perfect for data science, machine learning, and AI.

### Why Rust?

Rust delivers performance and safety, crucial for mission-critical trading systems.

The project is committed to eliminating soundness bugs, backed by the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html).

> **MSRV:** NautilusTrader's MSRV (Minimum Supported Rust Version) generally follows the latest stable Rust release.

## Integrations

NautilusTrader uses modular *adapters* for connecting to trading venues and data providers, translating their APIs into a unified interface.

Supported integrations include:

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
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

### Status Definitions

*   `building`: Under construction.
*   `beta`: Minimally working, in beta testing.
*   `stable`: Stabilized API, tested by developers and users.

## Versioning and Releases

NautilusTrader is under active development, following a bi-weekly release schedule.

### Branches

*   `master`: Latest released version.
*   `nightly`: Daily snapshots of the `develop` branch for testing.
*   `develop`: Active development branch.

> [!NOTE]
> A stable API for version 2.x is planned.

## Precision Modes

NautilusTrader supports two precision modes:

*   **High-Precision:** 128-bit integers, up to 16 decimals.
*   **Standard-Precision:** 64-bit integers, up to 9 decimals.

> [!NOTE]
> High-precision (128-bit) mode is the default for Linux and macOS wheels, while standard-precision (64-bit) is used on Windows.

## Installation

Install [nautilus_trader](https://pypi.org/project/nautilus_trader/) in a virtual environment using these methods:

1.  Pre-built binary wheel from PyPI or Nautech Systems package index.
2.  Build from source.

> [!TIP]
> Use the [uv](https://docs.astral.sh/uv) package manager.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From Nautech Systems Package Index

Install stable releases:

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

Install pre-releases (including development wheels):

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

Install a specific development wheel:

```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

View available versions:

```bash
curl -s https://packages.nautechsystems.io/simple/nautilus-trader/index.html | grep -oP '(?<=<a href=")[^"]+(?=")' | awk -F'#' '{print $1}' | sort
```

### From Source

1.  Install [rustup](https://rustup.rs/).
2.  Enable `cargo`.
3.  Install [clang](https://clang.llvm.org/).
4.  Install [uv](https://docs.astral.sh/uv/getting-started/installation).
5.  Clone the source and install:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

    Set environment variables for PyO3 compilation (Linux and macOS):

    ```bash
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

## Redis

Redis is optional, used as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

Use the Makefile for common tasks, including:

*   `make install`
*   `make build`
*   `make test`
*   `make docs`

Run `make help` for details.

## Examples

Find examples of indicators and strategies in Python and Cython:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py)
*   [indicator](/nautilus_trader/indicators/)
*   [strategy](/nautilus_trader/examples/strategies/)
*   [backtest](/examples/backtest/)

## Docker

Docker images are available with the following tags:

*   `nautilus_trader:latest`
*   `nautilus_trader:nightly`
*   `jupyterlab:latest`
*   `jupyterlab:nightly`

Pull and run a Docker container:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Open your browser: `http://127.0.0.1:8888/lab`

> [!WARNING]
> Set `log_level` to `ERROR` in examples.

## Development

Refer to the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html).

> [!TIP]
> Run `make build-debug` for efficient development.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for Rust testing.

Install:

```bash
cargo install cargo-nextest
```

Run tests:

```bash
make cargo-test
```

## Contributing

Contribute by opening an [issue](https://github.com/nautechsystems/nautilus_trader/issues) and following the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines.

> [!NOTE]
>  PRs should target the `develop` branch.

## Community

Join the [Discord](https://discord.gg/NautilusTrader) community.

> [!WARNING]
>
>  NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens. Official communications are through the website, [Discord server](https://discord.gg/NautilusTrader), and [X (Twitter) account](https://x.com/NautilusTrader).

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).  Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">