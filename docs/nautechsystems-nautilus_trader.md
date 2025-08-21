# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader empowers quantitative traders with a high-performance, AI-first platform for backtesting and deploying automated trading strategies.** ([Original Repo](https://github.com/nautechsystems/nautilus_trader))

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

| Platform           | Rust   | Python    |
| :----------------- | :----- | :-------- |
| `Linux (x86_64)`   | 1.89.0 | 3.11-3.13 |
| `Linux (ARM64)`    | 1.89.0 | 3.11-3.13 |
| `macOS (ARM64)`    | 1.89.0 | 3.11-3.13 |
| `Windows (x86_64)` | 1.89.0 | 3.11-3.13 |

- **Docs**: <https://nautilustrader.io/docs/>
- **Website**: <https://nautilustrader.io>
- **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Introduction

NautilusTrader is an open-source, production-grade algorithmic trading platform built for performance, reliability, and flexibility, enabling quantitative traders to seamlessly backtest and deploy trading strategies.  It's designed with an "AI-first" approach, ensuring parity between research/backtesting and live deployment in a Python-native environment. The platform prioritizes software correctness and safety, supporting Python-native, mission-critical trading system workloads. Asset-class-agnostic, it integrates with various venues via modular adapters.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Key Features

*   **High Performance:** Core written in Rust for speed, with asynchronous networking using [tokio](https://crates.io/crates/tokio).
*   **Reliable & Safe:**  Rust-powered type- and thread-safety, optional Redis-backed persistence.
*   **Cross-Platform:** OS independent, runs on Linux, macOS, and Windows, deployable via Docker.
*   **Modular & Flexible:**  Modular adapters for integrating any REST API or WebSocket feed.
*   **Advanced Order Types:** Time in force, execution instructions, and contingency orders.
*   **Customizable Architecture:** Allows for user-defined components and system-wide customization.
*   **Comprehensive Backtesting:** Backtest with various data types, instruments, and strategies simultaneously.
*   **Seamless Live Deployment:** Use the same strategy code for backtesting and live trading.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies across multiple venues.
*   **AI Training Ready:** Backtest engine optimized for training AI trading agents (RL/ES).

![Alt text](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png "nautilus")

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral.
> The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

*   **Enhanced Performance:** Leverages high-performance event-driven Python with native binary components.
*   **Code Consistency:** Ensures parity between backtesting and live trading with identical strategy code.
*   **Reduced Risk:** Improves risk management with enhanced functionality and type safety.
*   **Extensibility:** Offers flexibility with message bus, custom components and actors, and custom data adapters.

NautilusTrader overcomes the challenges of traditional trading strategy development, where research often relies on vectorized Python, while live trading requires reimplementation in statically-typed languages like C++. This platform allows for the creation of performant trading systems via Rust and Cython, offering a Python-native environment for professional quantitative traders and trading firms.

## Technology Stack

### Why Python?

Python's clean syntax and widespread adoption in data science, machine learning, and AI make it the **de facto lingua franca** for development.

### Why Rust?

Rust's performance, safety, and concurrency features make it ideal for building the core performance-critical components of NautilusTrader. The project adheres to the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html).

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader integrates with various trading venues and data providers via modular adapters.

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

| Name                                                                         | ID                    | Type                    | Status                                                  | Docs                                        |
| :--------------------------------------------------------------------------- | :-------------------- | :---------------------- | :------------------------------------------------------ | :------------------------------------------ |
| [Betfair](https://betfair.com)                                               | `BETFAIR`             | Sports Betting Exchange | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/betfair.md)       |
| [Binance](https://binance.com)                                               | `BINANCE`             | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/binance.md)       |
| [Binance US](https://binance.us)                                             | `BINANCE`             | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/binance.md)       |
| [Binance Futures](https://www.binance.com/en/futures)                        | `BINANCE`             | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/binance.md)       |
| [BitMEX](https://www.bitmex.com)                                             | `BITMEX`              | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/building-orange) | [Guide](docs/integrations/bitmex.md)        |
| [Bybit](https://www.bybit.com)                                               | `BYBIT`               | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/bybit.md)         |
| [Coinbase International](https://www.coinbase.com/en/international-exchange) | `COINBASE_INTX`       | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/coinbase_intx.md) |
| [Databento](https://databento.com)                                           | `DATABENTO`           | Data Provider           | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/databento.md)     |
| [dYdX](https://dydx.exchange/)                                               | `DYDX`                | Crypto Exchange (DEX)   | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/dydx.md)          |
| [Hyperliquid](https://hyperliquid.xyz)                                       | `HYPERLIQUID`         | Crypto Exchange (DEX)   | ![status](https://img.shields.io/badge/building-orange) | [Guide](docs/integrations/hyperliquid.md)   |
| [Interactive Brokers](https://www.interactivebrokers.com)                    | `INTERACTIVE_BROKERS` | Brokerage (multi-venue) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/ib.md)            |
| [OKX](https://okx.com)                                                       | `OKX`                 | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/beta-yellow)     | [Guide](docs/integrations/okx.md)           |
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

- **ID**: The default client ID for the integrations adapter clients.
- **Type**: The type of integration (often the venue type).

### Status

- `building`: Under construction and likely not in a usable state.
- `beta`: Completed to a minimally working state and in a beta testing phase.
- `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule, with breaking changes documented in the release notes.

### Branches

*   `master`: Latest released version (production).
*   `nightly`: Daily snapshots of the `develop` branch for testing.
*   `develop`: Active development branch.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader offers two precision modes for `Price`, `Quantity`, and `Money` types:

*   **High-precision**: 128-bit integers, up to 16 decimals of precision.
*   **Standard-precision**: 64-bit integers, up to 9 decimals of precision.

> [!NOTE]
>
> The official Python wheels ship in high-precision (128-bit) mode on Linux and macOS and standard-precision (64-bit) on Windows.

**Rust feature flag**: Enable high-precision in Rust with `high-precision` feature:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install NautilusTrader using the latest supported Python version inside a virtual environment.

**Installation Methods**:

1.  Pre-built binary wheel from PyPI or the Nautech Systems package index.
2.  Build from source.

> [!TIP]
>
>  Use the [uv](https://docs.astral.sh/uv/getting-started/installation) package manager for a seamless experience.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From Nautech Systems Package Index

Stable releases:

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

Pre-release (including development wheels):

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

Specific development wheel:

```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

#### Available versions

```bash
curl -s https://packages.nautechsystems.io/simple/nautilus-trader/index.html | grep -oP '(?<=<a href=")[^"]+(?=")' | awk -F'#' '{print $1}' | sort
```

### From Source

1.  Install [rustup](https://rustup.rs/) and dependencies.
2.  Install [clang](https://clang.llvm.org/)
3.  Install [uv](https://docs.astral.sh/uv/getting-started/installation)
4.  Clone the source and install:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

5.  Set environment variables (Linux and macOS only):

    ```bash
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for other options and further details.

## Redis

Redis is optional and required only for cache and message bus backends.  See the Redis section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for details.

## Makefile

The `Makefile` simplifies build and installation tasks.

- `make install`: Installs in `release` build mode.
- `make install-debug`: Installs in `debug` build mode.
- `make build`: Runs the build script in `release` build mode (default).
- `make build-wheel`: Runs uv build with a wheel format in `release` mode.
- `make cargo-test`: Runs all Rust crate tests using `cargo-nextest`.
- `make clean`: Deletes all build results.
- `make distclean`: Removes all artifacts not in the git index from the repository.
- `make docs`: Builds the documentation HTML.
- `make pre-commit`: Runs the pre-commit checks.
- `make ruff`: Runs ruff over all files using the `pyproject.toml` config (with autofix).
- `make pytest`: Runs all tests with `pytest`.

> [!TIP]
>
> Run `make help` for a list of make targets.

> [!TIP]
>
> See the [crates/infrastructure/TESTS.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/crates/infrastructure/TESTS.md) file for running the infrastructure integration tests.

## Examples

Find indicator and strategy examples in Python and Cython:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
*   [indicator](/nautilus_trader/indicators/) examples written in Cython.
*   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
*   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Docker images are built with `python:3.12-slim`.

*   `nautilus_trader:latest`: Latest release.
*   `nautilus_trader:nightly`: Head of `nightly` branch.
*   `jupyterlab:latest`: Latest release with JupyterLab.
*   `jupyterlab:nightly`: Head of `nightly` with JupyterLab and example notebooks.

Pull the images:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

Run the backtest example:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Access JupyterLab: `http://127.0.0.1:8888/lab`

> [!WARNING]
>
> The Jupyter notebook logs will show ERROR level only due to rate limiting issues, fix in progress.

## Development

For developers, `make build-debug` enables efficient development workflows. See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for more.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for Rust testing.

```bash
cargo install cargo-nextest
```

> [!TIP]
>
> Run Rust tests with `make cargo-test`, using **cargo-nextest**.

## Contributing

Contributions are welcome!  Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your ideas and review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file. Pull requests should target the `develop` branch. A Contributor License Agreement (CLA) is required.

> [!NOTE]
>
> Pull requests should target the `develop` branch (the default branch). This is where new features and improvements are integrated before release.

## Community

Join our community on [Discord](https://discord.gg/NautilusTrader) to chat and stay updated.

> [!WARNING]
>
> NautilusTrader does not issue or endorse any cryptocurrency tokens. Report any suspicious activity.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems. For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and summaries:

*   **SEO Optimization:** Keyword-rich headings (e.g., "High-Performance Algorithmic Trading Platform"), use of "algorithmic trading," "backtesting," "live trading."
*   **Concise Summary:** The one-sentence hook is at the beginning.
*   **Key Features:** Presented in a clear, bulleted format.
*   **Clear Structure:** Uses headings and subheadings for easy navigation.
*   **Concise Language:** Avoids unnecessary verbosity.
*   **Actionable Information:** Provides installation instructions and contribution guidelines.
*   **Community Building:** Includes a call to action to join the Discord.
*   **Complete Information:** Retains all essential information from the original README while being significantly more readable and user-friendly.
*   **Warnings/Important Notes:** Kept the important notes and warnings to help the user
*   **Consistent Style:** Updated style and format for easier consumption of information.
*   **Emphasis and Keywords:** Bolded important information.
*   **Added alt tags to images.**
*   **Corrected markdown syntax.**