# <img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="Nautilus Trader Logo">

**NautilusTrader: The open-source algorithmic trading platform built for speed, reliability, and AI-driven strategies.**

[View the Original Repository](https://github.com/nautechsystems/nautilus_trader)

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

## About NautilusTrader

NautilusTrader is an open-source, high-performance algorithmic trading platform designed for quantitative traders. Built with a Python-native environment, it allows you to develop, backtest, and deploy trading strategies with speed and reliability.  The platform is *AI-first* and asset-class-agnostic, capable of handling diverse financial instruments across multiple venues simultaneously.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Key Features

*   **High Performance:** Core components written in Rust with asynchronous networking via [tokio](https://crates.io/crates/tokio).
*   **Reliable & Safe:**  Rust-powered type and thread safety, with optional Redis-backed state persistence.
*   **Cross-Platform:**  Runs on Linux, macOS, and Windows, with Docker support for deployment.
*   **Modular Design:** Adaptable to any REST API or WebSocket feed with modular adapters.
*   **Advanced Order Types:** Supports advanced order types and triggers like `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`, and more, including execution instructions and contingency orders.
*   **Customizable:** Build custom components or entire systems leveraging the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Comprehensive Backtesting:**  Backtest with nanosecond resolution using historical data, including quote, trade, and order book data across multiple venues and instruments.
*   **Seamless Live Deployment:** Use the same strategy code for backtesting and live trading.
*   **Multi-Venue Support:**  Facilitates market-making and statistical arbitrage strategies.
*   **AI-Ready Backtesting:** Backtest engine is fast enough for training AI trading agents (RL/ES).

![Alt text](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png "nautilus")

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral. The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

*   **High-Performance Python:**  Leverages native binary core components for speed.
*   **Code Parity:**  Write trading strategies once and deploy them in both backtesting and live environments.
*   **Reduced Risk:** Enhanced risk management functionality, logical accuracy, and type safety for safer trading.
*   **Extensibility:**  Utilizes a message bus, custom components, actors, custom data, and custom adapters.

NautilusTrader eliminates the need for reimplementing strategies in lower-level languages by providing a Python-native environment. This approach allows quantitative traders and trading firms to use the tools they know best, while gaining the performance of compiled languages like Rust and Cython.

## Why Python?

Python is the world's most popular programming language, favored for data science, machine learning, and artificial intelligence.  It's known for its clear syntax and extensive ecosystem.  Cython improves performance by adding static typing to Python, addressing limitations for large-scale, latency-sensitive systems.

## Why Rust?

Rust is a modern, multi-paradigm language built for performance and safety, focusing on safe concurrency. Rust offers "blazingly fast" performance with memory efficiency comparable to C and C++, without a garbage collector.  Its type system guarantees memory- and thread-safety at compile time, reducing runtime bugs.

The project utilizes Rust for its core performance-critical components. Python bindings are implemented via Cython and [PyO3](https://pyo3.rs).

This project makes the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html):

> “The intent of this project is to be free of soundness bugs.
> The developers will do their best to avoid them, and welcome help in analyzing and fixing them.”

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader supports a range of integrations via modular adapters, enabling connectivity to various trading venues and data providers.

The following integrations are currently supported; see [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details:

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

-   **ID**:  The default client ID for the integrations adapter clients.
-   **Type**:  The type of integration (often the venue type).

### Status

-   `building`: Under construction and likely not in a usable state.
-   `beta`: Completed to a minimally working state and in a beta testing phase.
-   `stable`: Stabilized feature set and API; integration tested by developers and users.

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

**NautilusTrader is under active development.** Expect ongoing changes, and be sure to consult release notes.

We use a **bi-weekly release schedule**, with potential delays for major features.

### Branches

*   `master`:  Latest released version - recommended for production.
*   `nightly`: Daily snapshots of `develop` for testing. Merged at **14:00 UTC**.
*   `develop`:  Active development branch.

> [!NOTE]
>
> The [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x**. Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.

## Precision Mode

NautilusTrader supports two precision modes for its core value types:

-   **High-precision**: 128-bit integers with up to 16 decimals.
-   **Standard-precision**: 64-bit integers with up to 9 decimals.

> [!NOTE]
>
> The Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS, but uses standard-precision (64-bit) on Windows.

**Rust feature flag**:  To enable high-precision in Rust:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Follow the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for the detailed installation procedure. Here are the key steps:

We recommend using the latest supported version of Python and installing [nautilus\_trader](https://pypi.org/project/nautilus_trader/) inside a virtual environment to isolate dependencies.

**There are two supported ways to install**:

1.  Pre-built binary wheel from PyPI *or* the Nautech Systems package index.
2.  Build from source.

> [!TIP]
>
> We highly recommend installing using the [uv](https://docs.astral.sh/uv) package manager with a "vanilla" CPython.
>
> Conda and other Python distributions *may* work but aren’t officially supported.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

To install a specific development wheel:

```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

### From Source

1.  Install [rustup](https://rustup.rs/) (the Rust toolchain installer).
2.  Enable `cargo`.
3.  Install [clang](https://clang.llvm.org/).
4.  Install uv (see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation) for more details):
5.  Clone the source with `git`, and install from the project's root directory:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

6.  Set environment variables for PyO3 compilation (Linux and macOS only).

## Redis

[Redis](https://redis.io) is **optional** and is only required if configured for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
See the **Redis** section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for details.

## Makefile

The `Makefile` automates installation and build tasks for development:

*   `make install`: Installs in `release` build mode.
*   `make build`: Runs the build script in `release` mode.
*   `make build-wheel`: Runs uv build with a wheel format in `release` mode.
*   `make clean`: Deletes all build results.
*   `make docs`: Builds the documentation HTML.
*   `make ruff`: Runs ruff over all files using the `pyproject.toml` config (with autofix).
*   `make pytest`: Runs all tests with `pytest`.
*   `make test-performance`: Runs performance tests with [codspeed](https://codspeed.io).

> [!TIP]
>
> Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

## Examples

See the [examples](/nautilus_trader/examples/) directory for Python and Cython examples:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
*   [indicator](/nautilus_trader/indicators/) examples written in Cython.
*   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
*   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Pre-built Docker containers are available with the latest release and nightly builds:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

To launch the backtest example container:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for detailed development information.

### Testing with Rust

Run Rust tests with `make cargo-test`, which uses cargo-nextest.

## Contributing

Contributions are welcome!  See the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file and [open-source scope](/ROADMAP.md#open-source-scope) for guidelines.  Pull requests should target the `develop` branch.

## Community

Join our [Discord](https://discord.gg/NautilusTrader) for community support and announcements.

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens. Any claims or communications suggesting otherwise are unauthorized and false.
>
> All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
> or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
> If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

Licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).  Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements:

*   **SEO Optimization:** Focused on keywords like "algorithmic trading," "open source," "Python," "Rust," "backtesting," "high performance," and "AI."
*   **Clear Structure:** Uses headings and subheadings for better readability.
*   **Concise Summary:**  Starts with a one-sentence hook and a brief overview.
*   **Key Features:** Uses bullet points to highlight the main functionalities.
*   **Call to Action:**  Links to the original repository for easy access.
*   **More Context:** Expands on why users would choose NautilusTrader.
*   **Integration Table:** Cleanly presents supported integrations.
*   **Installation:**  Simplified installation steps with a clearer structure.
*   **Developer Focus:** Included information for developers.
*   **Community:**  Encourages engagement.
*   **License Information:**  Includes essential details.
*   **Branding:** Maintains the original logo and branding.
*   **Warnings and Notes:** Includes important info with emphasis using `[!NOTE]` and `[!WARNING]` blocks.
*   **Consolidated Instructions:** Combined installation steps.

The improved README is more informative, user-friendly, and SEO-optimized.