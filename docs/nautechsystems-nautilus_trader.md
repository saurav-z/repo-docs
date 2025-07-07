# <img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="Nautilus Trader Logo">

**Nautilus Trader: The High-Performance, Open-Source Algorithmic Trading Platform**

[**Explore the Nautilus Trader Repository**](https://github.com/nautechsystems/nautilus_trader)

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

## Introduction

Nautilus Trader is an open-source, high-performance algorithmic trading platform, empowering quantitative traders with backtesting and live deployment capabilities, all within a Python-native environment.  Built for speed, reliability, and flexibility, Nautilus Trader enables traders to efficiently develop, backtest, and deploy automated trading strategies.

### Key Features

*   **High Performance:**  Core components written in Rust with asynchronous networking using [tokio](https://crates.io/crates/tokio) for maximum speed.
*   **Reliable & Safe:**  Leverages Rust's type and thread safety features, with optional Redis-backed state persistence for robust operation.
*   **Cross-Platform:**  OS independent, supporting Linux, macOS, and Windows, with Docker deployment for easy portability.
*   **Modular & Extensible:**  Integrates with any REST API or WebSocket feed through modular adapters, enabling connectivity with various trading venues and data providers.
*   **Advanced Order Types:** Supports `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`, advanced order types and conditional triggers. Execution instructions `post-only`, `reduce-only`, and icebergs. Contingency orders including `OCO`, `OUO`, `OTO`.
*   **Customization:**  Allows for user-defined custom components and system assembly using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Comprehensive Backtesting:**  Backtest strategies with multiple venues, instruments, and strategies simultaneously, using historical data with nanosecond resolution.
*   **Seamless Live Deployment:**  Utilizes identical strategy implementations for both backtesting and live trading environments.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies with multi-venue capabilities.
*   **AI-Driven Trading:**  Backtest engine is fast enough to train AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

NautilusTrader offers a superior solution for algorithmic trading by combining high performance with a robust Python-native environment, offering significant advantages for quantitative traders:

*   **High-Performance Python Execution:** Leverages native binary core components for optimized performance.
*   **Code Parity Between Backtesting and Live Trading:** Enables identical strategy code across environments, eliminating the need for reimplementation.
*   **Enhanced Operational Risk Management:** Features enhanced risk management functionality, with logical accuracy and type safety for increased reliability.
*   **Unrivaled Extensibility:** Supports customization through its message bus, custom components, adapters, and custom data inputs.

## Technology Stack

NautilusTrader combines the strengths of Python and Rust to offer a powerful and efficient platform:

### Why Python?

Python is a powerful, flexible, and widely used language, and has become the standard for data science, machine learning, and artificial intelligence.

### Why Rust?

Rust is chosen for its performance and safety, offering blazingly fast execution and memory efficiency.  Rust's rich type system and ownership model guarantees memory-safety and thread-safety deterministically — eliminating many classes of bugs at compile-time.

The project increasingly utilizes Rust for core performance-critical components. Python bindings are implemented via Cython and [PyO3](https://pyo3.rs)—no Rust toolchain is required at install time.

This project makes the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html):

> “The intent of this project is to be free of soundness bugs.
> The developers will do their best to avoid them, and welcome help in analyzing and fixing them.”

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader utilizes *adapters* for connectivity, translating raw APIs into a unified interface for various trading venues and data providers.

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

-   **ID**: The default client ID for the integrations adapter clients.
-   **Type**: The type of integration (often the venue type).

### Status Definitions

*   `building`: Under construction and likely not in a usable state.
*   `beta`: Completed to a minimally working state and in a beta testing phase.
*   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule, aiming to maintain a stable API for the 2.x version.

### Branches

*   `master`: Represents the latest released version.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: The active branch for feature development and contributions.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports two precision modes for its core value types:

*   **High-precision**: 128-bit integers with up to 16 decimals of precision.
*   **Standard-precision**: 64-bit integers with up to 9 decimals of precision.

> [!NOTE]
>
> High-precision mode is enabled by default on Linux and macOS, while standard-precision is used on Windows.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

NautilusTrader can be installed from PyPI or built from source.

### Prerequisites

*   Latest Supported Python Version
*   Virtual Environment Recommended

### Installation Options

1.  **From PyPI:**
    ```bash
    pip install -U nautilus_trader
    ```

2.  **From Nautech Systems Package Index:**
    ```bash
    pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
    ```

    Pre-release versions (including development wheels) are also available using `--pre`:

    ```bash
    pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
    ```

3.  **From Source:**
    Requires [rustup](https://rustup.rs/) and [clang](https://clang.llvm.org/) installed, followed by:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

    Follow the instructions in the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for additional details.

## Redis

Redis is an optional requirement if you configure it as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus). See the **Redis** section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for installation details.

## Makefile

A `Makefile` streamlines common development tasks:

*   `make install`: Install with all dependencies.
*   `make install-debug`: Install with debug build mode.
*   `make build`: Build the project.
*   `make test`: Run all tests.
*   `make docs`: Build the documentation.
*   `make help`: View all available targets.

## Examples

Examples of indicators and strategies are available in Python and Cython, demonstrating performance and best practices:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py)
*   [indicator](/nautilus_trader/indicators/)
*   [strategy](/nautilus_trader/examples/strategies/)
*   [backtest](/examples/backtest/)

## Docker

Pre-built Docker containers are available, with the latest and nightly versions:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

To launch the backtest example:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Open your browser to `http://127.0.0.1:8888/lab`.

## Development

Refer to the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for guidance.

### Testing

Run Rust tests with `make cargo-test`.

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss enhancements or bug fixes, and review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file. Pull requests should target the `develop` branch.

## Community

Join the [Discord](https://discord.gg/NautilusTrader) community.

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens. Any claims or communications suggesting otherwise are unauthorized and false.
>
> All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
> or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
> If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).  Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology
company specializing in the development of high-performance trading systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">