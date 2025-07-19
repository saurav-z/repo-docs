# NautilusTrader: High-Performance Algorithmic Trading Platform

**NautilusTrader is an open-source, AI-first algorithmic trading platform designed for high-performance backtesting and live deployment of trading strategies.** Explore the original repository [here](https://github.com/nautechsystems/nautilus_trader).

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

*   **High Performance:** Core components written in Rust for speed and efficiency, with asynchronous networking using [tokio](https://crates.io/crates/tokio).
*   **Reliable and Safe:** Leveraging Rust's type and thread safety, optional Redis-backed state persistence.
*   **Cross-Platform:** Compatible with Linux, macOS, and Windows. Deploy using Docker.
*   **Modular Design:** Integrates with any REST API or WebSocket feed via modular adapters.
*   **Advanced Order Types:** Includes `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`, conditional triggers, execution instructions (`post-only`, `reduce-only`, icebergs), and contingency orders (`OCO`, `OUO`, `OTO`).
*   **Extensible and Customizable:** Create custom components or assemble entire systems from scratch using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Backtesting Capabilities:** Run backtests with multiple venues, instruments, and strategies concurrently using historical data with nanosecond resolution.
*   **Production Ready:** Use identical strategy implementations for both backtesting and live deployments, minimizing code changes.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies across multiple venues.
*   **AI Training:** The backtest engine is optimized for training AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

NautilusTrader streamlines the development, testing, and deployment of algorithmic trading strategies, bridging the gap between Python-based research and production environments.

*   **Optimized for Python:** Leverage high-performance event-driven Python environment using native binary core components.
*   **Backtesting to Live Parity:** Use the exact same strategy code for both backtesting and live trading.
*   **Reduced Risk:** Benefit from enhanced risk management features, logical accuracy, and type safety.
*   **Highly Extensible:** Expand functionality with the message bus, custom components and actors, custom data, and custom adapters.

## Why Use Python and Rust?

NautilusTrader combines the strengths of both Python and Rust:

*   **Python:** Python's clear syntax and extensive libraries, particularly in data science, machine learning, and AI, make it ideal for strategy research and development.
*   **Rust:** Rust provides the performance and safety needed for low-latency, mission-critical trading systems, ensuring reliability and efficiency through its memory-safe design and speed.

## Integrations

NautilusTrader offers a modular design that allows it to seamlessly integrate with trading venues and data providers through *adapters*, translating their raw APIs into a unified interface and a normalized domain model.

The following integrations are currently supported (see [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details):

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

**NautilusTrader is under active development.** Breaking changes are possible between releases, and we aim to document these in the release notes.

*   **Release Schedule:** Aiming for bi-weekly releases, though delays can occur.
*   **Branches:**
    *   `master`: Reflects the source code for the latest released version; recommended for production use.
    *   `nightly`: Daily snapshots of the `develop` branch for early testing.
    *   `develop`: Active development branch.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x**.

## Precision Mode

NautilusTrader supports two precision modes for its core value types (`Price`, `Quantity`, `Money`).

*   **High-precision**: 128-bit integers with up to 16 decimals of precision.
*   **Standard-precision**: 64-bit integers with up to 9 decimals of precision.

> [!NOTE]
>
> By default, the official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS.
> On Windows, only standard-precision (64-bit) is available.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Follow these steps to install NautilusTrader:

1.  **Python Environment:**  Use a virtual environment (recommended) to isolate dependencies.
2.  **Installation Options:**
    *   **From PyPI:**  `pip install -U nautilus_trader`
    *   **From Nautech Systems Package Index:**
        *   Stable:  `pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple`
        *   Pre-release:  `pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple`
        *   Specific Version:  `pip install nautilus_trader==<version> --index-url=https://packages.nautechsystems.io/simple`
    *   **From Source:**
        *   Install [rustup](https://rustup.rs/) and [clang](https://clang.llvm.org/).
        *   Clone the repository.
        *   Install build dependencies using `uv sync --all-extras`.
        *   Install with `uv` (see installation guide).

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for other options and further details.

## Redis

Redis is **optional** and required only if you're using it as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus). Refer to the **Redis** section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for instructions.

## Makefile

A `Makefile` simplifies common development tasks, including:

*   `make install`:  Installs with all dependencies in `release` mode.
*   `make install-debug`:  Installs with all dependencies in `debug` mode.
*   `make build`:  Runs the build script in `release` mode.
*   `make cargo-test`:  Runs Rust crate tests.
*   `make clean`:  Deletes build results.
*   `make docs`:  Builds documentation.
*   And more.

> [!TIP]
>
> Run `make help` for a complete list of `Makefile` targets.

## Examples

Explore Python and Cython examples to understand indicator and strategy development (see the linked files for examples).

## Docker

NautilusTrader provides Docker containers to ease deployment:

*   `nautilus_trader:latest`: Latest release.
*   `nautilus_trader:nightly`: Head of the `nightly` branch.
*   `jupyterlab:latest`: JupyterLab with the latest release.
*   `jupyterlab:nightly`: JupyterLab with the head of the `nightly` branch.

Pull the images and run them (e.g., JupyterLab example):

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Open your browser at `http://127.0.0.1:8888/lab`.

> [!WARNING]
>
> NautilusTrader currently exceeds the rate limit for Jupyter notebook logging. The `log_level` is set to `ERROR` to avoid causing the notebook to hang.

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development information.

> [!TIP]
>
>  `make build-debug` will recompile after any changes to Rust or Cython code.

### Testing with Rust

NautilusTrader uses [cargo-nextest](https://nexte.st) for Rust testing:

*   Install with `cargo install cargo-nextest`.
*   Run Rust tests with `make cargo-test`.

## Contributing

Contributions are welcome! Follow the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines and open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your ideas.  Target contributions to the `develop` branch.

## Community

Join our [Discord](https://discord.gg/NautilusTrader) community for discussions and updates.

> [!WARNING]
>
> NautilusTrader does not issue or endorse any cryptocurrency tokens. All official communications are through our website, Discord, or X (Twitter) account.

## License

Licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). Contributors must sign a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology
company specializing in the development of high-performance trading systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">