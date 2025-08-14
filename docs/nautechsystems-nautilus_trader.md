# NautilusTrader: High-Performance Algorithmic Trading Platform

[![NautilusTrader Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png "NautilusTrader")](https://github.com/nautechsystems/nautilus_trader)

**NautilusTrader is an open-source, AI-first trading platform built for speed and reliability, enabling traders to backtest and deploy strategies with ease.**

[View the original repository on GitHub](https://github.com/nautechsystems/nautilus_trader)

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
| `Linux (x86_64)`   | 1.89.0 | 3.11-3.13  |
| `Linux (ARM64)`    | 1.89.0 | 3.11-3.13  |
| `macOS (ARM64)`    | 1.89.0 | 3.11-3.13  |
| `Windows (x86_64)` | 1.89.0 | 3.11-3.13* |

\* Windows builds are currently pinned to CPython 3.13.2, see [installation guide](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/getting_started/installation.md).

- **Docs**: <https://nautilustrader.io/docs/>
- **Website**: <https://nautilustrader.io>
- **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Core components written in Rust for speed and efficiency, utilizing asynchronous networking with tokio.
*   **Reliability & Safety:** Leverage Rust's type and thread safety, with optional Redis integration for persistent state management.
*   **Cross-Platform Compatibility:** Supports Linux, macOS, and Windows, with Docker deployment for portability.
*   **Modular Design:** Easily integrate with any REST API or WebSocket feed through modular adapters.
*   **Advanced Order Types & Controls:** Includes IOC, FOK, GTC, GTD, DAY, AT_THE_OPEN, AT_THE_CLOSE, post-only, reduce-only, and iceberg orders.
*   **Customizable Architecture:** Build from the ground up using the cache and message bus.
*   **Backtesting Capabilities:** Run backtests with multiple venues, instruments, and strategies, using high-resolution historical data.
*   **Seamless Live Deployment:** Identical strategy implementations for backtesting and live trading environments.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   **AI Training Ready:** Backtesting engine designed for training AI trading agents (RL/ES).

![Nautilus Trader Architecture](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png "NautilusTrader Architecture")

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral.
> The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

*   **Event-Driven Python Optimized:** Native binary core components written in Rust & Cython.
*   **Code Parity:** Write your strategies once and deploy them across backtesting and live trading environments.
*   **Enhanced Security:** Advanced risk management functionality, type safety, and logical accuracy.
*   **Extensible Platform:** Fully customizable message bus, custom components, custom adapters, and data integrations.

## Technology Stack: Rust, Python, and Cython

NautilusTrader leverages a powerful combination of technologies to provide a robust and high-performance platform. The core components are written in Rust and Cython.

### Why Python?

Python is the *de facto lingua franca* of data science, machine learning, and artificial intelligence, making it an ideal choice for trading strategy development.

### Why Rust?

Rust provides the performance and safety needed for mission-critical trading systems.  Rust guarantees memory-safety and thread-safety at compile-time.

This project makes the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html):

> “The intent of this project is to be free of soundness bugs.
> The developers will do their best to avoid them, and welcome help in analyzing and fixing them.”

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader integrates with various trading venues and data providers via modular adapters, translating APIs into a unified interface:

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

### Status Definitions

*   `building`: Under active development.
*   `beta`: Minimally working and in beta testing.
*   `stable`: Stabilized and tested integration.

For detailed information, see the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation.

## Versioning and Releases

**NautilusTrader is actively developed.** Some features may be incomplete, and breaking changes can occur between releases. We aim for a **bi-weekly release schedule**.

### Branches

*   `master`: Latest released version; recommended for production.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch for contributors.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports two precision modes for core value types (`Price`, `Quantity`, `Money`):

*   **High-precision:** 128-bit integers, up to 16 decimal places.
*   **Standard-precision:** 64-bit integers, up to 9 decimal places.

> [!NOTE]
>
> Official Python wheels ship in high-precision mode (128-bit) on Linux and macOS. On Windows, only standard-precision (64-bit) is available.  Rust crates default to standard-precision unless `high-precision` feature flag is enabled.

**Rust Feature Flag**: Enable high-precision mode in Rust:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install `nautilus_trader` inside a virtual environment.

**Supported Installation Methods:**

1.  Pre-built binary wheel from PyPI or the Nautech Systems package index.
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

### From the Nautech Systems Package Index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

### Development Wheels

Install pre-release (development) wheels using:

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

### View Available Versions

```bash
curl -s https://packages.nautechsystems.io/simple/nautilus-trader/index.html | grep -oP '(?<=<a href=")[^"]+(?=")' | awk -F'#' '{print $1}' | sort
```

### Installation from Source

1.  Install [rustup](https://rustup.rs/).
2.  Enable `cargo` in the current shell.
3.  Install [clang](https://clang.llvm.org/).
4.  Install [uv](https://docs.astral.sh/uv).
5.  Clone the repository and install:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

6.  Set environment variables for PyO3 compilation.

## Redis (Optional)

Redis is only required if you use it as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

The `Makefile` simplifies installation and development tasks. Run `make help` to see available targets.

## Examples

Find Python and Cython examples in the `/nautilus_trader/examples` directory, including:

*   Indicators
*   Strategies
*   Backtesting

## Docker

Ready-to-use Docker containers are available on ghcr.io/nautechsystems.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

Example: Launch the backtest example:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development workflows.

> [!TIP]
>
> Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for reliable Rust testing.

```bash
cargo install cargo-nextest
```

Run tests with `make cargo-test`.

## Contributing

Contributions are welcome!  Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss enhancements and bug fixes.  Follow the guidelines in [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and sign a Contributor License Agreement (CLA).

> [!NOTE]
>
> Pull requests should target the `develop` branch (the default branch). This is where new features and improvements are integrated before release.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) for support and announcements.

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).
Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems. For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![Nautech Systems Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "Nautech Systems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO considerations:

*   **Clear Headline:**  Uses the project name followed by a clear description.
*   **SEO-Friendly Description:**  Uses keywords like "algorithmic trading," "trading platform," "backtesting," and "deploy strategies."
*   **Feature Highlighting:**  Uses a bulleted list of key features, making it easy to scan and understand the platform's capabilities.  Keywords like "Rust," "Python," "backtesting," "live deployment," and "AI" are used.
*   **Why NautilusTrader Section:** Adds a section that expands on the key selling points.
*   **Technology Stack Section:**  Clearly explains the technologies used (Rust, Python, Cython) and their benefits.
*   **Clear Installation Instructions:** Simplified and structured installation instructions.
*   **Contributing and Community Sections:**  Highlights how users can contribute and access community support.
*   **Docker Instructions:**  Provides clear instructions on how to use Docker containers.
*   **Use of Headings:**  Well-structured headings to organize the content and improve readability.
*   **Image Alt Text:**  Includes alt text for images, which is important for SEO and accessibility.
*   **Warning Boxes:** Uses markdown's callout syntax for warnings.
*   **Link to Original Repo:** Includes a direct link back to the GitHub repository at the beginning.
*   **Concise and Focused:** Removes unnecessary details and focuses on the core value proposition.
*   **Call to Action:** Encourages community participation and contribution.
*   **Updated Logos**: Removed an extraneous logo at the bottom.
*   **Roadmap Mention**: Included mention of the roadmap to provide context for the API stability.