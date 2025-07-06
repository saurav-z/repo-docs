# <img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="Nautilus Trader Logo">

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

## NautilusTrader: The High-Performance Algorithmic Trading Platform

**NautilusTrader provides a powerful and reliable platform for building, backtesting, and deploying AI-powered trading strategies with Python.**  [Explore the original repository](https://github.com/nautechsystems/nautilus_trader).

### Key Features:

*   **High Performance:** Built with Rust for speed and efficiency.
*   **Reliable & Safe:**  Leverages Rust's memory and thread safety features.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows, with Docker support.
*   **Modular & Flexible:** Integrates with various exchanges and data providers via adapters.
*   **Advanced Order Types:** Supports a wide range of order types and triggers.
*   **Customizable:** Allows for custom components and integrations.
*   **Comprehensive Backtesting:** Backtest strategies with nanosecond resolution.
*   **Seamless Live Deployment:** Use the same strategy code for backtesting and live trading.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   **AI-First Design**: Designed for AI training (RL/ES) and trading agents.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

*   **Blazing Fast Python:** Optimized for high-performance event-driven execution.
*   **Code Parity:**  Maintain consistent strategy code between backtesting and live trading.
*   **Reduced Risk:** Enhances risk management, logical accuracy, and type safety.
*   **Extensible Architecture:** Supports a wide range of customizations through a message bus, custom components, actors, and custom adapters.

NautilusTrader offers a significant advantage by allowing you to develop trading strategies in Python while benefiting from the performance of Rust. This streamlined approach eliminates the need to rewrite strategies in lower-level languages for live trading, saving time and reducing complexity.

## Technology Stack

*   **Rust:** Used for core performance-critical components, offering speed and memory safety.
*   **Python:** Utilized for strategy development within a Python-native environment.
*   **Cython:** Used to create Python bindings to Rust components to create high-performance Python extensions.

### Benefits of Rust:

*   **Speed and Efficiency:**  Rust's performance is comparable to C and C++.
*   **Memory Safety:** Rust's ownership model ensures memory safety and thread safety.
*   **No Garbage Collector:** Provides deterministic behavior.

### Benefits of Python:

*   **Ease of Use:**  Python has a clean syntax and is the *de facto lingua franca* of data science, machine learning, and artificial intelligence.
*   **Rich Ecosystem:** Access to a vast array of libraries and tools.

## Integrations

NautilusTrader offers modular *adapters* for connecting to various trading venues and data providers. This allows you to translate their APIs into a unified interface and normalized domain model.

See [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details.

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

### Integration Status

*   `building`: Under development
*   `beta`: Functionality is minimally complete, and in beta testing phase
*   `stable`: Stable and tested

## Versioning and Releases

**NautilusTrader is under active development.** The project follows a **bi-weekly release schedule**.

### Branches

*   `master`: Latest released version (production).
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch.

> [!NOTE]
>
>  Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).

## Precision Mode

NautilusTrader supports two precision modes for core data types:

*   **High-precision:** 128-bit integers with up to 16 decimals.
*   **Standard-precision:** 64-bit integers with up to 9 decimals.

> [!NOTE]
>
> By default, the official Python wheels ship in high-precision (128-bit) mode on Linux and macOS.
> On Windows, only standard-precision (64-bit) is available.

**Rust feature flag:** To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install NautilusTrader using `pip`, or from the Nautech Systems package index.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems Package Index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

### From Source

Follow the steps in the original README for detailed instructions, which include:

1.  Install Rustup and the Rust toolchain.
2.  Set up `cargo`.
3.  Install `clang`.
4.  Install `uv`.
5.  Clone the repository with `git`.
6.  Set environment variables.

## Redis

Redis is **optional** and only needed if used as a backend for the cache or message bus.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for further details.

## Makefile

Use the `Makefile` for common development tasks, including installation, building, testing, and documentation generation. Run `make help` for a list of targets.

## Examples

Explore Python and Cython examples for indicators, strategies, and backtesting:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py)
*   [indicator](/nautilus_trader/indicators/)
*   [strategy](/nautilus_trader/examples/strategies/)
*   [backtest](/examples/backtest/)

## Docker

Pre-built Docker containers are available:

*   `nautilus_trader:latest`
*   `nautilus_trader:nightly`
*   `jupyterlab:latest`
*   `jupyterlab:nightly`

Pull and run the Docker container:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Access the JupyterLab interface in your web browser at `http://127.0.0.1:8888/lab`.

## Development

The [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) provides helpful information for contributing.

> [!TIP]
>
> Run `make build-debug` after changes to Rust or Cython code.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for Rust testing:

```bash
cargo install cargo-nextest
```

> [!TIP]
>
> Run Rust tests with `make cargo-test`.

## Contributing

Contribute by opening an [issue](https://github.com/nautechsystems/nautilus_trader/issues) and following the guidelines in [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md).

> [!NOTE]
>
> Pull requests should target the `develop` branch.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader).

> [!WARNING]
>
> NautilusTrader does not promote cryptocurrency tokens.  All official communications will only be shared through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader), or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).

## License

The source code is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO considerations:

*   **Clear, Concise Summary:** Starts with a compelling one-sentence hook and provides a strong introduction.
*   **Keyword Optimization:**  Includes relevant keywords like "algorithmic trading," "high-performance," "backtesting," "AI trading," "Python," and "Rust."
*   **Structured Content:** Uses headings, bullet points, and concise paragraphs for readability.
*   **Actionable Information:** Provides installation and usage instructions.
*   **Call to Action:** Encourages community participation.
*   **SEO Titles and Descriptions (Implied):** Uses clear titles and headings for search engine understanding.  The bullet points and descriptions are also well-written, which helps with SEO.
*   **Clean Formatting:**  Uses Markdown for clear and readable content.
*   **Improved Descriptions:**  Better descriptions of features and technologies.
*   **Concise Sections:** Streamlined information for better comprehension.
*   **Targeted Keywords:** Uses keywords throughout the entire README to optimize for relevant search queries.
*   **Clear Structure:** Breaks down the information into logical sections that users can easily navigate.
*   **Updated Badge URLs:** The badging URLs were preserved.
*   **Includes key integration section.**