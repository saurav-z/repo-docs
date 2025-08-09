# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader empowers quantitative traders with a production-grade, AI-first platform for backtesting and live deployment of automated trading strategies.**

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

*   **High Performance:** Core written in Rust for speed and efficiency, with asynchronous networking using Tokio.
*   **Reliability & Safety:** Built with Rust's memory and thread safety features, with optional Redis-backed state persistence.
*   **Cross-Platform Compatibility:** Runs on Linux, macOS, and Windows; deploy with Docker.
*   **Flexible Integration:** Modular adapters support easy integration with any REST API or WebSocket feed.
*   **Advanced Order Types:** Includes IOC, FOK, GTC, GTD, DAY, AT_THE_OPEN, AT_THE_CLOSE, and conditional triggers, along with execution instructions like post-only, reduce-only, icebergs, and contingency orders (OCO, OUO, OTO).
*   **Customization:**  Allows the addition of user-defined components, and the assembly of complete trading systems, leveraging the cache and message bus.
*   **Backtesting Capabilities:** Backtest strategies with historical data, supporting multiple venues, instruments, and strategies simultaneously, with nanosecond resolution.
*   **Seamless Deployment:** Use the exact same strategy implementations for both backtesting and live deployments.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies across multiple venues.
*   **AI-Driven Development:** Backtest engine is fast enough for training AI trading agents.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

NautilusTrader offers a compelling solution for quantitative traders by providing:

*   **High-Performance Event-Driven Python:** Leverage the power of a Python-native environment with core components written in Rust.
*   **Code Parity:** Ensure consistency between backtesting and live trading environments with identical strategy code.
*   **Reduced Operational Risk:** Benefit from robust risk management features, logical accuracy, and enhanced type safety.
*   **Extensibility:** Customize the platform with a message bus, custom components and actors, custom data, and custom adapters.

NautilusTrader overcomes the limitations of traditional trading strategy development, allowing for high-performance trading systems development within a consistent Python-native environment.

## Technology Stack

### Python

Python is a versatile programming language, widely used in data science, machine learning, and artificial intelligence. NautilusTrader leverages Python's rich ecosystem for strategy development and backtesting.

### Rust

[Rust](https://www.rust-lang.org/) is a systems programming language renowned for performance, safety, and concurrency. NautilusTrader utilizes Rust for core performance-critical components, ensuring speed and reliability.

### Cython

Cython introduces static typing into Python's ecosystem, enhancing performance and type safety for large-scale systems.

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader's modular design enables seamless integration with various trading venues and data providers via adapters.

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
| [OKX](https://okx.com)                                                       | `OKX`                 | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/beta-yellow)     | [Guide](docs/integrations/okx.md)           |
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

### Status Definitions

*   `building`: Under construction, may not be in a usable state.
*   `beta`: Completed to a minimally working state and in a beta testing phase.
*   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for details.

## Installation

Choose from two supported methods to install NautilusTrader:

1.  **Pre-built binary wheel:** Install from PyPI or the Nautech Systems package index.
2.  **Build from source:** Install from source code after cloning the repository.

We highly recommend using the [uv](https://docs.astral.sh/uv) package manager.

### Installation Instructions

#### From PyPI

```bash
pip install -U nautilus_trader
```

#### From Nautech Systems Package Index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Available Versions

You can view all available versions of `nautilus_trader` on the [package index](https://packages.nautechsystems.io/simple/nautilus-trader/index.html).

#### From Source

Follow the source installation instructions to install from the repository.

### Additional Installation Notes

*   Ensure you have the latest supported Python version.
*   Always install in a virtual environment for dependency isolation.
*   See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for details, including Redis setup and more options.

## Using Redis (Optional)

Redis is optional and is only required if you configure NautilusTrader to use the cache or message bus features.

## Makefile

A `Makefile` simplifies the installation, build, and development processes.  Run `make help` for detailed instructions.

## Examples

Explore practical examples of indicators, strategies, and backtesting setups to get started.

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py)
*   [indicator](/nautilus_trader/indicators/)
*   [strategy](/nautilus_trader/examples/strategies/)
*   [backtest](/examples/backtest/)

## Docker

Use pre-built Docker images for quick setup and deployment.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Open your browser at: `http://127.0.0.1:8888/lab`

## Development

Review the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for best practices and workflows.  For optimal development, run `make build-debug` to compile after code changes.

### Testing with Rust

The standard Rust test runner, [cargo-nextest](https://nexte.st) is used for testing.

```bash
cargo install cargo-nextest
make cargo-test
```

## Contributing

Contributions are welcome! Review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) for support and updates.

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens. Any claims or communications suggesting otherwise are unauthorized and false.
>
> All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
> or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
> If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems.

Visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO optimizations:

*   **Clear Hook:** Starts with a compelling one-sentence description to capture the reader's attention and highlight the main benefit.
*   **Keyword Rich Headings:** Uses keywords like "Algorithmic Trading," "High Performance," and "Python" in headings.
*   **Bulleted Key Features:** Clearly lists the benefits of the platform, making it easy to scan.
*   **Well-Organized Structure:** Uses headings, subheadings, and lists for readability.
*   **SEO-Friendly Language:** Uses terms that traders would search for.
*   **Links to Important Resources:**  Includes links to the original repository, documentation, website, and community resources.
*   **Integration of Images:** Keeps the images to improve readability
*   **Concise and Focused:** Avoids unnecessary jargon and technical details, focusing on the value proposition.
*   **Call to Action (Implicit):** Encourages users to explore the project and its features.
*   **Corrected minor inconsistencies**
*   **Updated Status Badges**
*   **More detailed integration descriptions**
*   **Added installation method explanations**
*   **Minor Formatting fixes and clarifications**