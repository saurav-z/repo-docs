# NautilusTrader: High-Performance Algorithmic Trading Platform

**Unlock the power of AI-driven trading with NautilusTrader, the open-source platform built for speed, reliability, and Python-native development.**

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

[Visit the NautilusTrader GitHub Repository](https://github.com/nautechsystems/nautilus_trader)

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

NautilusTrader is a cutting-edge, open-source algorithmic trading platform designed for quantitative traders. It empowers you to backtest and deploy automated trading strategies with unmatched performance and reliability. Built with an *AI-first* approach, NautilusTrader bridges the gap between research, backtesting, and live deployment, ensuring consistency in your Python-native environment.

With its focus on software correctness and safety, the platform provides a robust environment for mission-critical trading applications. It offers asset-class agnosticism through modular adapters, supporting high-frequency trading across FX, Equities, Futures, Options, Crypto, DeFi, and Betting, and enabling seamless operations across multiple venues simultaneously.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Key Features

*   **Blazing Fast:** Core engine built in Rust with asynchronous networking using [tokio](https://crates.io/crates/tokio).
*   **Reliable & Safe:** Rust-powered type- and thread-safety, with optional Redis-backed state persistence.
*   **Cross-Platform:** OS independent, running on Linux, macOS, and Windows. Easily deploy with Docker.
*   **Modular & Flexible:** Integrate with any REST API or WebSocket feed through modular adapters.
*   **Advanced Order Types:** Supports time in force, execution instructions, and contingency orders for sophisticated trading.
*   **Highly Customizable:** Create custom components and entire systems using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Comprehensive Backtesting:** Run backtests with multiple venues, instruments, and strategies, using various data types with nanosecond resolution.
*   **Seamless Live Deployment:** Use the exact same strategy code for backtesting and live trading.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies.
*   **AI Training Ready:** Backtest engine designed for training AI trading agents (RL/ES).

![Alt text](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png "nautilus")

## Why Choose NautilusTrader?

*   **High-Performance Python with Native Core:** Benefit from a performant, event-driven core written in Rust and Cython.
*   **Backtesting and Live Parity:** Develop and deploy strategies with identical code, reducing errors and accelerating the development cycle.
*   **Reduced Operational Risk:** Leverage enhanced risk management features, logical accuracy, and type safety.
*   **Extensible and Adaptable:** Expand functionalities through the message bus, custom components and actors, custom data, and adapters.

NautilusTrader addresses the common challenge of having to rewrite trading strategies from Python (used for research/backtesting) to C++, C#, or Java for live trading environments. Its architecture enables high performance using systems programming languages to compile performant binaries, with CPython C extension modules and an environment suitable for quantitative traders.

## Why Python?

Python's clean syntax and extensive libraries have made it the *lingua franca* of data science, machine learning, and artificial intelligence.

## Why Rust?

Rust's focus on performance, safety, and concurrency (including thread safety) makes it ideal for building mission-critical, high-performance systems.

## Integrations

NautilusTrader utilizes modular adapters to connect with various trading venues and data providers, translating their APIs into a unified interface.

The following integrations are currently supported (see [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details):

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

*   `building`: Under active development and not yet usable.
*   `beta`: Minimally functional with active beta testing.
*   `stable`: Stable API, tested and ready for production use.

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule.

### Branches

-   `master`: Reflects the latest released version, recommended for production.
-   `nightly`: Daily snapshots of the `develop` branch.
-   `develop`: Active development branch.

## Precision Mode

NautilusTrader supports High-precision and Standard-precision modes for `Price`, `Quantity`, and `Money`.

*   **High-precision:** 128-bit integers with 16 decimals.
*   **Standard-precision:** 64-bit integers with 9 decimals.

## Installation

Install NautilusTrader using [pip](https://pip.pypa.io/en/stable/) or from source.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

### From Source

Follow the installation instructions in the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for detailed steps, including installing Rust, clang, and other dependencies.

## Redis

Redis is optional and only required if you use the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus). See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for more information.

## Makefile

The provided `Makefile` automates installation and build tasks:

*   `make install`: Installs with all dependency groups.
*   `make build`: Runs the build script.
*   `make test`: Runs tests.
*   `make docs`: Builds the documentation.

Run `make help` for details.

## Examples

Find indicator and strategy examples in both Python and Cython:

*   [indicator examples](/nautilus_trader/examples/indicators/ema_python.py and /nautilus_trader/indicators/)
*   [strategy examples](/nautilus_trader/examples/strategies/)
*   [backtest examples](/examples/backtest/)

## Docker

Use Docker containers for easy deployment:

*   `nautilus_trader:latest`: Latest release.
*   `nautilus_trader:nightly`: Nightly build.
*   `jupyterlab:latest`: Latest release with JupyterLab.
*   `jupyterlab:nightly`: Nightly build with JupyterLab.

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

Refer to the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development best practices.

## Testing

*   Use `cargo-nextest` for reliable Rust testing. Run Rust tests with `make cargo-test`.

## Contributing

We welcome contributions! Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your ideas, review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file, and sign the Contributor License Agreement (CLA).

> Pull requests should target the `develop` branch.

## Community

Join the [Discord](https://discord.gg/NautilusTrader) community.

## License

Licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:**  Strong title with a compelling one-sentence hook to grab attention immediately.
*   **Keyword Optimization:**  Uses relevant keywords such as "algorithmic trading," "high-performance," "open-source," "Python," "Rust," "backtesting," "live trading," "AI," and asset-class mentions.
*   **Concise Summarization:**  Avoids redundancy by focusing on key features and benefits.
*   **Improved Formatting:**  Uses headings, bulleted lists, and tables for better readability and SEO.
*   **Clear Sectioning:**  Logically organizes content for easy navigation.
*   **Emphasis on Benefits:**  Highlights *why* users should choose NautilusTrader, not just *what* it is.
*   **Strategic Links:**  Includes links to the original repository, documentation, and other resources.
*   **Updated Logos and Badges:** Keeps the badges to help show the current version and build status.
*   **Targeted Metadata (Implicit):** The use of headings and keywords implicitly helps search engines understand the content.
*   **Call to Action (Implied):** Encourages users to join the community and contribute.
*   **Consistent Tone:** Maintains a professional yet engaging tone throughout.
*   **Clear Integrations Section**: Emphasized Integrations and their status
*   **Concise Installation Section**
*   **Clear Development Section**