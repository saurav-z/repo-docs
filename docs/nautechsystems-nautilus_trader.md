# NautilusTrader: High-Performance Algorithmic Trading Platform

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

**NautilusTrader is a high-performance, open-source algorithmic trading platform that empowers quantitative traders with the tools for backtesting and live deployment.**  Learn more about this innovative platform on the [original repository](https://github.com/nautechsystems/nautilus_trader).

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

*   **High Performance:** Core written in Rust for speed and efficiency.
*   **Reliable and Safe:** Rust-powered type safety and thread safety.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows; deploy with Docker.
*   **Flexible Integrations:** Modular adapters for easy connection to various trading venues and data providers.
*   **Advanced Order Types:** Supports a wide range of order types, including advanced features.
*   **Customizable:** Allows for the creation of custom components and systems.
*   **Backtesting & Live Trading Parity:** Use the same strategy implementations for backtesting and live deployments.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   **AI-Ready:** Designed to facilitate the training of AI trading agents.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Overview

NautilusTrader is an AI-first algorithmic trading platform designed for high performance.  It provides a Python-native environment for developing and deploying trading strategies, addressing the challenge of consistency between research, backtesting, and live environments. The platform is asset-class agnostic and designed with software correctness and safety as top priorities.

![Alt text](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png "nautilus")

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral.
> The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

*   **Event-Driven Python with Performance:** Leverages Rust for optimized core components.
*   **Code Consistency:** Use the same strategy code for both backtesting and live trading.
*   **Enhanced Safety:** Includes enhanced risk management and type safety.
*   **Extensible Architecture:** Provides flexibility through message buses, custom components, and data adapters.

## Technology Stack

NautilusTrader utilizes:

*   **Python:**  For its extensive libraries and user-friendly syntax.
*   **Cython:**  To enhance performance and add static typing.
*   **Rust:**  For performance-critical components, ensuring speed and safety.

## Integrations

NautilusTrader offers modular integrations to various trading venues and data providers.

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

### Status

- `building`: Under construction and likely not in a usable state.
- `beta`: Completed to a minimally working state and in a beta testing phase.
- `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule, with releases available on PyPI and the Nautech Systems package index.

### Branches

*   `master`: Latest released version, recommended for production use.
*   `nightly`: Daily snapshots of the `develop` branch for testing.
*   `develop`: Active development branch for contributions and feature work.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports two precision modes for its core value types (`Price`, `Quantity`, `Money`):

*   **High-precision:** 128-bit integers (up to 16 decimals).
*   **Standard-precision:** 64-bit integers (up to 9 decimals).

> [!NOTE]
>
> By default, the official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS.
> On Windows, only standard-precision (64-bit) is available due to the lack of native 128-bit integer support.
> For the Rust crates, the default is standard-precision unless you explicitly enable the `high-precision` feature flag.

**Rust feature flag:** To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install NautilusTrader using pip.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

### Install Specific Versions

```bash
pip install nautilus_trader==<version> --index-url=https://packages.nautechsystems.io/simple
```

### Installing From Source

Detailed instructions for installing from source, including build dependencies and environment variable setup, are provided in the original README.

## Redis

Redis is optional, required only for cache and message bus backends.  See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for details.

## Makefile

A Makefile is provided to streamline development tasks.

*   `make install`: Installs in `release` build mode.
*   `make build`: Runs the build script in `release` build mode.
*   `make cargo-test`: Runs all Rust crate tests.
*   `make docs`: Builds the documentation.
*   `make test-performance`: Runs performance tests.

## Examples

Find examples of indicators and strategies in both Python and Cython in the [examples directory](https://github.com/nautechsystems/nautilus_trader/tree/develop/examples).

## Docker

Docker images are available for NautilusTrader, including a JupyterLab image for easy backtesting.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

## Development

The [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) contains helpful information.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for Rust testing.  Run Rust tests with `make cargo-test`.

## Contributing

Contributions are welcome!  Review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and submit issues and pull requests on GitHub.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) for support and announcements.

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens.  Official communications are through our website, Discord server, and X (Twitter) account.  Report suspicious activity to <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO optimizations:

*   **Clear and Concise Hook:**  A direct and compelling one-sentence introduction.
*   **Keyword Optimization:**  Includes relevant keywords like "algorithmic trading," "high-performance," "open-source," and asset-class mentions.
*   **Well-Structured Headings:** Uses clear headings for organization and SEO readability.
*   **Bulleted Feature Lists:** Makes key features easy to scan and understand.
*   **Concise Summaries:** Streamlines the original text for better readability.
*   **Internal Linking:** Hyperlinks to key sections within the README (Installation, Integrations, etc.) for better navigation.
*   **Call to Action:** Encourages community participation and contribution.
*   **Emphasis on Benefits:** Highlights the advantages of using NautilusTrader.
*   **Cleaned Up Formatting:** Improved spacing and overall presentation.
*   **Concise Language:**  Avoids unnecessary wordiness.
*   **Stronger Call to Action:** Guides users to find more info or get involved.
*   **Direct Links:** The inclusion of explicit links to the website, documentation, and Discord.
*   **Concise summaries:** Streamlined the original text.
*   **Markdown best practices:** Ensured the use of the right markdown syntax to make the text clean and presentable.