# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader empowers quantitative traders with a robust, high-performance platform for backtesting and deploying algorithmic trading strategies.** ([Original Repo](https://github.com/nautechsystems/nautilus_trader))

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

*   **Docs**: <https://nautilustrader.io/docs/>
*   **Website**: <https://nautilustrader.io>
*   **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features of NautilusTrader

*   **Blazing Fast Performance:** Built with Rust for high-speed execution and efficient memory usage.
*   **Reliable & Safe:** Leveraging Rust's type- and thread-safety for robust trading systems.
*   **Cross-Platform Compatibility:** Runs seamlessly on Linux, macOS, and Windows.
*   **Modular and Extensible:** Integrate any REST API or WebSocket feed with modular adapters.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, and more, alongside contingency orders.
*   **Customizable Components:** Build or customize systems with user-defined components, a message bus, and cache.
*   **Comprehensive Backtesting:** Utilize historical data with nanosecond resolution for thorough strategy testing.
*   **Seamless Live Deployment:** Deploy strategies live without code changes, mirroring backtesting results.
*   **Multi-Venue Support:** Enables market-making and arbitrage strategies across multiple venues.
*   **AI-First Design:** Designed for training AI trading agents through a fast backtesting engine.

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png" alt="Nautilus Art">

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral.
> The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

*   **High-Performance Python:** Achieve superior performance with core components written in Rust, combined with a Python-native environment.
*   **Backtesting & Live Parity:**  Ensure consistent strategy execution between backtesting and live trading environments.
*   **Reduced Operational Risk:** Benefit from enhanced risk management features, logical accuracy, and type safety.
*   **Highly Extensible:** Extend and customize with a message bus, custom components, data adapters, and more.

NautilusTrader addresses the challenge of maintaining parity between Python-based strategy research and the production trading environment.  It streamlines the transition from backtesting to live deployment, saving time and ensuring accuracy.

## Technology Stack

NautilusTrader combines the strengths of Python and Rust to deliver a powerful and reliable trading platform.

### Python

Python provides a clean, easy-to-use syntax and is the *lingua franca* of data science and AI. NautilusTrader utilizes Python for its ease of use and extensive ecosystem of libraries.

### Rust

Rust is a multi-paradigm programming language focused on performance, safety, and concurrency. NautilusTrader uses Rust for its core performance-critical components, ensuring speed and reliability.

## Integrations

NautilusTrader's modular design supports easy integration with various trading venues and data providers through adapters.

See [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details:

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

*   **ID**: The default client ID for the integrations adapter clients.
*   **Type**: The type of integration (often the venue type).

### Status

*   `building`: Under construction and likely not in a usable state.
*   `beta`: Completed to a minimally working state and in a beta testing phase.
*   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

## Versioning and Releases

NautilusTrader is under active development, with bi-weekly releases planned. Breaking changes may occur, but are documented in the release notes.

### Branches

*   `master`: Reflects the latest released version.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader uses high-precision and standard-precision modes for value types.

*   **High-precision**: 128-bit integers with up to 16 decimals of precision, and a larger value range.
*   **Standard-precision**: 64-bit integers with up to 9 decimals of precision, and a smaller value range.

> [!NOTE]
>
> By default, the official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS.
> On Windows, only standard-precision (64-bit) is available due to the lack of native 128-bit integer support.
> For the Rust crates, the default is standard-precision unless you explicitly enable the `high-precision` feature flag.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

It is recommended to install `nautilus_trader` inside a virtual environment.

**Install with Pre-built Wheel**

```bash
pip install -U nautilus_trader
```

**Install with Nautech Systems Package Index**

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

**Development Wheels**

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

## From Source

Follow the steps in the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) to install from source. Requires [rustup](https://rustup.rs/) and [clang](https://clang.llvm.org/).

## Redis

Redis is optional and only required if used as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

The Makefile automates installation and build tasks. Use `make help` to view available targets.

## Examples

Explore examples of indicators, strategies, and backtesting:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) (Python)
*   [indicator](/nautilus_trader/indicators/) (Cython)
*   [strategy](/nautilus_trader/examples/strategies/) (Python)
*   [backtest](/examples/backtest/) (BacktestEngine)

## Docker

Pre-built Docker images are available on Docker Hub.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

## Development

Refer to the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development best practices.

## Contributing

Contributions are welcome! Please review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file before contributing.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) to connect with other users and developers.

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens. Any claims or communications suggesting otherwise are unauthorized and false.
>
> All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
> or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
> If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

Licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). Contributions require a Contributor License Agreement (CLA).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">