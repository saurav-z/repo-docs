# Nautilus Trader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="Nautilus Trader Logo">

**Nautilus Trader empowers quantitative traders with a production-grade, Python-native platform for backtesting and live deployment of algorithmic trading strategies.**

[Go to the original repository](https://github.com/nautechsystems/nautilus_trader)

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

*   **High-Performance:** Built with Rust for speed and efficiency.
*   **Python-Native:** Develop and deploy strategies in a familiar Python environment.
*   **Backtesting & Live Trading:** Use the same code for both backtesting and live deployment.
*   **Modular and Flexible:** Integrate with various data feeds and trading venues through modular adapters.
*   **Advanced Order Types:** Supports a wide range of order types and conditional triggers.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   **AI Training:** Backtest engine designed for training AI trading agents.

## Introduction

NautilusTrader is an open-source, production-grade algorithmic trading platform designed for quantitative traders. It provides a powerful and robust environment for developing, backtesting, and deploying automated trading strategies. Built with a core written in Rust, the platform offers high performance, reliability, and type safety. NautilusTrader addresses the "parity challenge" by allowing developers to seamlessly transition strategies from research and backtesting to live trading, all within a Python-native environment.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why NautilusTrader?

*   **Exceptional Performance:** Leveraging Rust for core components and asynchronous networking.
*   **Code Reusability:** Identical strategy code for backtesting and live trading.
*   **Enhanced Risk Management:** Advanced risk management functionality, accuracy, and type safety.
*   **Extensibility:** Customize your trading systems with message bus, custom components, actors, data and adapters.

NautilusTrader overcomes the limitations of traditional trading platforms by integrating high-performance Rust components to Python,
allowing for both event-driven backtesting and live trading without reimplementing the strategy logic.

## Technology

NautilusTrader leverages the power of both Python and Rust:

### Python

Python is the *de facto lingua franca* of data science, machine learning, and artificial intelligence. Python is the language of choice for trading strategy research and backtesting.

### Rust

Rust provides performance, safety and memory efficiency, and is used for core performance-critical components.

This project makes the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html).

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader's modular design allows for easy integration with various trading venues and data providers.

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

### Status

- `building`: Under construction and likely not in a usable state.
- `beta`: Completed to a minimally working state and in a beta testing phase.
- `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule.

*   `master`: Reflects the source code for the latest released version.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports two precision modes for its core value types (`Price`, `Quantity`, `Money`):

-   **High-precision**: 128-bit integers with up to 16 decimals of precision, and a larger value range.
-   **Standard-precision**: 64-bit integers with up to 9 decimals of precision, and a smaller value range.

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

Install NautilusTrader easily using `pip`.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

Install with the Nautech Systems package index for access to pre-release versions.

#### Stable wheels

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development wheels

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

### From Source

Install from source by building the project.  Requires Rust, clang, and uv.

See [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for instructions.

## Redis

Redis is optional and required if configured as the backend for a [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

Use the `Makefile` to automate installation and build tasks.

-   `make install`: Installs with dependencies and extras.
-   `make build`: Runs the build script.
-   `make test`: Runs tests.
-   `make docs`: Builds documentation.
-   `make help`: Displays all available make targets.

## Examples

Find examples of indicators, strategies, and backtests in the `examples` directory.

## Docker

Use Docker containers for easy deployment.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

## Development

Develop with ease with our helpful [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html).

### Testing with Rust

Run tests with cargo-nextest:

```bash
make cargo-test
```

## Contributing

Contribute to NautilusTrader by opening an [issue](https://github.com/nautechsystems/nautilus_trader/issues) and following the guidelines in [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md).

> [!NOTE]
>
> Pull requests should target the `develop` branch (the default branch). This is where new features and improvements are integrated before release.

## Community

Join our community on [Discord](https://discord.gg/NautilusTrader) to connect with other users and contributors.

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
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">