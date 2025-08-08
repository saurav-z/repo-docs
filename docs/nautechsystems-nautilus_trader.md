# NautilusTrader: High-Performance Algorithmic Trading Platform

**Supercharge your trading strategies with NautilusTrader, the open-source, AI-first platform engineered for speed, reliability, and Python-native development.** ([View on GitHub](https://github.com/nautechsystems/nautilus_trader))

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

NautilusTrader empowers quantitative traders with a robust, high-performance platform for developing, backtesting, and deploying algorithmic trading strategies.

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

- **Docs:** <https://nautilustrader.io/docs/>
- **Website:** <https://nautilustrader.io>
- **Support:** [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Core components written in Rust for speed and efficiency.
*   **Python-Native:** Designed for seamless integration with Python-based workflows.
*   **Backtesting & Live Deployment:** Identical strategy code for consistent results.
*   **Modular & Flexible:** Integrate with any REST API or WebSocket feed using modular adapters.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, DAY, AT\_THE\_OPEN, AT\_THE\_CLOSE, and more.
*   **Asset-Class Agnostic:** Supports FX, Equities, Futures, Options, Crypto, DeFi, and Betting.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies.
*   **AI-First:** Fast backtest engine for training AI trading agents.
*   **Reliable:** Type and thread safety, with optional Redis-backed persistence.
*   **Portable:** Runs on Linux, macOS, and Windows; Docker deployment.

## Introduction

NautilusTrader is a cutting-edge, open-source algorithmic trading platform designed for speed, reliability, and Python-native development, empowering quantitative traders to build, backtest, and deploy trading strategies with ease. The platform leverages the power of Rust and Cython to create a high-performance environment ideal for backtesting and live deployment, addressing the "parity challenge" between research and production. It's universal, asset-class-agnostic, and supports high-frequency trading across various asset classes, including FX, Equities, Futures, Options, Crypto, DeFi, and Betting.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

*   **High-Performance Event-Driven Python:** Leverages native binary core components.
*   **Seamless Parity Between Backtesting and Live Trading:** Ensures identical strategy code.
*   **Enhanced Operational Risk Management:** Improves risk management with logical accuracy and type safety.
*   **Extensible Architecture:** Supports a message bus, custom components, custom data, and custom adapters.

## Technologies: Python, Cython, and Rust

NautilusTrader combines the strengths of different languages to offer a powerful and efficient trading platform:

*   **Python:** The core language for strategy development, known for its clean syntax and rich ecosystem of libraries.
*   **Cython:** Used to introduce static typing into Python and address performance limitations in Python.
*   **Rust:** Utilized for performance-critical components, offering "blazingly fast" performance, memory efficiency, and thread safety.  NautilusTrader makes a [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html) to ensure the reliability of its core components.

> [!NOTE]
>
> **MSRV:** The Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader's modular design allows easy integration with various trading venues and data providers.

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
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule.

### Branches

*   `master`: Latest released version (for production use).
*   `nightly`: Daily snapshots of the `develop` branch for early testing.
*   `develop`: Active development branch for contributors and feature work.

> [!NOTE]
>
> A stable API is planned for version 2.x.

## Precision Mode

NautilusTrader supports two precision modes.

*   **High-precision:** 128-bit integers with up to 16 decimals of precision.
*   **Standard-precision:** 64-bit integers with up to 9 decimals of precision.

## Installation

Install the latest version using `pip`:

```bash
pip install -U nautilus_trader
```

Or, install from the [Nautech Systems package index](https://packages.nautechsystems.io/simple/nautilus-trader/index.html).

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for detailed instructions.

## Redis

Redis is **optional** and only required if configured as the backend for a cache or message bus.

## Makefile

The `Makefile` automates installation and build tasks. Use `make help` for details.

## Examples

Find example indicators, and strategies in the examples directory.

## Docker

Docker images are available for easy deployment.

*   `nautilus_trader:latest`: Latest release version.
*   `nautilus_trader:nightly`: Head of the `nightly` branch.
*   `jupyterlab:latest`: Latest release version with JupyterLab.
*   `jupyterlab:nightly`: Head of the `nightly` branch with JupyterLab.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for information.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and the [open-source scope](/ROADMAP.md#open-source-scope) for guidelines.

## Community

Join the [Discord](https://discord.gg/NautilusTrader) community.

> [!WARNING]
>
> NautilusTrader does not endorse any cryptocurrency tokens.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems. For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">