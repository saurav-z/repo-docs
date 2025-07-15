# NautilusTrader: High-Performance Algorithmic Trading Platform

[![Nautilus Trader Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png)](https://github.com/nautechsystems/nautilus_trader)

**NautilusTrader is an open-source, AI-first algorithmic trading platform built for performance and reliability, enabling quantitative traders to backtest and deploy strategies with ease.** Explore the original repository at [https://github.com/nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader).

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

*   **Docs**: <https://nautilustrader.io/docs/>
*   **Website**: <https://nautilustrader.io>
*   **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Core components written in Rust for speed and efficiency.
*   **Reliable & Safe:** Rust-powered type and thread safety, with optional Redis persistence.
*   **Cross-Platform:** Compatible with Linux, macOS, and Windows, deployable via Docker.
*   **Modular & Flexible:** Easily integrate with any REST API or WebSocket feed via adapters.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, and more, along with contingency orders like OCO, OUO, and OTO.
*   **Customizable:** Add user-defined components and assemble entire systems.
*   **Backtesting & Live Trading Parity:** Identical strategy code for backtesting and live deployment.
*   **Multi-Venue Support:** Facilitates market-making and arbitrage strategies.
*   **AI Training Ready:** Backtest engine suitable for training AI trading agents.

![Nautilus Trader Art](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png "Nautilus Art")

## Why Choose NautilusTrader?

NautilusTrader offers a powerful solution for algorithmic trading, combining the strengths of Python with the performance of Rust, providing:

*   **Blazing-Fast Execution:** Leverage the speed of Rust for core components.
*   **Unified Strategy Development:** Write strategy code once for both backtesting and live deployment.
*   **Reduced Risk:** Benefit from enhanced risk management, accuracy, and type safety.
*   **Extensible Architecture:** Easily integrate custom components and data sources.

## Introduction

NautilusTrader is an open-source, high-performance, production-grade algorithmic trading platform. It empowers quantitative traders with the ability to backtest portfolios of automated trading strategies on historical data with an event-driven engine and deploy those same strategies live, without any code changes.

The platform is designed to develop and deploy algorithmic trading strategies within a highly performant and robust Python-native environment. This aids in addressing the challenge of maintaining consistency between the Python research/backtest environment and the production live trading environment.

NautilusTrader's design, architecture, and implementation prioritize software correctness and safety at the highest level. The aim is to support Python-native, mission-critical, trading system backtesting and live deployment workloads.

The platform is asset-class-agnostic and universal, with any REST API or WebSocket feed able to be integrated via modular adapters. It supports high-frequency trading across a wide range of asset classes and instrument types, including FX, Equities, Futures, Options, Crypto, and Betting, enabling seamless operations across multiple venues simultaneously.

## Why Python & Rust?

*   **Python:** Python, the *lingua franca* of data science and AI, provides an accessible and flexible environment for strategy development.  NautilusTrader leverages Python's extensive libraries and community.
*   **Rust:** Rust provides performance and safety, guaranteeing memory-safety and thread-safety deterministically. NautilusTrader uses Rust for core, performance-critical components, maximizing speed and reliability.

## Integrations

NautilusTrader offers modular integrations with various trading venues and data providers. Adapters translate raw APIs into a unified interface and normalized domain model.

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

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

*   **ID**: The default client ID for the integrations adapter clients.
*   **Type**: The type of integration (often the venue type).

### Status

*   `building`: Under construction and likely not in a usable state.
*   `beta`: Completed to a minimally working state and in a beta testing phase.
*   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

## Versioning and Releases

**NautilusTrader is actively developed**, and breaking changes can occur. We strive to document these changes in the release notes.

*   **Bi-weekly release schedule** (aim).

### Branches

*   `master`: Latest released version (recommended for production).
*   `nightly`: Daily snapshots of the `develop` branch (for early testing).
*   `develop`: Active development branch (for contributors).

## Precision Mode

NautilusTrader supports:

*   **High-precision:** 128-bit integers (up to 16 decimals).
*   **Standard-precision:** 64-bit integers (up to 9 decimals).

By default, official wheels ship in high-precision mode on Linux and macOS, and standard-precision on Windows.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

## Installation

We recommend using the latest supported Python version and installing within a virtual environment.

*   **From PyPI:**
    ```bash
    pip install -U nautilus_trader
    ```

*   **From Nautech Systems package index:**
    ```bash
    pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
    ```
    Install pre-release versions:
    ```bash
    pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
    ```

*   **From Source:** Requires Rust and relevant build tools (see instructions).

## Redis

Redis is **optional** and only required if configured for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

A `Makefile` streamlines development tasks, including installation, building, testing, and documentation generation.  Run `make help` for details.

## Examples

Examples of indicators, strategies, and backtests are available in both Python and Cython (recommended for performance).

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
*   [indicator](/nautilus_trader/indicators/) examples written in Cython.
*   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
*   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Docker images are available with the latest release and `nightly` builds. Example:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Open in your browser: `http://127.0.0.1:8888/lab`

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development information.

### Testing with Rust

*   Use `cargo-nextest` for efficient and reliable Rust testing.
*   Run Rust tests with `make cargo-test`.

## Contributing

Contributions are welcome! Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your ideas. Follow the guidelines in [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and sign a CLA.  Pull requests should target the `develop` branch.

## Community

Join the community on [Discord](https://discord.gg/NautilusTrader) to connect with other users and developers.

> [!WARNING]
>
> NautilusTrader does not endorse or promote any cryptocurrency tokens.  Official communications are shared via <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader), or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).  Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems. Visit <https://nautilustrader.io> for more information.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">