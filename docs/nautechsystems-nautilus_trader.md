# NautilusTrader: High-Performance Algorithmic Trading Platform

[![Codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![Codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nautilus_trader)](https://pypi.org/project/nautilus_trader/)
[![PyPI Version](https://img.shields.io/pypi/v/nautilus_trader)](https://pypi.org/project/nautilus_trader/)
[![PyPI Package Format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)](https://pypi.org/project/nautilus_trader/)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

NautilusTrader is an open-source, AI-first algorithmic trading platform providing professional quantitative traders with backtesting and live deployment capabilities for high-frequency trading strategies. [Explore the NautilusTrader repository](https://github.com/nautechsystems/nautilus_trader).

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

-   **Docs**: <https://nautilustrader.io/docs/>
-   **Website**: <https://nautilustrader.io>
-   **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Core components written in Rust with asynchronous networking using [tokio](https://crates.io/crates/tokio) for speed and efficiency.
*   **Robust & Reliable:** Type- and thread-safe due to Rust, with optional Redis-backed state persistence.
*   **Cross-Platform Compatibility:** Runs on Linux, macOS, and Windows; deployment is simplified with Docker.
*   **Modular & Flexible:** Integrate any REST API or WebSocket feed easily with modular adapters.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, DAY, AT_THE_OPEN, AT_THE_CLOSE, and conditional order triggers, alongside execution instructions like post-only, reduce-only, icebergs, and contingency orders including OCO, OUO, OTO.
*   **Customizable:** Build your own components or assemble entire systems from scratch by leveraging the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Comprehensive Backtesting:** Analyze strategies using historical data, including quote tick, trade tick, bar, order book, and custom data with nanosecond resolution, across multiple venues.
*   **Seamless Live Deployment:** Use the exact same strategy code for both backtesting and live trading.
*   **Multi-Venue Support:** Facilitate market-making and statistical arbitrage strategies.
*   **AI Training:** Offers a fast backtest engine suitable for training AI trading agents (RL/ES).

## Why Choose NautilusTrader?

*   **Performance-Driven:** Benefit from a high-performance, event-driven platform powered by Python with native binary core components.
*   **Code Parity:** Maintain identical strategy implementations for backtesting and live trading, reducing development time and risk.
*   **Reduced Risk:** Minimize operational risks with enhanced risk management, improved logical accuracy, and type safety.
*   **Extensible Design:** Build upon the framework via the message bus, custom components, custom data, and custom adapters.

## Python, Rust, and the NautilusTrader Advantage

NautilusTrader leverages the strengths of both Python and Rust. Python offers ease of use and a vast ecosystem for data science and AI, while Rust delivers performance, memory safety, and thread safety essential for high-frequency trading.  This platform bridges the gap between research/backtesting and live trading.

## Integrations

NautilusTrader uses modular *adapters* for connecting to trading venues and data providers.

The following integrations are currently supported; see [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details:

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

NautilusTrader follows a bi-weekly release schedule.

### Branches

*   `master`: Reflects the latest released version.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch.

## Precision Mode

NautilusTrader supports two precision modes for its core value types.

-   **High-precision**: 128-bit integers.
-   **Standard-precision**: 64-bit integers.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

## Installation

Install using pip or build from source.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

Development wheels are also available.

### From Source

Follow the installation steps outlined in the README.

## Redis

Redis is optional, used for caching and message bus functionality.

## Makefile

The `Makefile` automates many development and build tasks.

## Examples

Examples are available for indicators, strategies, and backtesting.

## Docker

Docker containers are provided for easy deployment.

## Development

Consult the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development information.

### Testing with Rust

Use `cargo-nextest` for Rust testing.

## Contributing

Contributions are welcome!  See the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file.

## Community

Join the [Discord](https://discord.gg/NautilusTrader) community.