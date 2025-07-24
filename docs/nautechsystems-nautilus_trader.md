# NautilusTrader: High-Performance Algorithmic Trading Platform

[![Codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![Python Versions](https://img.shields.io/pypi/pyversions/nautilus_trader)
![PyPI Version](https://img.shields.io/pypi/v/nautilus_trader)
![PyPI Format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

NautilusTrader is an open-source, *AI-first* algorithmic trading platform empowering quantitative traders to build, backtest, and deploy high-performance trading strategies.  [Explore the original repository](https://github.com/nautechsystems/nautilus_trader).

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

*   **High Performance:** Powered by a Rust-native core for exceptional speed and efficiency.
*   **Python-Native:**  Provides a Python-native environment for research and backtesting.
*   **Backtesting & Live Trading Parity:**  Use the same strategy code for both backtesting and live deployment.
*   **Modular and Extensible:** Integrate with various trading venues and data providers via modular adapters.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows.
*   **Advanced Order Types:** Supports sophisticated order types and conditional triggers.
*   **AI Training Capable:**  Backtest engine fast enough for training AI trading agents.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why NautilusTrader?

NautilusTrader addresses the challenge of maintaining consistency between Python-based research/backtesting environments and production trading systems, by providing:

*   **High-Performance Event-Driven Python:** Leverages native binary core components.
*   **Identical Strategy Code:** Enables consistent strategy implementation across backtesting and live trading.
*   **Enhanced Risk Management:**  Provides enhanced risk management functionality, logical accuracy, and type safety.
*   **Extensive Customization:** Offers flexibility via the message bus, custom components and actors, custom data, and custom adapters.

## Technology Stack

NautilusTrader utilizes a powerful combination of technologies for optimal performance and reliability:

*   **Rust:** The core of the platform, written in Rust for speed, memory safety, and concurrency.
*   **Python:** Used for strategy development, backtesting, and live deployment.
*   **Cython:**  Used for bridging the gap between Rust and Python, introducing static typing for added performance.

## Integrations

NautilusTrader integrates with a wide range of trading venues and data providers.  See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

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

## Installation

NautilusTrader is easily installed using `pip`, or built from source.

### Installation Methods

*   **From PyPI:** `pip install -U nautilus_trader`
*   **From Nautech Systems Package Index:**  `pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple`
*   **From Source:** Follow the instructions in the README for installing Rust, clang, and setting up your environment.

## Contributing

Contribute to the development of NautilusTrader by opening an [issue](https://github.com/nautechsystems/nautilus_trader/issues) on GitHub to discuss your contributions, and by reviewing the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) to connect with other users and contributors.

```