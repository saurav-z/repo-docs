# NautilusTrader: High-Performance Algorithmic Trading Platform

[![NautilusTrader Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png)](https://github.com/nautechsystems/nautilus_trader)

**Unleash the power of algorithmic trading with NautilusTrader, an open-source platform designed for performance, reliability, and flexibility.** ([View on GitHub](https://github.com/nautechsystems/nautilus_trader))

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

- **Docs:** <https://nautilustrader.io/docs/>
- **Website:** <https://nautilustrader.io>
- **Support:** [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Core components written in Rust for speed and efficiency, utilizing asynchronous networking with tokio.
*   **Reliability:** Built with Rust's type- and thread-safety, enhanced with optional Redis-backed state persistence.
*   **Cross-Platform Compatibility:** Runs seamlessly on Linux, macOS, and Windows.  Docker deployment supported.
*   **Modular and Flexible:** Integrates with any REST API or WebSocket feed through modular adapters.
*   **Advanced Order Types:** Supports a wide range of advanced order types and conditional triggers for sophisticated trading strategies. Execution instructions like `post-only`, `reduce-only`, and icebergs, along with contingency orders like `OCO`, `OUO`, and `OTO`.
*   **Customizable:** Allows users to create custom components or build entire systems from scratch using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Robust Backtesting:**  Backtest strategies with nanosecond resolution using historical data, supporting multiple venues, instruments, and strategies simultaneously.
*   **Seamless Live Deployment:** Use the same strategy implementations for both backtesting and live trading.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies through multi-venue capabilities.
*   **AI Training Ready:** Backtest engine optimized for training AI trading agents (RL/ES).

## Why Choose NautilusTrader?

NautilusTrader offers a compelling solution for quantitative traders seeking a robust and efficient algorithmic trading platform:

*   **Superior Performance:** Benefit from a high-performance, event-driven Python environment with native binary components.
*   **Code Reusability:** Achieve parity between backtesting and live trading environments with identical strategy code.
*   **Reduced Risk:** Leverage enhanced risk management features, logical accuracy, and type safety for improved operational efficiency.
*   **Extensibility:**  Create and customize through a flexible architecture offering a message bus, custom components and actors, custom data and adapters.

## Technology Stack: Python, Rust, and Cython

NautilusTrader leverages the strengths of several key technologies to provide a powerful and efficient trading platform:

*   **Python:** Provides a user-friendly and versatile environment for strategy development, leveraging its extensive data science and machine learning libraries.
*   **Rust:** Powers the core performance-critical components with its blazingly fast speed, memory efficiency, and memory and thread safety.
*   **Cython:** Bridges the gap between Python and Rust, enabling static typing and performance optimizations for critical components.

## Integrations

NautilusTrader is designed for seamless integration with various trading venues and data providers. Its modular adapter system enables you to connect to the following:

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

## Installation

Choose from several installation methods to get started:

*   **PyPI:** Install the latest release with `pip install -U nautilus_trader`.
*   **Nautech Systems Package Index:** Access stable and development wheels with `pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple`.
*   **Build from Source:** Follow detailed instructions in the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation).

## Additional Information

*   **Precision Mode:**  NautilusTrader supports High-precision and Standard-precision modes for enhanced accuracy.
*   **Redis:** Optional integration for cache and message bus functionality.
*   **Makefile:** Simplifies development with targets for installation, building, testing, and documentation.
*   **Examples:** Explore Python and Cython examples for indicators, strategies, and backtesting.
*   **Docker:** Utilize pre-built Docker containers for easy deployment and testing.
*   **Development:** Consult the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development workflows.
*   **Contributing:**  Contribute to NautilusTrader by opening issues and following the guidelines in [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md).
*   **Community:** Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader).

---

NautilusTrader™ is developed and maintained by Nautech Systems. For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO considerations:

*   **Clear Hook:** The first sentence immediately grabs the reader's attention and introduces the core benefit.
*   **Keyword Optimization:**  Uses relevant keywords throughout (e.g., "algorithmic trading," "high-performance," "open-source," "Python," "Rust").
*   **Concise Headings:**  Well-structured with clear headings for readability and SEO.
*   **Bulleted Lists:**  Highlights key features and benefits for easy scanning.
*   **Benefit-Driven Language:** Focuses on what the platform *does* for the user, not just its technical aspects.
*   **Internal Linking:**  Links to key concepts within the platform (e.g., cache, message bus).
*   **External Linking:** Uses descriptive link text for all external links.
*   **Strong Call to Action (Implied):**  The entire README encourages exploration and contribution.
*   **Mobile-Friendly:**  Uses standard Markdown for good rendering on all devices.
*   **Updated Information:**  Includes current links, and project status badges.
*   **Removed Duplication:** Streamlined repeated information from original README.