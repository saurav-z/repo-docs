# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader is an open-source, AI-first trading platform that empowers quantitative traders to backtest and deploy automated strategies with speed and precision.**

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

[Visit the NautilusTrader GitHub Repository](https://github.com/nautechsystems/nautilus_trader)

## Key Features

*   **High Performance:** Core engine written in Rust with asynchronous networking for speed.
*   **Reliability:**  Leverages Rust for type and thread safety and optionally uses Redis for state persistence.
*   **Cross-Platform Compatibility:**  Runs on Linux, macOS, and Windows, with Docker support.
*   **Modular and Flexible:** Integrates with any REST API or WebSocket feed via modular adapters.
*   **Advanced Order Types:** Includes time-in-force, execution instructions, and contingency orders.
*   **Customizable:** Extensible with user-defined components and custom data integration.
*   **Robust Backtesting:** Backtest with nanosecond resolution using historical data across multiple venues.
*   **Live Deployment:**  Use the same strategy code for backtesting and live trading.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   **AI-Driven Development:**  The backtest engine is designed to facilitate training AI trading agents.

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png" alt="Nautilus Art" width="400">

## Why Choose NautilusTrader?

*   **Optimized for Speed:** Leverage the power of Rust and Python for efficient event-driven trading.
*   **Simplified Development:** Achieve code parity between backtesting and live environments.
*   **Reduced Risk:** Benefit from enhanced risk management, logical accuracy, and type safety.
*   **Extensible Architecture:**  Customize the platform with a message bus, custom components, actors, and adapters.

## Technology Stack

NautilusTrader leverages the strengths of both Rust and Python:

*   **Rust:** Provides a "blazingly fast" and memory-efficient core for performance and safety, with guaranteed memory and thread safety.
*   **Python:**  Offers a developer-friendly environment for rapid prototyping, data science, and AI integration.  Cython is used to bridge the performance gap.

## Integrations

NautilusTrader supports a growing list of integrations, translating trading venue APIs into a unified interface.

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

## Installation

NautilusTrader offers flexible installation options:

*   **PyPI:** Install pre-built binary wheels.
*   **Nautech Systems Package Index:** Access stable and development (pre-release) wheels.
*   **Build from Source:**  Build from source for greater control.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for detailed instructions.

## Development

The project welcomes contributions. See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for more information.

*   **Testing:**  Use `cargo-nextest` for reliable Rust testing.
*   **Contribution:**  Open an issue, and follow the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines, including the CLA.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) to collaborate and stay updated.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128" alt="Ferris the Rust mascot">
```
Key improvements and SEO considerations:

*   **Keyword Optimization:**  Strategically incorporates relevant keywords such as "algorithmic trading platform," "backtesting," "high-performance," "Rust," and "Python" throughout the headings and content.
*   **Concise Hook:** Provides a strong, clear sentence to immediately attract the user.
*   **Clear Headings:**  Uses descriptive headings to organize the information and improve readability.
*   **Bulleted Lists:** Highlights key features, making it easier for users to scan and grasp the platform's capabilities.
*   **Concise Language:** Simplifies the language to be clear and easy to understand.
*   **Call to Action:** Encourages users to visit the GitHub repository.
*   **Emphasis on Benefits:**  Focuses on the advantages of using NautilusTrader.
*   **Integration Table Improvement:** The integrations are included in an easy to read table.
*   **Accessibility:** Includes alt text for images.
*   **Community Focus:**  Highlights the Discord community.
*   **Maintainability:** Is easier to read and maintain.