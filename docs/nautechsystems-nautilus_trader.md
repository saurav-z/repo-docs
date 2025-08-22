# NautilusTrader: The High-Performance Algorithmic Trading Platform

[![Nautilus Trader Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png)](https://github.com/nautechsystems/nautilus_trader)

**Develop and deploy lightning-fast, Python-native algorithmic trading strategies with NautilusTrader, an open-source platform.**  [Explore the original repository](https://github.com/nautechsystems/nautilus_trader).

**Key Features:**

*   ⚡ **High Performance:** Built with Rust for speed and efficiency.
*   ✅ **Reliable & Safe:** Rust-powered, type-safe core with optional Redis persistence.
*   💻 **Cross-Platform:** Runs on Linux, macOS, and Windows, with Docker support.
*   🔌 **Modular & Flexible:** Integrate with any REST API or WebSocket feed via adapters.
*   ⚙️ **Advanced Order Types:**  Supports IOC, FOK, GTC, GTD, and more, plus execution instructions.
*   🎨 **Customizable:**  Add custom components and build systems leveraging the cache and message bus.
*   📊 **Backtesting & Live Trading Parity:** Identical strategy code for backtesting and live deployment.
*   🌎 **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   🤖 **AI Training Ready:** Backtest engine fast enough for AI trading agent training.

## Core Concepts

NautilusTrader is designed to overcome the common challenge of maintaining consistency between Python-based research environments and production trading systems. The platform's core is written in Rust, ensuring high performance and reliability, while offering a Python-native environment for quantitative traders.

## Why Choose NautilusTrader?

*   **High-Performance Python:** Leverage the speed of native binary components.
*   **Code Parity:** Use the same strategy code for both backtesting and live trading.
*   **Reduced Risk:** Benefit from enhanced risk management and type safety.
*   **Extensibility:** Easily integrate custom components, data sources, and adapters.

## Key Technologies

*   **Rust:**  Ensures performance, memory safety, and thread safety.
*   **Python:** Provides a familiar and powerful environment for data science and strategy development.
*   **Cython:** Bridges the gap between Python and compiled languages for high-performance modules.

## Integrations

NautilusTrader connects to various trading venues and data providers through modular adapters.  See the [Integrations Documentation](https://nautilustrader.io/docs/latest/integrations/index.html) for full details.

*   Betfair
*   Binance
*   Binance US
*   Binance Futures
*   BitMEX
*   Bybit
*   Coinbase International
*   Databento
*   dYdX
*   Hyperliquid
*   Interactive Brokers
*   OKX
*   Polymarket
*   Tardis

## Installation

Install NautilusTrader using `pip` (recommended), or build from source.

### From PyPI (Recommended)

```bash
pip install -U nautilus_trader
```

### From Nautech Systems Package Index (For Pre-releases)

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

### From Source

Follow the instructions in the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for setting up Rust, and then use `uv` to install.

## Development

Explore the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for coding guidelines, testing, and contributions.

## Community & Support

*   **Discord:** Join the NautilusTrader community for discussions and support: [Discord](https://discord.gg/NautilusTrader)
*   **Docs:** [https://nautilustrader.io/docs/](https://nautilustrader.io/docs/)
*   **Website:** [https://nautilustrader.io](https://nautilustrader.io)
*   **Support:** [support@nautilustrader.io](mailto:support@nautilustrader.io)

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

*NautilusTrader™ is developed and maintained by Nautech Systems, a technology company specializing in the development of high-performance trading systems. For more information, visit [https://nautilustrader.io](https://nautilustrader.io).*