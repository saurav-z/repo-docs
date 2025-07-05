# NautilusTrader: High-Performance Algorithmic Trading Platform

[![NautilusTrader Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png)](https://github.com/nautechsystems/nautilus_trader)

**NautilusTrader empowers quantitative traders with a high-performance, open-source platform for backtesting and live deployment of algorithmic trading strategies.**

[**View the original repository on GitHub**](https://github.com/nautechsystems/nautilus_trader)

---

**Key Features:**

*   **High Performance:** Core components written in Rust for speed and efficiency.
*   **Reliable & Safe:**  Rust's type and thread safety, with optional Redis-backed state persistence.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows with Docker support.
*   **Modular & Flexible:** Integrate any data feed or venue with modular adapters.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, and more.
*   **Backtesting & Live Trading Parity:** Identical strategy code for backtesting and live deployment.
*   **AI Training Ready:** The fast backtest engine can be used to train AI trading agents (RL/ES).
*   **Multi-Venue Support**: Facilitates market-making and statistical arbitrage strategies.

---

## Introduction

NautilusTrader is a powerful open-source algorithmic trading platform designed for professional quantitative traders. It offers a high-performance environment for backtesting, and live deployment of automated trading strategies.  Built with a focus on performance and reliability, NautilusTrader empowers traders to build and deploy sophisticated strategies across various asset classes with ease. The platform leverages the strengths of both Python and Rust to provide a robust and efficient trading experience.

## Why Choose NautilusTrader?

*   **Event-Driven Python with Binary Core:** Experience the speed of Rust-powered core components within a Python-native environment.
*   **Code Consistency:** Leverage the same strategy code for both backtesting and live trading.
*   **Reduced Operational Risk:** Benefit from enhanced risk management, logical accuracy, and type safety.
*   **Highly Extensible:** Customize your experience with a message bus, custom components, actors, custom data, and adapters.

## Key Technologies

*   **Python:** The front-end for strategy development, leveraging Python's extensive libraries for data science and AI.
*   **Rust:** Used for core performance-critical components.  Rust ensures memory safety and thread safety, leading to more reliable trading systems.
*   **Cython:** Bridges the gap between Python and Rust, allowing for seamless integration and optimized performance.

## Integrations

NautilusTrader offers a modular design with *adapters* to connect to various trading venues and data providers.

**[See the Integrations Documentation](https://nautilustrader.io/docs/latest/integrations/) for a comprehensive list.**

Some of the supported integrations include:

*   Betfair
*   Binance
*   Bybit
*   Coinbase International
*   Databento
*   dYdX
*   Interactive Brokers
*   OKX
*   Polymarket
*   Tardis

## Installation

Install `nautilus_trader` using Python's `pip` package manager.

```bash
pip install -U nautilus_trader
```

You can also install development versions using the Nautech Systems package index.

**See [the installation guide](https://nautilustrader.io/docs/latest/getting_started/installation) for complete instructions.**

## Community

Join our community on [Discord](https://discord.gg/NautilusTrader) to connect with other users, stay updated on the latest developments, and get your questions answered.

---

**Disclaimer:**  NautilusTrader is an open-source project and is not affiliated with, nor does it endorse, any specific cryptocurrency tokens. All official communications are through the official website, [Discord](https://discord.gg/NautilusTrader), or X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).

**License:**  The source code is available under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).