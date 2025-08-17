# NautilusTrader: High-Performance Algorithmic Trading Platform

[![NautilusTrader Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png)](https://github.com/nautechsystems/nautilus_trader)

**NautilusTrader empowers quantitative traders with a production-grade, AI-first algorithmic trading platform for backtesting and live deployment.**

[Explore the NautilusTrader Repository](https://github.com/nautechsystems/nautilus_trader)

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

- **Docs**: <https://nautilustrader.io/docs/>
- **Website**: <https://nautilustrader.io>
- **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Built with Rust for speed and efficiency, featuring asynchronous networking with tokio.
*   **Reliability & Safety:**  Leverages Rust's type- and thread-safety, with optional Redis-backed state persistence.
*   **Cross-Platform Compatibility:**  Runs on Linux, macOS, and Windows.  Supports Docker deployment.
*   **Flexible Integration:**  Modular adapters for easy integration with REST APIs and WebSocket feeds.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, DAY, AT\_THE\_OPEN, AT\_THE\_CLOSE, and more, plus conditional triggers and contingency orders (OCO, OUO, OTO).
*   **Customization & Extensibility:**  Allows custom components, system assembly via [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Robust Backtesting:**  Backtest with multiple venues, instruments, and strategies simultaneously using historical data.
*   **Seamless Live Deployment:** Identical strategy code for backtesting and live trading.
*   **Multi-Venue Support:**  Enables market-making and statistical arbitrage strategies.
*   **AI Trading Ready:** Backtest engine suitable for training AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

NautilusTrader provides a powerful, Python-native environment for quantitative traders, offering:

*   **Blazing Fast Performance:** Python-native environment leveraging Rust for core components.
*   **Backtesting and Live Parity:**  Use the exact same strategy code for both backtesting and live trading.
*   **Reduced Operational Risk:**  Enhanced risk management features, logical accuracy, and type safety.
*   **Extensive Extensibility:**  Message bus, custom components and actors, custom data, custom adapters.

## Deep Dive: Python, Rust, and the Parity Challenge

NautilusTrader addresses the critical "parity challenge" in algorithmic trading: the discrepancy between research/backtesting environments (often Python) and production live trading systems (frequently C++, C#, or Java for performance).  By leveraging Rust for core components and providing a Python-native interface, NautilusTrader enables seamless transition from backtesting to live deployment, simplifying the development and deployment of algorithmic trading strategies.

## Integrations

NautilusTrader offers a modular design, allowing for integration with various trading venues and data providers through *adapters*.

### Supported Integrations:

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

-   **ID**:  Integration client ID.
-   **Type**:  Integration type (e.g., Exchange, Data Provider).
-   **Status**: Current state of integration (e.g., building, beta, stable).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

*NautilusTrader is actively developed, with bi-weekly releases planned.*

### Branches

*   `master`: Latest released version, recommended for production.
*   `nightly`: Daily snapshots of `develop`.
*   `develop`: Active development branch.

> [!NOTE]
>
> **Versioning:** We aim for a stable API for version 2.x, with formal deprecation processes planned.

## Precision Mode

NautilusTrader supports two precision modes for `Price`, `Quantity`, and `Money` values:

*   **High-precision:**  128-bit integers, up to 16 decimals, wider value range.
*   **Standard-precision:** 64-bit integers, up to 9 decimals, smaller value range.

> [!NOTE]
>
> High-precision (128-bit) mode is default for official Python wheels on Linux and macOS. Standard-precision (64-bit) is used on Windows.

**Rust Feature Flag:** To enable high-precision in Rust, add the feature to `Cargo.toml`:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

*   **Prerequisites:**  Latest Python version, virtual environment recommended.

**Installation Methods:**

1.  Pre-built binary wheels from PyPI or Nautech Systems package index.
2.  Build from source.

> [!TIP]
>
> Use the [uv](https://docs.astral.sh/uv) package manager.

### From PyPI:

```bash
pip install -U nautilus_trader
```

### From Nautech Systems Package Index:

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

### From Source:

1.  Install Rust, clang, and uv (see the README for complete instructions).
2.  Clone the repository: `git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader`
3.  `cd nautilus_trader`
4.  `uv sync --all-extras`
5.  Set environment variables as detailed in the original README.
6.  `uv build`

## Redis

Using Redis is *optional* and needed only if configured for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus). See the installation guide for details.

## Makefile

The Makefile simplifies many development tasks.

*   `make install`: Install in release mode.
*   `make build`: Run the build script in release build mode.
*   `make cargo-test`: Runs all Rust crate tests using `cargo-nextest`.
*   `make clean`: Deletes build files.
*   `make docs`: Builds documentation.
*   `make ruff`: Runs ruff over all files using the `pyproject.toml` config (with autofix).
*   `make pytest`: Runs all tests with `pytest`.

Run `make help` for a complete list.

## Examples

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
*   [indicator](/nautilus_trader/indicators/) examples written in Cython.
*   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
*   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Docker images are available:

*   `nautilus_trader:latest`:  Latest release.
*   `nautilus_trader:nightly`: Head of `nightly`.
*   `jupyterlab:latest`: Release version with JupyterLab and example notebook.
*   `jupyterlab:nightly`: Nightly version with JupyterLab and example notebook.

Pull and run example:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Open browser at:  `http://127.0.0.1:8888/lab`

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for details.

*   `make build-debug` for efficient development.

### Testing

*   [cargo-nextest](https://nexte.st) for reliable Rust testing.
*   `make cargo-test` runs the tests.

## Contributing

Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss enhancements or bug fixes.  Follow the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines.

> [!NOTE]
>
>  Target pull requests to the `develop` branch.

## Community

Join our [Discord](https://discord.gg/NautilusTrader) for community discussions.

> [!WARNING]
>
>  NautilusTrader is not affiliated with any cryptocurrency or token promotions. Official updates and communications are only through our website, Discord, or X (Twitter) account.

## License

Licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).  Contributions require a CLA.

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:** The one-sentence hook immediately highlights the platform's core value.
*   **Keyword Optimization:** Keywords like "algorithmic trading," "high-performance," "backtesting," "live deployment," "AI-first," "Python," and "Rust" are strategically placed throughout the document.
*   **Strategic Heading Use:**  Clear headings and subheadings organize the information logically.
*   **Bulleted Key Features:** Makes the main selling points immediately scannable.
*   **Concise Summaries:**  Replaces lengthy descriptions with summarized information, emphasizing key benefits.
*   **Emphasis on Benefits:**  The "Why Choose NautilusTrader?" and "Why Python/Rust?" sections focus on user advantages.
*   **Clear Call to Action:**  The "Explore the NautilusTrader Repository" link provides a direct way for users to get started.
*   **Internal Linking:** Use of internal links to improve the navigation.
*   **Improved Readability:** Uses markdown formatting for improved readability.
*   **Complete Information:**  Includes all important information from the original README, but restructured and condensed for clarity.
*   **SEO-Friendly Formatting:** Utilizes headings, bold text, and lists for improved search engine visibility.
*   **Mobile-Friendly:** Uses standard markdown that displays well on various devices.
*   **Warnings and Tips:** Added using the [!NOTE] [!TIP] and [!WARNING] for better user experience and discoverability.
*   **Focus on Performance and Rust:**  Greater emphasis given to the performance benefits and the use of Rust, a key differentiator.