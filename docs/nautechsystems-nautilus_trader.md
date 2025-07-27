# NautilusTrader: The High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader is an open-source, AI-first algorithmic trading platform designed for high-performance backtesting and live deployment of trading strategies.**  Developed by [Nautech Systems](https://nautilustrader.io), it provides quantitative traders with a robust and efficient environment to build and deploy automated trading strategies. [Check out the original repo](https://github.com/nautechsystems/nautilus_trader).

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

*   **High Performance:** Core components written in Rust, optimized for speed and efficiency.
*   **Event-Driven Architecture:**  Designed for low-latency trading strategies.
*   **Backtesting and Live Trading Parity:**  Use the same strategy code for backtesting and live deployment.
*   **Python-Native Environment:**  Built for quantitative traders, focusing on consistent Python environments.
*   **Cross-Platform Compatibility:**  Runs on Linux, macOS, and Windows.
*   **Modular Design:** Integrates with various trading venues and data providers through modular adapters.
*   **Flexible Order Types:** Supports advanced order types and conditional triggers.
*   **Customizable Components:**  Allows for user-defined components and system assembly.
*   **AI-Training Capable:** Fast backtesting engine suitable for training AI trading agents.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

*   **Performance:** Leverages Rust for speed and reliability.
*   **Consistency:** Maintains code parity between backtesting and live trading environments.
*   **Safety:** Enhanced risk management and type safety to reduce operational risk.
*   **Extensibility:** Offers a flexible architecture for custom development.

## Why Python and Rust?

NautilusTrader leverages the strengths of both Python and Rust:

*   **Python:**  Provides a clean and accessible syntax, making it ideal for rapid development and machine learning in finance.
*   **Rust:** Ensures high performance and safety for critical components, allowing for safe concurrency.

## Integrations

NautilusTrader provides modular *adapters* to connect with various trading venues and data providers.

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

-   **ID**:  Default client ID for the integration adapter clients.
-   **Type**: The type of integration (often the venue type).

### Status

-   `building`: Under construction
-   `beta`: Minimally working and in beta testing
-   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

**NautilusTrader is actively developed.**

*   **Bi-weekly release schedule.**
*   `master`: Source code for the latest released version.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch.

## Precision Mode

NautilusTrader supports two precision modes for its core value types (`Price`, `Quantity`, `Money`):

*   **High-precision**: 128-bit integers (16 decimals)
*   **Standard-precision**: 64-bit integers (9 decimals)

By default, the Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS. On Windows, only standard-precision (64-bit) is available.

## Installation

It is recommended to install [nautilus_trader](https://pypi.org/project/nautilus_trader/) in a virtual environment.

**Two supported installation methods:**

1.  Pre-built binary wheel from PyPI *or* the Nautech Systems package index.
2.  Build from source.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

## From Source

1.  Install Rust
2.  Install clang
3.  Install uv
4.  Clone the source: `git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader`
5.  `cd nautilus_trader`
6.  Install: `uv sync --all-extras`

> [!NOTE]
>
> Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

## Redis

Optional; required if using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).  See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for further details.

## Make Targets

Use the provided `Makefile` for automating installation and build tasks:

*   `make install`: Installs with all dependencies.
*   `make install-debug`: Installs in debug mode.
*   `make build`: Runs the build script in release mode.
*   `make test-performance`: Runs performance tests with [codspeed](https://codspeed.io).

## Examples

Examples available to start:
*   [indicator](/nautilus_trader/examples/indicators/ema_python.py)
*   [indicator](/nautilus_trader/indicators/)
*   [strategy](/nautilus_trader/examples/strategies/)
*   [backtest](/examples/backtest/)

## Docker

Docker images are available with different tags:

*   `nautilus_trader:latest` - Latest release version.
*   `nautilus_trader:nightly` - Head of the `nightly` branch.
*   `jupyterlab:latest` - Latest release version with JupyterLab.
*   `jupyterlab:nightly` - Head of the `nightly` branch with JupyterLab.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for more information.

## Contributing

Contributions are welcome.  Please see the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and [open-source scope](/ROADMAP.md#open-source-scope) for more details.

## Community

Join the community on [Discord](https://discord.gg/NautilusTrader) for discussions and updates.

> [!WARNING]
>
> NautilusTrader does not promote or endorse any cryptocurrency tokens. All official communications are on <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader), or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).

## License

The code is available under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO considerations:

*   **Clear Title:**  The title now uses the primary keyword, "Algorithmic Trading Platform".
*   **Concise Hook:**  A strong one-sentence introduction is placed at the beginning.
*   **Keyword Optimization:**  Uses relevant keywords like "high-performance," "backtesting," "live deployment," "Python," and "Rust" throughout the document.
*   **Headings:** Added clear headings for better organization and SEO.
*   **Bulleted Key Features:**  Highlights important features for easy scanning and understanding.
*   **Emphasis on Benefits:** Clearly states *why* someone should choose NautilusTrader.
*   **Clear Installation Instructions:**  Improved clarity for installation methods.
*   **Community and Support:** Clearly highlights community links.
*   **Roadmap and Contributing Links:** Improved emphasis for the contributing process.
*   **Visual Appeal:** Keeps the images and uses Markdown formatting for readability.
*   **Reduced Verbosity:** Removed some redundant information.
*   **Corrected some of the markdown syntax**