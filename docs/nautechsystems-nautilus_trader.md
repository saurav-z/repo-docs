# NautilusTrader: High-Performance Algorithmic Trading Platform

**NautilusTrader is an open-source, AI-first trading platform, empowering quantitative traders to build, backtest, and deploy sophisticated strategies with ease.** [Explore the NautilusTrader Repository](https://github.com/nautechsystems/nautilus_trader)

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

*   **High Performance:** Built with Rust for speed and efficiency.
*   **Python-Native:** Seamless integration with Python for strategy development.
*   **Backtesting & Live Trading:** Identical strategy code for both backtesting and live deployment.
*   **Modular Design:** Easily integrate with any REST API or WebSocket feed.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows.
*   **Advanced Order Types:** Supports complex order types and conditional triggers.
*   **AI Ready:** Fast backtesting engine for training AI trading agents.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

*   **Maximize Performance:** Benefit from a high-performance, event-driven core.
*   **Reduce Risk:** Leverage enhanced risk management and type safety.
*   **Accelerate Development:** Eliminate the need to rewrite strategies for live trading.
*   **Customize & Extend:** Utilize message bus, custom components, and adapters.

NautilusTrader bridges the gap between Python-based strategy research and production trading environments by combining the expressiveness of Python with the performance of Rust.

## Tech Stack

### Rust

*   Rust is utilized for core performance-critical components, offering speed, memory efficiency, and thread safety.
*   [Rust](https://www.rust-lang.org/) provides deterministic memory and thread safety and eliminates numerous bugs at compile time.

### Python

*   Python is the foundation for strategy development due to its clean syntax and extensive libraries.
*   Cython is used to overcome Python performance limitations by introducing static typing.

## Integrations

NautilusTrader offers modular *adapters* for seamless connectivity to trading venues and data providers.

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
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

-   **ID**: The default client ID for the integrations adapter clients.
-   **Type**: The type of integration (often the venue type).

### Status

-   `building`: Under construction and likely not in a usable state.
-   `beta`: Completed to a minimally working state and in a beta testing phase.
-   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

**NautilusTrader is under active development.** The API is evolving, and breaking changes are possible between releases.

### Branches

*   `master`: Latest released version (recommended for production).
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch for new features.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports two precision modes for core data types:

*   **High-precision**: 128-bit integers, up to 16 decimals.
*   **Standard-precision**: 64-bit integers, up to 9 decimals.

> [!NOTE]
>
> High-precision is enabled by default on Linux and macOS; standard-precision on Windows.

**Rust feature flag**: Enable high-precision with `high-precision` in `Cargo.toml`.

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install NautilusTrader using `pip`:

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

### From Source

1.  Install Rust and dependencies.
2.  Clone the repository.
3.  Install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
4.  Install the project:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```
    ```bash
    # Set environment variables for PyO3 compilation (Linux and macOS only):
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for detailed instructions.

## Redis

Redis is **optional** and only needed for cache or message bus backends. See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for setup.

## Makefile

A `Makefile` automates common tasks; use `make help` for details.

## Examples

Explore Python and Cython examples for indicators, strategies, and backtesting.

## Docker

Pre-built Docker images are available:

*   `nautilus_trader:latest`: Latest release.
*   `nautilus_trader:nightly`: Head of `nightly` branch.
*   `jupyterlab:latest`: Latest release with JupyterLab.
*   `jupyterlab:nightly`: Head of `nightly` branch with JupyterLab.

Pull and run with:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development information.

> [!TIP]
>
> Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

### Testing with Rust

*   Use [cargo-nextest](https://nexte.st) for Rust tests.

    ```bash
    cargo install cargo-nextest
    ```

> [!TIP]
>
> Run Rust tests with `make cargo-test`.

## Contributing

Contribute to NautilusTrader! Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues), review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file, and sign a Contributor License Agreement (CLA).

> [!NOTE]
>
> Pull requests should target the `develop` branch (the default branch). This is where new features and improvements are integrated before release.

## Community

Join our [Discord](https://discord.gg/NautilusTrader) community.

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens.  All official updates and communications are shared through our website, Discord server, or X (Twitter) account [@NautilusTrader](https://x.com/NautilusTrader).

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology
company specializing in the development of high-performance trading systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">