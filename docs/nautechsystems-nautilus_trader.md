# NautilusTrader: The High-Performance Algorithmic Trading Platform

[![NautilusTrader Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png)](https://github.com/nautechsystems/nautilus_trader)

**NautilusTrader empowers quantitative traders with a robust, high-performance platform for backtesting and deploying automated trading strategies, all with the same code.**  [Visit the original repository](https://github.com/nautechsystems/nautilus_trader).

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

## Core Features

*   **High Performance:**  Built with Rust for speed and efficiency.
*   **Python-Native:** Develop and deploy strategies within a familiar Python environment.
*   **Backtesting & Live Trading Parity:** Use the same code for backtesting and live deployment.
*   **Event-Driven Architecture:**  Provides a robust foundation for complex trading logic.
*   **Asset Class Agnostic:**  Supports a wide range of assets, including FX, Equities, Futures, Options, Crypto, and Betting.
*   **Modular & Extensible:** Integrate with various data feeds and trading venues through modular adapters.
*   **Advanced Order Types:** Includes advanced order types like IOC, FOK, GTC, GTD, and contingency orders.
*   **Backtesting Capabilities:** Backtest with high-resolution data, multiple venues, and strategies simultaneously.
*   **AI Training Ready:**  Fast backtesting engine for training AI trading agents (RL/ES).

## Why Choose NautilusTrader?

*   **Speed and Performance:** Leveraging Rust for critical components.
*   **Reduced Operational Risk:** Enhanced risk management and type safety.
*   **Unified Environment:** Consistent environment for research, backtesting, and live trading.
*   **Extensibility:**  Modular design allows for custom components and integrations.

## Technical Details

NautilusTrader addresses the challenge of maintaining parity between research/backtesting and live trading environments.  It achieves this by:

*   **Rust Core:** Performance-critical components are written in Rust.
*   **Python Integration:** Python bindings are implemented with Cython, providing a Python-native experience.
*   **Open Source:**  Empowering the quant community.

## Integrations

NautilusTrader offers a flexible, modular design with adapters for various trading venues and data providers.

### Supported Integrations:

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
| [Interactive Brokers](https://www.interactivebrokers.com)                    | `INTERACTIVE_BROKERS` | Brokerage (multi-venue) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/ib.md)            |
| [OKX](https://okx.com)                                                       | `OKX`                 | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/building-orange) | [Guide](docs/integrations/okx.md)           |
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

*   **ID:** The unique identifier for the integration.
*   **Type:** The integration type.
*   **Status:** Indicates the current state of the integration (building, beta, or stable).

### Integration Statuses:

*   `building`: Under development.
*   `beta`:  Functionality is minimally working.
*   `stable`: Fully tested and functional.

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule and utilizes the following branches:

*   `master`:  Latest released version.
*   `nightly`:  Daily snapshots from the `develop` branch.
*   `develop`:  Active development branch.

## Precision Mode

NautilusTrader offers high and standard precision modes for core value types: `Price`, `Quantity`, and `Money`.  The default is high-precision on Linux and macOS and standard precision on Windows.

*   **High-precision:** 128-bit integers, up to 16 decimal places.
*   **Standard-precision:** 64-bit integers, up to 9 decimal places.

**Rust feature flag:** To enable high-precision in Rust, include `high-precision` in your `Cargo.toml`:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install NautilusTrader using the pre-built binary wheel from PyPI or the Nautech Systems package index, or by building from source.

**Recommended:** Use `uv` with a "vanilla" CPython.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems Package Index

#### Stable wheels

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development wheels

Install pre-release versions using:

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

Install a specific development wheel:

```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

#### Available Versions

See available versions at:  [package index](https://packages.nautechsystems.io/simple/nautilus-trader/index.html).

### From Source

1.  Install Rustup and required tools (clang, etc.).
2.  Clone the repository:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```
3. Set environment variables for PyO3 compilation (Linux and macOS only):

    ```bash
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```
4. Install from the project root using `uv`.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for complete instructions.

## Redis

Redis is optional and used as a backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

The provided `Makefile` streamlines common tasks:

*   `make install`: Installs with all dependencies.
*   `make build`: Builds the project.
*   `make test`: Runs tests.
*   `make docs`: Builds documentation.

## Examples

Find example indicators and strategies in Python and Cython:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) (Python)
*   [indicator](/nautilus_trader/indicators/) (Cython)
*   [strategy](/nautilus_trader/examples/strategies/) (Python)
*   [backtest](/examples/backtest/)

## Docker

Pre-built Docker images are available on `ghcr.io/nautechsystems/`.  Use the `jupyterlab` variants for a ready-to-use JupyterLab environment.

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

Refer to the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development insights.

### Testing with Rust

Use `cargo-nextest` to run Rust tests efficiently. Install with:

```bash
cargo install cargo-nextest
```

Run Rust tests with `make cargo-test`.

## Contributing

Contribute to NautilusTrader by opening an [issue](https://github.com/nautechsystems/nautilus_trader/issues) and following the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines, including signing a Contributor License Agreement (CLA).  Pull requests should target the `develop` branch.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) to connect with other users and developers.

> [!WARNING]
> NautilusTrader is not associated with any cryptocurrency tokens.  Official communications are limited to the website, Discord server, and X (Twitter) account.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).
A [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md) is required for contributions.

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">