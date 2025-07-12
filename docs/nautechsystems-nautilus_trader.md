# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader is an open-source, AI-first algorithmic trading platform designed for high-performance backtesting and live deployment, all in a Python-native environment.**  [Explore the NautilusTrader Repository](https://github.com/nautechsystems/nautilus_trader).

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

## Key Features of NautilusTrader:

*   **High Performance:** Built with Rust for speed and efficiency, leveraging asynchronous networking with [tokio](https://crates.io/crates/tokio).
*   **Reliability & Safety:**  Rust-powered type and thread safety with optional Redis persistence.
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, and Windows, with Docker deployment support.
*   **Modular & Extensible:** Flexible adapters for easy integration with any REST API or WebSocket feed.
*   **Advanced Order Types:** Supports complex order types, including IOC, FOK, GTC, GTD, and contingent orders.
*   **Customization:** Build your own components or assemble systems from scratch using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Backtesting:**  Run backtests across multiple venues, instruments, and strategies with nanosecond resolution.
*   **Live Trading:**  Use the same strategy implementations for both backtesting and live trading, minimizing code changes.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies across various venues.
*   **AI Training:** The backtest engine is designed to support the training of AI trading agents.

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png" alt="Nautilus Art">

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral.
> The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

NautilusTrader offers:

*   **Python for Algorithmic Trading**: Develop strategies in Python while taking advantage of performance optimizations using Rust and Cython.
*   **Code Reusability**: Write strategies once and deploy them for both backtesting and live trading.
*   **Enhanced Risk Management**: Benefit from increased risk management functionality, improved logical accuracy, and type safety.
*   **Extensibility**: Integrate custom components, actors, and data sources through the message bus.

NautilusTrader addresses the traditional challenges of algorithmic trading, providing a powerful, efficient, and reliable platform for quantitative traders.

## Technology Stack

NautilusTrader leverages:

*   **Python**:  Python is used extensively for strategy development, backtesting, and live trading due to its ease of use, extensive libraries, and strong community.
*   **Rust**:  Rust provides the performance and safety needed for the core components of the platform, including low-latency networking and order execution.
*   **Cython**: Cython is used to bridge the gap between Python and Rust, providing high-performance Python bindings.

## Integrations

NautilusTrader provides modular *adapters* for seamless connectivity to trading venues and data providers.

*   **Betfair:** Sports Betting Exchange
*   **Binance:** Crypto Exchange (CEX)
*   **Binance US:** Crypto Exchange (CEX)
*   **Binance Futures:** Crypto Exchange (CEX)
*   **Bybit:** Crypto Exchange (CEX)
*   **Coinbase International:** Crypto Exchange (CEX)
*   **Databento:** Data Provider
*   **dYdX:** Crypto Exchange (DEX)
*   **Interactive Brokers:** Brokerage (multi-venue)
*   **OKX:** Crypto Exchange (CEX)
*   **Polymarket:** Prediction Market (DEX)
*   **Tardis:** Crypto Data Provider

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

### Status

- `building`: Under construction and likely not in a usable state.
- `beta`: Completed to a minimally working state and in a beta testing phase.
- `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

## Versioning and Releases

**NautilusTrader is under active development.** The API is becoming more stable, but breaking changes may occur.
We aim for a bi-weekly release schedule.

### Branches

*   `master`:  Reflects the latest released version (recommended for production).
*   `nightly`: Daily snapshots of the `develop` branch for testing.
*   `develop`: The active development branch for contributions.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports two precision modes for its core value types (`Price`, `Quantity`, `Money`):

*   **High-Precision:** 128-bit integers with up to 16 decimals.
*   **Standard-Precision:** 64-bit integers with up to 9 decimals.

> [!NOTE]
>
> By default, the official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS.
> On Windows, only standard-precision (64-bit) is available due to the lack of native 128-bit integer support.
> For the Rust crates, the default is standard-precision unless you explicitly enable the `high-precision` feature flag.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

**Rust feature flag:** To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Installation of NautilusTrader via `pip` inside of a virtual environment is recommended.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

#### Stable wheels

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development wheels

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

To install a specific development wheel (e.g., `1.208.0a20241212` for December 12, 2024):

```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

#### Available versions

You can view all available versions of `nautilus_trader` on the [package index](https://packages.nautechsystems.io/simple/nautilus-trader/index.html).

To programmatically fetch and list available versions:

```bash
curl -s https://packages.nautechsystems.io/simple/nautilus-trader/index.html | grep -oP '(?<=<a href=")[^"]+(?=")' | awk -F'#' '{print $1}' | sort
```

#### Branch updates

- `develop` branch wheels (`.dev`): Build and publish continuously with every merged commit.
- `nightly` branch wheels (`a`): Build and publish daily when we automatically merge the `develop` branch at **14:00 UTC** (if there are changes).

#### Retention policies

- `develop` branch wheels (`.dev`): We retain only the most recent wheel build.
- `nightly` branch wheels (`a`): We retain only the 10 most recent wheel builds.

### From Source

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

1. Install [rustup](https://rustup.rs/) (the Rust toolchain installer).
2. Clone the source with `git`, and install from the project's root directory:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

> [!NOTE]
>
> The `--depth 1` flag fetches just the latest commit for a faster, lightweight clone.

3. Set environment variables for PyO3 compilation (Linux and macOS only).

## Redis

Redis integration is **optional** and only required if you are using it for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus). See the **Redis** section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for further details.

## Makefile

The `Makefile` simplifies building and testing.

*   `make install`: Installs with all dependencies.
*   `make build`: Runs the build script.
*   `make cargo-test`: Runs Rust crate tests.
*   `make clean`: Deletes build artifacts.
*   `make docs`: Builds documentation.
*   `make pytest`: Runs Python tests.

> [!TIP]
>
> Run `make help` for available make targets.

## Examples

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) (Python)
*   [indicator](/nautilus_trader/indicators/) (Cython)
*   [strategy](/nautilus_trader/examples/strategies/) (Python)
*   [backtest](/examples/backtest/)

## Docker

Docker images are available with the latest release, nightly, and JupyterLab environments.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

Launch the backtest example:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development guidance.

> [!TIP]
>
> Run `make build-debug` for efficient development workflow.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for Rust testing (isolated processes).

```bash
cargo install cargo-nextest
```

> [!TIP]
>
> Run Rust tests with `make cargo-test`.

## Contributing

Contributions are welcome. Please review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and [open-source scope](/ROADMAP.md#open-source-scope).  All contributions require a Contributor License Agreement (CLA).

> [!NOTE]
>
> Pull requests should target the `develop` branch (the default branch). This is where new features and improvements are integrated before release.

## Community

Join us on [Discord](https://discord.gg/NautilusTrader).

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens.
>
> All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
> or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
> If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

NautilusTrader is available under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">