# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader empowers quantitative traders with a high-performance, open-source platform for backtesting and live deployment of algorithmic trading strategies.** Dive deeper and explore the project on [GitHub](https://github.com/nautechsystems/nautilus_trader).

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

*   **High Performance:** Built with a Rust core for speed and efficiency.
*   **Reliability:** Leverages Rust's memory safety and thread safety.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows, with Docker support.
*   **Modular Design:** Integrates with various trading venues and data providers through adapters.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, and more, plus conditional triggers and contingency orders.
*   **Customizable:** Allows for user-defined components and system assembly with cache and message bus.
*   **Backtesting:** Enables simultaneous backtesting across multiple venues, instruments, and strategies with nanosecond resolution.
*   **Live Trading:** Utilizes identical strategy implementations for both backtesting and live deployment.
*   **Multi-Venue Support:** Facilitates market-making and arbitrage strategies.
*   **AI-Ready:** The fast backtesting engine is ideal for training AI trading agents.

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png" alt="nautilus-trader">

## Introduction

NautilusTrader is a production-grade algorithmic trading platform designed for quantitative traders. It provides a robust environment for backtesting strategies on historical data and deploying them live without code changes. The platform prioritizes software correctness and safety, aiming to support Python-native, mission-critical trading system workloads. NautilusTrader supports high-frequency trading across various asset classes and instrument types.

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png" alt="nautilus">

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral.
> The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

*   **High-Performance Python:** Benefit from native binary core components.
*   **Backtesting & Live Parity:** Use the same strategy code for backtesting and live trading.
*   **Reduced Operational Risk:** Enhanced risk management functionality and type safety.
*   **Extensible:** Leverage the message bus, custom components, and adapters.

NautilusTrader addresses the need to bridge the gap between backtesting in Python and live trading in compiled languages. By using Rust and Cython for core components, the platform provides the performance and safety required for professional trading while maintaining a Python-native environment.

## Why Python?

Python is the *de facto lingua franca* of data science, machine learning, and artificial intelligence.

## Why Rust?

[Rust](https://www.rust-lang.org/) is a multi-paradigm programming language designed for performance and safety, especially safe concurrency. Rust is "blazingly fast" and memory-efficient (comparable to C and C++) with no garbage collector. It can power mission-critical systems, run on embedded devices, and easily integrates with other languages.

## Integrations

NautilusTrader uses *adapters* to connect to trading venues and data providers.

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

### Status

-   `building`: Under construction
-   `beta`: Minimally working, in beta testing
-   `stable`: Stabilized feature set and API.

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

**NautilusTrader is actively developed and follows a bi-weekly release schedule.**

### Branches

-   `master`: Latest released version (recommended for production).
-   `nightly`: Daily snapshots of the `develop` branch.
-   `develop`: Active development branch.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports high-precision and standard-precision modes for its core value types.

-   **High-precision**: 128-bit integers.
-   **Standard-precision**: 64-bit integers.

> [!NOTE]
>
> By default, the official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS.
> On Windows, only standard-precision (64-bit) is available due to the lack of native 128-bit integer support.
> For the Rust crates, the default is standard-precision unless you explicitly enable the `high-precision` feature flag.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

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

### From Source

1.  Install [rustup](https://rustup.rs/) and build tools.
2.  Enable `cargo` in the current shell.
3.  Install [clang](https://clang.llvm.org/).
4.  Install uv:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

5.  Clone and install from source:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

6.  Set environment variables for PyO3 compilation (Linux and macOS only):

    ```bash
    # Set the library path for the Python interpreter (in this case Python 3.13.4)
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"

    # Set the Python executable path for PyO3
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for other options and further details.

## Redis

Redis is optional for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

See the **Redis** section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for further details.

## Makefile

A `Makefile` is provided to automate most installation and build tasks for development. It provides the following targets:

- `make install`: Installs in `release` build mode with all dependency groups and extras.
- `make install-debug`: Same as `make install` but with `debug` build mode.
- `make install-just-deps`: Installs just the `main`, `dev` and `test` dependencies (does not install package).
- `make build`: Runs the build script in `release` build mode (default).
- `make build-debug`: Runs the build script in `debug` build mode.
- `make build-wheel`: Runs uv build with a wheel format in `release` mode.
- `make build-wheel-debug`: Runs uv build with a wheel format in `debug` mode.
- `make clean`: Deletes all build results, such as `.so` or `.dll` files.
- `make distclean`: **CAUTION** Removes all artifacts not in the git index from the repository. This includes source files which have not been `git add`ed.
- `make docs`: Builds the documentation HTML using Sphinx.
- `make pre-commit`: Runs the pre-commit checks over all files.
- `make ruff`: Runs ruff over all files using the `pyproject.toml` config (with autofix).
- `make pytest`: Runs all tests with `pytest`.
- `make test-performance`: Runs performance tests with [codspeed](https://codspeed.io).

> [!TIP]
>
> Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

## Examples

- [indicator](/nautilus_trader/examples/indicators/ema_python.py)
- [indicator](/nautilus_trader/indicators/)
- [strategy](/nautilus_trader/examples/strategies/)
- [backtest](/examples/backtest/)

## Docker

Docker containers are built using the base image `python:3.12-slim` with the following variant tags:

- `nautilus_trader:latest` has the latest release version installed.
- `nautilus_trader:nightly` has the head of the `nightly` branch installed.
- `jupyterlab:latest` has the latest release version installed along with `jupyterlab` and an
  example backtest notebook with accompanying data.
- `jupyterlab:nightly` has the head of the `nightly` branch installed along with `jupyterlab` and an
  example backtest notebook with accompanying data.

You can pull the container images as follows:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

You can launch the backtest example container by running:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Then open your browser at the following address:

```bash
http://127.0.0.1:8888/lab
```

> [!WARNING]
>
> NautilusTrader currently exceeds the rate limit for Jupyter notebook logging (stdout output).
> As a result, the `log_level` in the examples is set to `ERROR`. Lowering this level to see more
> logging will cause the notebook to hang during cell execution. We are investigating a fix, which
> may involve either raising the configured rate limits for Jupyter or throttling the log flushing
> from Nautilus.
>
> - <https://github.com/jupyterlab/jupyterlab/issues/12845>
> - <https://github.com/deshaw/jupyterlab-limit-output>

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for helpful information.

### Testing with Rust

Run Rust tests with `make cargo-test`.

## Contributing

Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) on GitHub to discuss it with the team. See the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and [open-source scope](/ROADMAP.md#open-source-scope).

## Community

Join our community on [Discord](https://discord.gg/NautilusTrader).

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens. Any claims or communications suggesting otherwise are unauthorized and false.
>
> All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
> or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
> If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

The source code for NautilusTrader is available under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).
Contributions require the completion of a standard [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology
company specializing in the development of high-performance trading systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">