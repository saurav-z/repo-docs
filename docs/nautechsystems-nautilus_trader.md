# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader is an open-source, AI-first algorithmic trading platform designed for high-performance backtesting and live deployment of automated trading strategies.** Explore the power of a unified environment for quantitative traders.

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

**Key Features:**

*   **High Performance:** Built with Rust for speed and efficiency.
*   **Reliable & Safe:** Leverage Rust's type and thread safety.
*   **Cross-Platform:** Supports Linux, macOS, and Windows.
*   **Modular:** Integrate any REST API or WebSocket feed with modular adapters.
*   **Advanced Order Types:** Supports IOC, FOK, GTC, GTD, DAY, AT_THE_OPEN, AT_THE_CLOSE, conditional triggers, and more. Includes post-only, reduce-only, iceberg execution instructions, and OCO, OUO, OTO contingency orders.
*   **Customizable:** Build your own components and systems leveraging cache and message bus.
*   **Backtesting:** Run strategies simultaneously with historical data using nanosecond resolution.
*   **Seamless Live Deployment:** Utilize the same strategy code for backtesting and live trading.
*   **Multi-Venue:** Facilitate market making and statistical arbitrage strategies.
*   **AI Training Ready:** Backtest engine is fast enough to train AI trading agents (RL/ES).

**For more details, see the original repository: [https://github.com/nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader)**

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

-   **Docs:** <https://nautilustrader.io/docs/>
-   **Website:** <https://nautilustrader.io>
-   **Support:** [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Introduction

NautilusTrader offers a high-performance solution for quantitative traders, enabling backtesting, and live deployment with identical code. The platform prioritizes software correctness and safety, supporting Python-native workloads. It is asset-class-agnostic and integrates via modular adapters to support high-frequency trading across a wide range of instruments.

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why NautilusTrader?

*   **High-Performance Python:** Leverages native binary core components.
*   **Backtesting and Live Parity:** Uses identical strategy code for both environments.
*   **Reduced Operational Risk:** Enhanced risk management and type safety.
*   **Extensible:** Offers a message bus, custom components and actors, custom data, and adapters.

## Why Python?

Python is the *lingua franca* for data science and AI. It has a clean syntax and rich ecosystem of libraries and communities.

## Why Rust?

Rust offers performance and safety, especially with concurrency, with no garbage collection. It provides a rich type system and memory and thread-safety.

The project follows the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html).

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader uses modular *adapters* for connecting to trading venues and data providers, with a unified interface and normalized domain model.

See [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details.

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

-   **ID:** The default client ID for the integrations adapter clients.
-   **Type:** The type of integration (often the venue type).

### Status

-   `building`: Under construction
-   `beta`: Minimally working and in beta testing
-   `stable`: Stabilized feature set and API, tested by developers and users

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) for more details.

## Versioning and Releases

**NautilusTrader is under active development.** We aim for a **bi-weekly release schedule**.

### Branches

-   `master`: Latest released version (recommended for production).
-   `nightly`: Daily snapshots of `develop` for testing.
-   `develop`: Active development (for contributors).

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x**.

## Precision Mode

NautilusTrader supports both high-precision (128-bit) and standard-precision (64-bit) modes.

> [!NOTE]
>
> By default, the official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS.
> On Windows, only standard-precision (64-bit) is available due to the lack of native 128-bit integer support.
> For the Rust crates, the default is standard-precision unless you explicitly enable the `high-precision` feature flag.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for more details.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Recommended: Use the latest Python and install `nautilus_trader` inside a virtual environment.

**Two supported methods:**

1.  Pre-built binary wheel from PyPI or the Nautech Systems package index.
2.  Build from source.

> [!TIP]
>
> We highly recommend installing using the [uv](https://docs.astral.sh/uv) package manager with a "vanilla" CPython.
>
> Conda and other Python distributions *may* work but aren’t officially supported.

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development wheels

To install the latest available pre-release (including development wheels):

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

To install a specific development wheel:

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

-   `develop` branch wheels (`.dev`): Build and publish continuously with every merged commit.
-   `nightly` branch wheels (`a`): Build and publish daily when we automatically merge the `develop` branch at **14:00 UTC** (if there are changes).

#### Retention policies

-   `develop` branch wheels (`.dev`): We retain only the most recent wheel build.
-   `nightly` branch wheels (`a`): We retain only the 10 most recent wheel builds.

### From Source

1.  Install [rustup](https://rustup.rs/).
2.  Enable `cargo`.
3.  Install [clang](https://clang.llvm.org/).
4.  Install [uv](https://docs.astral.sh/uv/getting-started/installation).
5.  Clone the source and install:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

> [!NOTE]
>
> The `--depth 1` flag fetches just the latest commit for a faster, lightweight clone.

6.  Set environment variables for PyO3 compilation (Linux and macOS only):

    ```bash
    # Set the library path for the Python interpreter (in this case Python 3.13.4)
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"

    # Set the Python executable path for PyO3
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

> [!NOTE]
>
> Adjust the Python version and architecture in the `LD_LIBRARY_PATH` to match your system.
> Use `uv python list` to find the exact path for your Python installation.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for other options and further details.

## Redis

Redis is **optional** and required for [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus) backends.

## Makefile

The `Makefile` automates installation and build tasks.

-   `make install`: Installs in `release` build mode with all dependency groups and extras.
-   `make install-debug`: Same as `make install` but with `debug` build mode.
-   `make install-just-deps`: Installs just the `main`, `dev` and `test` dependencies (does not install package).
-   `make build`: Runs the build script in `release` build mode (default).
-   `make build-debug`: Runs the build script in `debug` build mode.
-   `make build-wheel`: Runs uv build with a wheel format in `release` mode.
-   `make build-wheel-debug`: Runs uv build with a wheel format in `debug` mode.
-   `make cargo-test`: Runs all Rust crate tests using `cargo-nextest`.
-   `make clean`: Deletes all build results, such as `.so` or `.dll` files.
-   `make distclean`: **CAUTION** Removes all artifacts not in the git index from the repository. This includes source files which have not been `git add`ed.
-   `make docs`: Builds the documentation HTML using Sphinx.
-   `make pre-commit`: Runs the pre-commit checks over all files.
-   `make ruff`: Runs ruff over all files using the `pyproject.toml` config (with autofix).
-   `make pytest`: Runs all tests with `pytest`.
-   `make test-performance`: Runs performance tests with [codspeed](https://codspeed.io).

> [!TIP]
>
> Run `make help` for documentation on all available make targets.

> [!TIP]
>
> See the [crates/infrastructure/TESTS.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/crates/infrastructure/TESTS.md) file for running the infrastructure integration tests.

## Examples

Indicators and strategies can be developed in Python and Cython.

-   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
-   [indicator](/nautilus_trader/indicators/) examples written in Cython.
-   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
-   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Docker containers are built using the base image `python:3.12-slim`.

*   `nautilus_trader:latest`: Latest release version.
*   `nautilus_trader:nightly`: Head of the `nightly` branch.
*   `jupyterlab:latest`: Latest release version with `jupyterlab`.
*   `jupyterlab:nightly`: Head of the `nightly` branch with `jupyterlab`.

Pull the container images:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

Launch the backtest example container:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Open your browser:

```bash
http://127.0.0.1:8888/lab
```

> [!WARNING]
>
> NautilusTrader currently exceeds the rate limit for Jupyter notebook logging (stdout output).
> Therefore, we set the `log_level` to `ERROR` in the examples. Lowering this level to see more
> logging will cause the notebook to hang during cell execution. We are investigating a fix that
> may involve either raising the configured rate limits for Jupyter or throttling the log flushing
> from Nautilus.
>
> - <https://github.com/jupyterlab/jupyterlab/issues/12845>
> - <https://github.com/deshaw/jupyterlab-limit-output>

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html).

> [!TIP]
>
> Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

### Testing with Rust

[cargo-nextest](https://nexte.st) is the standard Rust test runner.

Install:

```bash
cargo install cargo-nextest
```

> [!TIP]
>
> Run Rust tests with `make cargo-test`.

## Contributing

Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss enhancements or bug fixes.

Follow the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines.

> [!NOTE]
>
> Pull requests target the `develop` branch.

## Community

Join our [Discord](https://discord.gg/NautilusTrader).

> [!WARNING]
>
> NautilusTrader does not issue or endorse any cryptocurrency tokens.
>
> All official updates are on <https://nautilustrader.io>, [Discord](https://discord.gg/NautilusTrader), and [@NautilusTrader](https://x.com/NautilusTrader).

## License

[GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). Requires a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems. Visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">