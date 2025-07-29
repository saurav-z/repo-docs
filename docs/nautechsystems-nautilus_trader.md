# NautilusTrader: High-Performance Algorithmic Trading Platform

[![NautilusTrader Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png)](https://github.com/nautechsystems/nautilus_trader)

**NautilusTrader is an open-source, AI-first, high-performance algorithmic trading platform, enabling quantitative traders to build, backtest, and deploy automated trading strategies with unmatched speed and flexibility.** Access the original repo [here](https://github.com/nautechsystems/nautilus_trader).

[![Codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![Codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![Python Versions](https://img.shields.io/pypi/pyversions/nautilus_trader)
![PyPI Version](https://img.shields.io/pypi/v/nautilus_trader)
![PyPI Format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
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

*   **Docs**: <https://nautilustrader.io/docs/>
*   **Website**: <https://nautilustrader.io>
*   **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Core written in Rust with asynchronous networking using [tokio](https://crates.io/crates/tokio) for speed.
*   **Reliability:**  Rust-powered type- and thread-safety, with optional Redis-backed state persistence for stability.
*   **Cross-Platform Compatibility:**  OS independent, deployable on Linux, macOS, and Windows, with Docker support.
*   **Modular & Flexible:** Easy integration with any REST API or WebSocket feed through modular adapters.
*   **Advanced Order Types:** Supports  `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`,  conditional triggers, execution instructions, and contingency orders.
*   **Customizable:** Add user-defined custom components, or assemble entire systems from scratch leveraging the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Comprehensive Backtesting:**  Run backtests with nanosecond resolution using historical data, supporting multiple venues, instruments, and strategies simultaneously.
*   **Seamless Live Deployment:**  Use the exact same strategy implementations for backtesting and live trading.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies across multiple venues.
*   **AI Training Ready:** Backtest engine optimized for training AI trading agents (RL/ES).

![Nautilus Art](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png "nautilus")

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral. The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

*   **Event-Driven Python with Exceptional Performance:** Benefit from native binary core components written in Rust.
*   **Backtesting and Live Trading Parity:** Use identical strategy code for both backtesting and live deployment, reducing development time and risk.
*   **Enhanced Risk Management:** Enjoy increased risk management capabilities, accuracy, and type safety to minimize operational risk.
*   **Extensibility:** Leverage a message bus, custom components, actors, custom data, and custom adapters for a flexible trading environment.

Traditionally, trading strategy research and backtesting would be conducted in Python using vectorized methods. Then, the strategy needs to be reimplemented in a more event-driven way using C++, C#, Java, or other statically typed languages. This is because vectorized backtesting code cannot express the granular time and event-dependent complexity of real-time trading.

A key advantage of NautilusTrader is that the reimplementation step is now circumvented. The critical core components of the platform have been written entirely in [Rust](https://www.rust-lang.org/) and [Cython](https://cython.org/). This approach utilizes systems programming languages that compile performant binaries. CPython C extension modules then offer a Python-native environment suitable for professional quantitative traders and trading firms.

## Why Python?

Python's clean syntax has made it the *lingua franca* of data science, machine learning, and AI, but it has performance and typing limitations. Cython addresses these issues by introducing static typing into Python's ecosystem.

## Why Rust?

[Rust](https://www.rust-lang.org/) is a multi-paradigm programming language designed for performance and safety, especially safe concurrency. Rust is memory-efficient and can power mission-critical systems with no garbage collector. Rust’s rich type system and ownership model guarantees memory-safety and thread-safety.

The project utilizes Rust for core performance-critical components. Python bindings are implemented via Cython and [PyO3](https://pyo3.rs).

The project adheres to the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html).

> [!NOTE]
>
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader is modularly designed with *adapters* for seamless connectivity to trading venues and data providers, translating their APIs into a unified interface and normalized domain model.

The following integrations are currently supported; see [docs/integrations/](https://nautilustrader.io/docs/latest/integrations/) for details:

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

-   **ID**: The default client ID for the integrations adapter clients.
-   **Type**: The type of integration (often the venue type).

### Status

-   `building`: Under construction and likely not in a usable state.
-   `beta`: Completed to a minimally working state and in a beta testing phase.
-   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

**NautilusTrader is under active development**, with potential breaking changes. We document changes in release notes on a best-effort basis.

We aim to follow a **bi-weekly release schedule**, though experimental features may cause delays.

### Branches

We aim to maintain a stable, passing build across all branches.

-   `master`: Reflects the source code for the latest released version; recommended for production use.
-   `nightly`: Daily snapshots of the `develop` branch for early testing; merged at **14:00 UTC** or on demand.
-   `develop`: Active development branch for contributors and feature work.

> [!NOTE]
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x**.
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports two precision modes for its core value types (`Price`, `Quantity`, `Money`):

-   **High-precision**: 128-bit integers with up to 16 decimals of precision, and a larger value range.
-   **Standard-precision**: 64-bit integers with up to 9 decimals of precision, and a smaller value range.

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

We recommend installing [nautilus_trader](https://pypi.org/project/nautilus_trader/) inside a virtual environment.

**There are two supported ways to install**:

1.  Pre-built binary wheel from PyPI *or* the Nautech Systems package index.
2.  Build from source.

> [!TIP]
>
> We highly recommend installing using the [uv](https://docs.astral.sh/uv) package manager with a "vanilla" CPython.
>
> Conda and other Python distributions *may* work but aren’t officially supported.

### From PyPI

To install the latest binary wheel (or sdist package) from PyPI using Python's pip package manager:

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems Package Index

The Nautech Systems package index (`packages.nautechsystems.io`) complies with [PEP-503](https://peps.python.org/pep-0503/) and hosts both stable and development binary wheels for `nautilus_trader`.
This enables users to install either the latest stable release or pre-release versions for testing.

#### Stable Wheels

Stable wheels correspond to official releases of `nautilus_trader` on PyPI, and use standard versioning.

To install the latest stable release:

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development Wheels

Development wheels are published from both the `nightly` and `develop` branches,
allowing users to test features and fixes ahead of stable releases.

**Note**: Wheels from the `develop` branch are only built for the Linux x86_64 platform to save time
and compute resources, while `nightly` wheels support additional platforms as shown below.

| Platform           | Nightly | Develop |
| :----------------- | :------ | :------ |
| `Linux (x86_64)`   | ✓       | ✓       |
| `Linux (ARM64)`    | ✓       | -       |
| `macOS (ARM64)`    | ✓       | -       |
| `Windows (x86_64)` | ✓       | -       |

This process also helps preserve compute resources and ensures easy access to the exact binaries tested in CI pipelines,
while adhering to [PEP-440](https://peps.python.org/pep-0440/) versioning standards:

-   `develop` wheels use the version format `dev{date}+{build_number}` (e.g., `1.208.0.dev20241212+7001`).
-   `nightly` wheels use the version format `a{date}` (alpha) (e.g., `1.208.0a20241212`).

> [!WARNING]
>
> We do not recommend using development wheels in production environments, such as live trading controlling real capital.

#### Installation Commands

By default, pip will install the latest stable release. Adding the `--pre` flag ensures that pre-release versions, including development wheels, are considered.

To install the latest available pre-release (including development wheels):

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

To install a specific development wheel (e.g., `1.208.0a20241212` for December 12, 2024):

```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

#### Available Versions

You can view all available versions of `nautilus_trader` on the [package index](https://packages.nautechsystems.io/simple/nautilus-trader/index.html).

To programmatically fetch and list available versions:

```bash
curl -s https://packages.nautechsystems.io/simple/nautilus-trader/index.html | grep -oP '(?<=<a href=")[^"]+(?=")' | awk -F'#' '{print $1}' | sort
```

#### Branch Updates

-   `develop` branch wheels (`.dev`): Build and publish continuously with every merged commit.
-   `nightly` branch wheels (`a`): Build and publish daily when we automatically merge the `develop` branch at **14:00 UTC** (if there are changes).

#### Retention Policies

-   `develop` branch wheels (`.dev`): We retain only the most recent wheel build.
-   `nightly` branch wheels (`a`): We retain only the 10 most recent wheel builds.

### From Source

Install from source using pip after installing build dependencies from `pyproject.toml`:

1.  Install [rustup](https://rustup.rs/) (the Rust toolchain installer):
    -   Linux and macOS:

        ```bash
        curl https://sh.rustup.rs -sSf | sh
        ```

    -   Windows:
        -   Download and install [`rustup-init.exe`](https://win.rustup.rs/x86_64)
        -   Install "Desktop development with C++" with [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)
    -   Verify (any system):
        from a terminal session run: `rustc --version`

2.  Enable `cargo` in the current shell:
    -   Linux and macOS:

        ```bash
        source $HOME/.cargo/env
        ```

    -   Windows:
        -   Start a new PowerShell

3.  Install [clang](https://clang.llvm.org/) (a C language frontend for LLVM):
    -   Linux:

        ```bash
        sudo apt-get install clang
        ```

    -   Windows:
        1.  Add Clang to your [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16):
            -   Start | Visual Studio Installer | Modify | C++ Clang tools for Windows (12.0.0 - x64…) = checked | Modify
        2.  Enable `clang` in the current shell:

            ```powershell
            [System.Environment]::SetEnvironmentVariable('path', "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\Llvm\x64\bin\;" + $env:Path,"User")
            ```

    -   Verify (any system):
        from a terminal session run: `clang --version`

4.  Install uv (see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation) for more details):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

5.  Clone the source with `git`, and install from the project's root directory:

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

Using [Redis](https://redis.io) with NautilusTrader is **optional** and only required if configured as the backend for a
[cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
See the **Redis** section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for further details.

## Makefile

A `Makefile` automates installation and build tasks for development. Some targets include:

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

Indicators and strategies can be developed in both Python and Cython. For performance-critical applications, we recommend using Cython. Examples include:

-   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
-   [indicator](/nautilus_trader/indicators/) examples written in Cython.
-   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
-   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Docker containers are built using the base image `python:3.12-slim` with the following variant tags:

-   `nautilus_trader:latest` has the latest release version installed.
-   `nautilus_trader:nightly` has the head of the `nightly` branch installed.
-   `jupyterlab:latest` has the latest release version installed along with `jupyterlab` and an example backtest notebook with accompanying data.
-   `jupyterlab:nightly` has the head of the `nightly` branch installed along with `jupyterlab` and an example backtest notebook with accompanying data.

Pull the container images:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

Launch the backtest example container:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Open your browser at:

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
> -   <https://github.com/jupyterlab/jupyterlab/issues/12845>
> -   <https://github.com/deshaw/jupyterlab-limit-output>

## Development

We aim to provide the best developer experience. See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html).

> [!TIP]
>
> Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

### Testing with Rust

[cargo-nextest](https://nexte.st) is the standard Rust test runner for NautilusTrader. Its key benefit is isolating each test.

Install cargo-nextest:

```bash
cargo install cargo-nextest
```

> [!TIP]
>
> Run Rust tests with `make cargo-test`.

## Contributing

Contributions are welcome!  Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your idea. Follow the guidelines in the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) and sign the [CLA](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

> [!NOTE]
>
> Pull requests should target the `develop` branch.

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

NautilusTrader is available under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology company. Visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![Nautech Systems Logo](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">