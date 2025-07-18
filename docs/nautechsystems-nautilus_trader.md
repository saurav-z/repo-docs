# NautilusTrader: High-Performance Algorithmic Trading Platform

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

NautilusTrader is an open-source, AI-first, production-grade algorithmic trading platform empowering quantitative traders to build, backtest, and deploy high-performance trading strategies. Access the original repo [here](https://github.com/nautechsystems/nautilus_trader).

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
*   **Reliable & Safe:** Leverages Rust's type and thread safety.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows.
*   **Modular Design:** Integrates with any REST API or WebSocket feed.
*   **Advanced Order Types:** Supports various order types and conditional triggers.
*   **Backtesting & Live Trading:** Use the same strategy code for both backtesting and live deployment.
*   **Multi-Venue Support:** Facilitates market-making and arbitrage strategies across multiple venues.
*   **AI Ready:** Optimized for training AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

NautilusTrader offers a powerful and flexible platform for quantitative traders by providing:

*   **High-Performance Python:** Native binary core components for fast execution.
*   **Code Parity:** Identical strategy code for backtesting and live trading.
*   **Enhanced Reliability:** Improved risk management and type safety.
*   **Extensibility:** Message bus, custom components, and data integration capabilities.

## Technical Details

NautilusTrader uses a combination of Rust and Python to deliver a high-performance trading platform.  The core components are written in Rust for speed and safety, with Python bindings created via Cython and PyO3. This architecture allows for a Python-native environment that meets the demands of professional quantitative traders.  The platform's design emphasizes software correctness and safety for mission-critical trading workloads.

## Integrations

NautilusTrader's modular design enables seamless integration with various trading venues and data providers through adapters.

### Supported Integrations

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

- **ID**: The default client ID for the integrations adapter clients.
- **Type**: The type of integration (often the venue type).

### Status

-   `building`: Under construction and likely not in a usable state.
-   `beta`: Completed to a minimally working state and in a beta testing phase.
-   `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Installation

NautilusTrader can be installed from PyPI or built from source.

### Installation Guide

We recommend using the latest supported version of Python and installing [nautilus_trader](https://pypi.org/project/nautilus_trader/) inside a virtual environment to isolate dependencies.

**There are two supported ways to install**:

1.  Pre-built binary wheel from PyPI *or* the Nautech Systems package index.
2.  Build from source.

>   [!TIP]
>
>   We highly recommend installing using the [uv](https://docs.astral.sh/uv) package manager with a "vanilla" CPython.
>
>   Conda and other Python distributions *may* work but aren’t officially supported.

### From PyPI

To install the latest binary wheel (or sdist package) from PyPI using Python's pip package manager:

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems package index

The Nautech Systems package index (`packages.nautechsystems.io`) complies with [PEP-503](https://peps.python.org/pep-0503/) and hosts both stable and development binary wheels for `nautilus_trader`.
This enables users to install either the latest stable release or pre-release versions for testing.

#### Stable wheels

Stable wheels correspond to official releases of `nautilus_trader` on PyPI, and use standard versioning.

To install the latest stable release:

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development wheels

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

>   [!WARNING]
>
>   We do not recommend using development wheels in production environments, such as live trading controlling real capital.

#### Installation commands

By default, pip will install the latest stable release. Adding the `--pre` flag ensures that pre-release versions, including development wheels, are considered.

To install the latest available pre-release (including development wheels):

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

-   `develop` branch wheels (`.dev`): Build and publish continuously with every merged commit.
-   `nightly` branch wheels (`a`): Build and publish daily when we automatically merge the `develop` branch at **14:00 UTC** (if there are changes).

#### Retention policies

-   `develop` branch wheels (`.dev`): We retain only the most recent wheel build.
-   `nightly` branch wheels (`a`): We retain only the 10 most recent wheel builds.

### From Source

It's possible to install from source using pip if you first install the build dependencies as specified in the `pyproject.toml`.

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

>   [!NOTE]
>
>   The `--depth 1` flag fetches just the latest commit for a faster, lightweight clone.

6.  Set environment variables for PyO3 compilation (Linux and macOS only):

    ```bash
    # Set the library path for the Python interpreter (in this case Python 3.13.4)
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"

    # Set the Python executable path for PyO3
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

>   [!NOTE]
>
>   Adjust the Python version and architecture in the `LD_LIBRARY_PATH` to match your system.
>   Use `uv python list` to find the exact path for your Python installation.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for other options and further details.

## Precision Mode

NautilusTrader supports two precision modes for its core value types (`Price`, `Quantity`, `Money`), which differ in their internal bit-width and maximum decimal precision.

-   **High-precision**: 128-bit integers with up to 16 decimals of precision, and a larger value range.
-   **Standard-precision**: 64-bit integers with up to 9 decimals of precision, and a smaller value range.

>   [!NOTE]
>
>   By default, the official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS.
>   On Windows, only standard-precision (64-bit) is available due to the lack of native 128-bit integer support.
>   For the Rust crates, the default is standard-precision unless you explicitly enable the `high-precision` feature flag.

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for further details.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Optional Components

### Redis

Using [Redis](https://redis.io) with NautilusTrader is **optional** and only required if configured as the backend for a
[cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
See the **Redis** section of the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation#redis) for further details.

### Makefile

A `Makefile` is provided to automate most installation and build tasks for development. Some of the targets include:

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

>   [!TIP]
>
>   Run `make help` for documentation on all available make targets.

>   [!TIP]
>
>   See the [crates/infrastructure/TESTS.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/crates/infrastructure/TESTS.md) file for running the infrastructure integration tests.

## Examples

Discover how to develop indicators and strategies in Python and Cython.

-   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
-   [indicator](/nautilus_trader/indicators/) examples written in Cython.
-   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
-   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

Pre-built Docker containers are available.

*   `nautilus_trader:latest`: Installs the latest release version.
*   `nautilus_trader:nightly`: Installs the head of the `nightly` branch.
*   `jupyterlab:latest`: Latest release with JupyterLab and example backtest notebook.
*   `jupyterlab:nightly`: Nightly build with JupyterLab and example backtest notebook.

Pull the container images:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

Run the backtest example container:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Then open your browser at the following address:

```bash
http://127.0.0.1:8888/lab
```

>   [!WARNING]
>
>   NautilusTrader currently exceeds the rate limit for Jupyter notebook logging (stdout output).
>   Therefore, we set the `log_level` to `ERROR` in the examples. Lowering this level to see more
>   logging will cause the notebook to hang during cell execution. We are investigating a fix that
>   may involve either raising the configured rate limits for Jupyter or throttling the log flushing
>   from Nautilus.
>
>   -   <https://github.com/jupyterlab/jupyterlab/issues/12845>
>   -   <https://github.com/deshaw/jupyterlab-limit-output>

## Development

Learn how to contribute and get the most out of NautilusTrader with our [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html).

>   [!TIP]
>
>   Run `make build-debug` to compile after changes to Rust or Cython code for the most efficient development workflow.

### Testing with Rust

[cargo-nextest](https://nexte.st) is the standard Rust test runner for NautilusTrader.
Its key benefit is isolating each test in its own process, ensuring test reliability
by avoiding interference.

You can install cargo-nextest by running:

```bash
cargo install cargo-nextest
```

>   [!TIP]
>
>   Run Rust tests with `make cargo-test`, which uses **cargo-nextest** with an efficient profile.

## Contributing

We welcome contributions!

*   Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your ideas.
*   Review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file for guidelines.
*   Pull requests should target the `develop` branch.

## Community

Join our [Discord](https://discord.gg/NautilusTrader) to connect with other users and contributors.

>   [!WARNING]
>
>   NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens. Any claims or communications suggesting otherwise are unauthorized and false.
>
>   All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
>   or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
>   If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology
company specializing in the development of high-performance trading systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">