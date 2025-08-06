# NautilusTrader: High-Performance Algorithmic Trading Platform

[![Codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![Codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
[![Python Versions](https://img.shields.io/pypi/pyversions/nautilus_trader)](https://img.shields.io/pypi/pyversions/nautilus_trader)
[![PyPI Version](https://img.shields.io/pypi/v/nautilus_trader)](https://img.shields.io/pypi/v/nautilus_trader)
[![PyPI Package Type](https://img.shields.io/pypi/format/nautilus_trader?color=blue)](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader empowers quantitative traders with a high-performance, open-source platform for building and deploying algorithmic trading strategies.** Dive into the world of automated trading with a platform that bridges the gap between research, backtesting, and live deployment.  See the [original repo](https://github.com/nautechsystems/nautilus_trader) for more details.

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

## Key Features of NautilusTrader

*   **High Performance:** Built with Rust for speed and efficiency, using asynchronous networking with [tokio](https://crates.io/crates/tokio).
*   **Reliability:** Leverages Rust's type and thread safety, with optional Redis-backed state persistence.
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, and Windows; supports Docker deployment.
*   **Modular Design:** Integrates with various trading venues and data providers via modular adapters.
*   **Advanced Order Types:** Supports advanced order types like IOC, FOK, GTC, GTD, DAY, AT_THE_OPEN, AT_THE_CLOSE, and conditional triggers.
*   **Customization:** Allows for user-defined components and systems built using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) and [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).
*   **Backtesting Capabilities:** Efficiently backtests strategies across multiple venues, instruments, and strategies using historical data with nanosecond resolution.
*   **Live Deployment Parity:** Uses the same strategy implementations for both backtesting and live trading.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies across multiple venues.
*   **AI-Ready:** Backtest engine suitable for training AI trading agents (RL/ES).

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png" alt="Nautilus Illustration" width="500">

## Why Choose NautilusTrader?

NautilusTrader stands out by offering a robust and high-performance environment for algorithmic trading, addressing key pain points in strategy development and deployment.  Here's why you should consider it:

*   **High-Performance Event-Driven Python:** Benefit from native binary core components for speed.
*   **Backtesting and Live Trading Parity:**  Use identical strategy code for both environments.
*   **Reduced Operational Risk:** Enhanced risk management, logical accuracy, and type safety.
*   **Highly Extensible:** Extend functionality with a message bus, custom components, actors, custom data, and custom adapters.

## Technology Stack

NautilusTrader utilizes a powerful combination of technologies to provide a reliable and performant trading platform:

*   **Rust:** Used for core performance-critical components, ensuring speed, memory safety, and concurrency.
*   **Python:** Serves as the primary language for strategy development, leveraging its extensive libraries and community support.
*   **Cython:** Bridges the gap between Python and Rust, enabling static typing and improved performance for critical sections of the code.
*   **Tokio:** Asynchronous runtime for efficient networking.

## Integrations

NautilusTrader's modular design allows for seamless integration with various trading venues and data providers through the use of *adapters*. These adapters translate the raw APIs into a unified interface and normalized domain model.

Here are the currently supported integrations:

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
| [OKX](https://okx.com)                                                       | `OKX`                 | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/beta-yellow)     | [Guide](docs/integrations/okx.md)           |
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

-   **ID**: The default client ID for the integrations adapter clients.
-   **Type**: The type of integration (often the venue type).

### Integration Status

*   `building`: Under construction and likely not in a usable state.
*   `beta`: Completed to a minimally working state and in a beta testing phase.
*   `stable`: Stabilized feature set and API; the integration has been tested by developers and users to a reasonable level (some bugs may still remain).

For more information, please consult the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation.

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule to provide regular updates, bug fixes, and new features. While the API is evolving, we aim to document breaking changes in the release notes. The project is under active development.

### Branches

*   `master`: Reflects the source code for the latest released version. This branch is recommended for production use.
*   `nightly`: Contains daily snapshots of the `develop` branch, for early testing purposes. Merged at **14:00 UTC** or on demand.
*   `develop`: This is the active development branch for contributors and feature work.

> **Roadmap:**
>
> Our [roadmap](/ROADMAP.md) aims to achieve a **stable API for version 2.x** (likely after the Rust port).
> Once this milestone is reached, we plan to implement a formal deprecation process for any API changes.
> This approach allows us to maintain a rapid development pace for now.

## Precision Mode

NautilusTrader supports two precision modes for its core value types, impacting the internal bit-width and the maximum decimal precision:

*   **High-Precision**: 128-bit integers offering up to 16 decimals of precision and a broader value range.
*   **Standard-Precision**: 64-bit integers with up to 9 decimals of precision and a smaller value range.

> **Default Settings:**
>
> By default, the official Python wheels **ship** in high-precision (128-bit) mode on Linux and macOS. On Windows, only standard-precision (64-bit) is available due to the lack of native 128-bit integer support. For the Rust crates, the default is standard-precision unless you explicitly enable the `high-precision` feature flag.

For detailed information, see the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation).

To enable high-precision mode in Rust, incorporate the `high-precision` feature to your `Cargo.toml` file:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Follow these steps to install NautilusTrader. It's recommended to use the latest supported version of Python and to install `nautilus_trader` within a virtual environment:

**Supported Installation Methods:**

1.  Pre-built binary wheel from PyPI or the Nautech Systems package index.
2.  Build from source.

> **Recommendation:**
>
> Installing using the [uv](https://docs.astral.sh/uv) package manager with a "vanilla" CPython.
> Conda and other Python distributions *may* work but aren’t officially supported.

### Install from PyPI

To install the latest binary wheel from PyPI:

```bash
pip install -U nautilus_trader
```

### Install from the Nautech Systems Package Index

The Nautech Systems package index (`packages.nautechsystems.io`) hosts stable and development binary wheels for `nautilus_trader`, compliant with [PEP-503](https://peps.python.org/pep-0503/).

#### Stable Wheels

Install the latest stable release:

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development Wheels

Install the latest pre-release:

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

You can also specify a specific development wheel:

```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

#### Available Versions

To list all available versions of `nautilus_trader`:

```bash
curl -s https://packages.nautechsystems.io/simple/nautilus-trader/index.html | grep -oP '(?<=<a href=")[^"]+(?=")' | awk -F'#' '{print $1}' | sort
```

### Install from Source

1.  Install [rustup](https://rustup.rs/) (the Rust toolchain installer):
    *   Linux and macOS:

        ```bash
        curl https://sh.rustup.rs -sSf | sh
        ```

    *   Windows:
        *   Download and install [`rustup-init.exe`](https://win.rustup.rs/x86_64)
        *   Install "Desktop development with C++" with [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)
    *   Verify (any system):
        from a terminal session run: `rustc --version`

2.  Enable `cargo` in the current shell:
    *   Linux and macOS:

        ```bash
        source $HOME/.cargo/env
        ```

    *   Windows:
        *   Start a new PowerShell

3.  Install [clang](https://clang.llvm.org/) (a C language frontend for LLVM):
    *   Linux:

        ```bash
        sudo apt-get install clang
        ```

    *   Windows:
        1.  Add Clang to your [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16):
            *   Start | Visual Studio Installer | Modify | C++ Clang tools for Windows (12.0.0 - x64…) = checked | Modify
        2.  Enable `clang` in the current shell:

            ```powershell
            [System.Environment]::SetEnvironmentVariable('path', "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\Llvm\x64\bin\;" + $env:Path,"User")
            ```

    *   Verify (any system):
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

    > The `--depth 1` flag fetches just the latest commit for a faster, lightweight clone.

6.  Set environment variables for PyO3 compilation (Linux and macOS only):

    ```bash
    # Set the library path for the Python interpreter (in this case Python 3.13.4)
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"

    # Set the Python executable path for PyO3
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for additional options.

## Redis

NautilusTrader can optionally use [Redis](https://redis.io) for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) database and the [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

The provided `Makefile` simplifies installation and build tasks. Some targets include:

*   `make install`: Installs in `release` build mode.
*   `make install-debug`: Installs in `debug` build mode.
*   `make build`: Runs the build script in `release` mode.
*   `make test-performance`: Runs performance tests with [codspeed](https://codspeed.io).
*   `make help`: Provides documentation on available targets.

## Examples

Explore example indicators and strategies developed in Python and Cython:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) example written in Python.
*   [indicator](/nautilus_trader/indicators/) examples written in Cython.
*   [strategy](/nautilus_trader/examples/strategies/) examples written in Python.
*   [backtest](/examples/backtest/) examples using a `BacktestEngine` directly.

## Docker

NautilusTrader offers Docker containers with various tags for easy deployment:

*   `nautilus_trader:latest`: Latest release version installed.
*   `nautilus_trader:nightly`: Head of the `nightly` branch installed.
*   `jupyterlab:latest`: Latest release version installed along with `jupyterlab` and an example backtest notebook.
*   `jupyterlab:nightly`: Head of the `nightly` branch installed with `jupyterlab` and an example backtest notebook.

Pull container images:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

Launch the backtest example container:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

Access the JupyterLab interface at:

```bash
http://127.0.0.1:8888/lab
```

## Development

The project provides a pleasant developer experience for Python, Cython, and Rust. See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for details.

> Run `make build-debug` for efficient development after changes to Rust or Cython code.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for Rust testing.

```bash
cargo install cargo-nextest
```

> Run Rust tests with `make cargo-test`.

## Contributing

We welcome contributions to enhance NautilusTrader.  To contribute, please:

*   Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your idea.
*   Review the [open-source scope](/ROADMAP.md#open-source-scope).
*   Follow the guidelines in the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) file.
*   Sign a Contributor License Agreement (CLA).

> Pull requests should target the `develop` branch.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) for discussions and updates.

> NautilusTrader does not endorse any cryptocurrency tokens and all official updates are shared through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader), or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).  Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems. For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128" alt="Ferris the Rustacean">