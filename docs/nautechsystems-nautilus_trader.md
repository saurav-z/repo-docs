# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**NautilusTrader empowers quantitative traders with a robust, open-source platform for backtesting and deploying algorithmic trading strategies.**

[View the original repository on GitHub](https://github.com/nautechsystems/nautilus_trader)

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

## Key Features of NautilusTrader

*   **High Performance:** Core components written in Rust for speed and efficiency.
*   **Reliable & Safe:** Rust-powered with type and thread safety, optional Redis persistence.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows; deploy with Docker.
*   **Modular Design:** Integrate any REST API or WebSocket feed with modular adapters.
*   **Advanced Order Types:** Includes IOC, FOK, GTC, GTD, DAY, and more, plus advanced triggers and contingency orders.
*   **Customizable:** Add custom components and assemble systems using cache and message bus.
*   **Backtesting & Live Trading Parity:** Use the same strategy code for backtesting and live deployments.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   **AI-Ready:** Backtest engine is optimized for training AI trading agents (RL/ES).

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-art.png" alt="Nautilus Art">

> *nautilus - from ancient Greek 'sailor' and naus 'ship'.*
>
> *The nautilus shell consists of modular chambers with a growth factor which approximates a logarithmic spiral.
> The idea is that this can be translated to the aesthetics of design and architecture.*

## Why Choose NautilusTrader?

NautilusTrader provides a powerful Python-native environment for building, testing, and deploying algorithmic trading strategies. Key benefits include:

*   **High-Performance Event-Driven Python:** Leverages native binary components for speed.
*   **Consistent Backtesting and Live Trading:** Execute the same strategy code in both environments.
*   **Reduced Operational Risk:** Enhanced risk management, accuracy, and type safety.
*   **Extensible Platform:** Supports message bus, custom components, data, and adapters.

## Technology Behind NautilusTrader

NautilusTrader addresses the challenge of bridging the gap between Python research/backtesting environments and production trading systems.

### Python for Quantitative Research

Python is the *lingua franca* of data science and AI. It is an ideal language for research, but can sometimes be limited by its performance and typing for large-scale trading systems. Cython is a helpful tool for addressing these limitations.

### Rust for Performance and Safety

NautilusTrader's core components are written in Rust for blazing-fast performance, memory efficiency, and thread safety. Rust's strong typing and ownership model eliminate many bugs at compile-time.  The project is committed to the [Soundness Pledge](https://raphlinus.github.io/rust/2020/01/18/soundness-pledge.html).

The project increasingly utilizes Rust for core performance-critical components. Python bindings are implemented via Cython and [PyO3](https://pyo3.rs)—no Rust toolchain is required at install time.

> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader uses a modular adapter design to connect to trading venues and data providers.

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

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

### Status Key

*   `building`: Under construction, not fully usable.
*   `beta`: Minimally working, in beta testing.
*   `stable`: Stabilized feature set and API, reasonably tested.

## Versioning and Releases

*   NautilusTrader is under active development; breaking changes may occur between releases.
*   We aim for a **bi-weekly release schedule**.
*   `master`: Latest released version, recommended for production.
*   `nightly`: Daily snapshots of the `develop` branch, for early testing.
*   `develop`: Active development branch.
*   Roadmap aims for a **stable API for version 2.x**.

## Precision Mode

NautilusTrader supports two precision modes for its core value types:

*   **High-precision:** 128-bit integers with up to 16 decimals of precision.
*   **Standard-precision:** 64-bit integers with up to 9 decimals of precision.

By default, the official Python wheels ship in high-precision (128-bit) mode on Linux and macOS.
On Windows, only standard-precision (64-bit) is available due to the lack of native 128-bit integer support.
For the Rust crates, the default is standard-precision unless you explicitly enable the `high-precision` feature flag.

**Rust Feature Flag:** To enable high-precision in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Follow these steps to install NautilusTrader:

### From PyPI

```bash
pip install -U nautilus_trader
```

### From the Nautech Systems Package Index

#### Stable Wheels

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```

#### Development Wheels

```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

To install a specific development wheel (e.g., `1.208.0a20241212` for December 12, 2024):

```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

### From Source

1.  Install [rustup](https://rustup.rs/) (the Rust toolchain installer) and enable `cargo`:
    *   Linux and macOS:

        ```bash
        curl https://sh.rustup.rs -sSf | sh
        source $HOME/.cargo/env
        ```

    *   Windows:
        1.  Download and install [`rustup-init.exe`](https://win.rustup.rs/x86_64)
        2.  Install "Desktop development with C++" with [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)

2.  Install [clang](https://clang.llvm.org/) and enable it:

    *   Linux:

        ```bash
        sudo apt-get install clang
        ```

    *   Windows:
        1.  Add Clang to your [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16):
            - Start | Visual Studio Installer | Modify | C++ Clang tools for Windows (12.0.0 - x64…) = checked | Modify
        2.  Enable `clang` in the current shell:

            ```powershell
            [System.Environment]::SetEnvironmentVariable('path', "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\Llvm\x64\bin\;" + $env:Path,"User")
            ```

3.  Install [uv](https://docs.astral.sh/uv/getting-started/installation):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

4.  Clone the source code and install:

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

5.  Set environment variables for PyO3 compilation (Linux and macOS only):

    ```bash
    # Set the library path for the Python interpreter (in this case Python 3.13.4)
    export LD_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.13.4-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH"

    # Set the Python executable path for PyO3
    export PYO3_PYTHON=$(pwd)/.venv/bin/python
    ```

## Redis

Redis is optional. It is only required when configured as the backend for the [cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

The `Makefile` automates installation and build tasks:

*   `make install`: Installs with all dependencies.
*   `make install-debug`: Installs in debug mode.
*   `make build`: Runs the build script in `release` build mode.
*   `make build-wheel`: Builds a wheel.
*   `make cargo-test`: Runs Rust crate tests.
*   `make clean`: Deletes build results.
*   `make distclean`: Removes all artifacts not in the git index.
*   `make docs`: Builds the documentation.
*   `make pre-commit`: Runs pre-commit checks.
*   `make ruff`: Runs ruff.
*   `make pytest`: Runs pytest.
*   `make test-performance`: Runs performance tests with [codspeed](https://codspeed.io).

## Examples

Explore trading strategies and indicators in Python and Cython:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py) (Python)
*   [indicator](/nautilus_trader/indicators/) (Cython)
*   [strategy](/nautilus_trader/examples/strategies/) (Python)
*   [backtest](/examples/backtest/)

## Docker

Pre-built Docker containers are available:

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```

Launch the backtest example container:

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

See the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development workflow details.
Run `make build-debug` for efficient development.

### Testing with Rust

Use [cargo-nextest](https://nexte.st) for reliable Rust testing.

```bash
cargo install cargo-nextest
make cargo-test
```

## Contributing

Contributions are welcome!

1.  Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your idea.
2.  Follow the guidelines in [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md).
3.  Target the `develop` branch.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) to discuss, learn, and get updates.

> **Important Safety Notice:** NautilusTrader does not endorse any cryptocurrency tokens. Report any suspicious activity to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).
Contributions require a [Contributor License Agreement (CLA)](https://github.com/nautechsystems/nautilus_trader/blob/develop/CLA.md).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">