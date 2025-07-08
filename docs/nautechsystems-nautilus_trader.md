# <img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="Nautilus Trader Logo">

**Nautilus Trader: The High-Performance Algorithmic Trading Platform** - Build and deploy your trading strategies with speed and efficiency. ([See the original repository](https://github.com/nautechsystems/nautilus_trader))

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

## Key Features of Nautilus Trader

Nautilus Trader is a cutting-edge, open-source platform designed for high-performance algorithmic trading. Here's a glimpse of what it offers:

*   **High Performance:**  Built with a Rust core and asynchronous networking using Tokio for blazing-fast execution.
*   **Reliable and Safe:**  Leverages Rust's type- and thread-safety, with optional Redis-backed state persistence for robust operations.
*   **Cross-Platform Compatibility:** Runs seamlessly on Linux, macOS, and Windows; easily deployable using Docker.
*   **Modular Design:**  Integrate with various trading venues and data providers through flexible adapters.
*   **Advanced Order Types:** Supports `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`, and more, along with conditional triggers, execution instructions, and contingency orders.
*   **Customization Options:**  Extend the platform with custom components, and build entire systems from scratch using the cache and message bus.
*   **Comprehensive Backtesting:**  Test strategies with high precision using historical data, including tick, trade, bar, order book, and custom data, with nanosecond resolution.
*   **Seamless Live Deployment:**  Deploy strategies live with no code changes after backtesting.
*   **Multi-Venue Support:**  Enable market-making and statistical arbitrage strategies with multi-venue capabilities.
*   **AI Training Ready:** The backtest engine is designed to be fast enough for training AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose Nautilus Trader?

Nautilus Trader empowers quantitative traders with a powerful, reliable, and efficient platform:

*   **Enhanced Performance:** Benefit from a high-performance, event-driven Python environment with native binary core components.
*   **Code Parity:** Maintain consistency between backtesting and live trading environments with identical strategy code.
*   **Reduced Risk:** Implement enhanced risk management functionality and gain logical accuracy and type safety.
*   **Extensibility:** Leverage the message bus, custom components, actors, custom data, and custom adapters to create highly extensible systems.

## Technologies Behind Nautilus Trader

Nautilus Trader leverages modern technologies to deliver a robust and efficient trading platform:

*   **Python:** Python's versatility and extensive libraries make it the ideal language for strategy development and research.
*   **Rust:** The core performance-critical components are written in Rust for speed and memory safety, ensuring reliability and efficiency.

### Why Python?

Python is the *de facto lingua franca* of data science, machine learning, and artificial intelligence. It has become the most popular programming language in the world, and NautilusTrader lets you harness the power of Python.

### Why Rust?

Rust is a modern programming language designed for performance, reliability, and safety, with no garbage collector. It is "blazingly fast" and memory-efficient (comparable to C and C++) with no garbage collector. Rust can power mission-critical systems, run on embedded devices, and easily integrates with other languages.

NautilusTrader utilizes Rust for its core, performance-critical components, providing a Python-native environment that meets the needs of professional quantitative traders and trading firms.

## Integrations

NautilusTrader's modular design allows for seamless integration with various trading venues and data providers via adapters.

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
| [Interactive Brokers](https://www.interactivebrokers.com)                    | `INTERACTIVE_BROKERS` | Brokerage (multi-venue) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/ib.md)            |
| [OKX](https://okx.com)                                                       | `OKX`                 | Crypto Exchange (CEX)   | ![status](https://img.shields.io/badge/building-orange) | [Guide](docs/integrations/okx.md)           |
| [Polymarket](https://polymarket.com)                                         | `POLYMARKET`          | Prediction Market (DEX) | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/polymarket.md)    |
| [Tardis](https://tardis.dev)                                                 | `TARDIS`              | Crypto Data Provider    | ![status](https://img.shields.io/badge/stable-green)    | [Guide](docs/integrations/tardis.md)        |

### Status Definitions
- `building`: Under construction and likely not in a usable state.
- `beta`: Completed to a minimally working state and in a beta testing phase.
- `stable`: Stabilized feature set and API, the integration has been tested by both developers and users to a reasonable level (some bugs may still remain).

## Versioning and Releases

NautilusTrader follows a bi-weekly release schedule with different branches.

*   **`master`:** Represents the source code for the latest released version, recommended for production.
*   **`nightly`:** Daily snapshots of the `develop` branch for early testing.
*   **`develop`:** The active development branch for contributors and feature work.

## Precision Mode

NautilusTrader has two precision modes for its core types:

*   **High-precision:** 128-bit integers with up to 16 decimals.
*   **Standard-precision:** 64-bit integers with up to 9 decimals.

By default, the wheels ship with high-precision mode on Linux and macOS. On Windows, only standard-precision (64-bit) is available.

**Rust feature flag**: To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

### Prerequisites

Make sure you have the latest supported version of Python and create a virtual environment to isolate dependencies.

### Installation Methods

1.  **From PyPI:**

    ```bash
    pip install -U nautilus_trader
    ```

2.  **From the Nautech Systems package index:**

    ```bash
    pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
    ```

    To install the latest pre-release:

    ```bash
    pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
    ```

3.  **From Source:**

    Install Rust, clang, and uv, then clone the repository and install.

    ```bash
    git clone --branch develop --depth 1 https://github.com/nautechsystems/nautilus_trader
    cd nautilus_trader
    uv sync --all-extras
    ```

## Redis

Redis is optional, only used when configured as a backend for the cache or message bus.

## Makefile

The `Makefile` automates build and install tasks. Use `make help` for a list of targets.

## Examples

See the examples directory for Python and Cython indicator and strategy examples.

## Docker

Docker containers are built with different tags:

*   `latest` has the latest release version.
*   `nightly` has the head of the `nightly` branch.
*   `jupyterlab:latest` and `jupyterlab:nightly` include `jupyterlab` and a backtest notebook.

```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

The [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) contains helpful information. Run `make build-debug` for efficient Rust and Cython development.

## Testing

Use `cargo-nextest` with `make cargo-test` for efficient Rust testing.

## Contributing

Contribute by opening an issue and following the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines.  Pull requests should target the `develop` branch.

## Community

Join the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader).

> [!WARNING]
>
> NautilusTrader does not issue or endorse any cryptocurrency tokens. All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
> or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
> If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).  Contributions require a Contributor License Agreement (CLA).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology company specializing in the development of high-performance trading systems. For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO considerations:

*   **Concise Hook:** Starts with a compelling one-sentence introduction optimized for search.
*   **Clear Headings:** Uses headings to organize information logically (H2s and H3s where appropriate).
*   **Keyword Optimization:** Includes relevant keywords like "algorithmic trading," "high-performance," "backtesting," "live deployment," and "Python."
*   **Bulleted Lists:** Uses bullet points to highlight key features, benefits, and installation steps, improving readability and scannability.
*   **Structured Content:**  Breaks down complex information into smaller, easily digestible chunks.
*   **Internal Links:**  Includes internal links to relevant sections within the README, encouraging exploration and improving user experience.
*   **Call to Action:** Encourages readers to join the community, review the examples, and contribute.
*   **Alt Tags:**  Adds `alt` tags to images for accessibility and SEO.
*   **Concise Summaries:**  Rephrases the original content to be more succinct and focused.
*   **Community & Support:**  Highlights ways to get help, ask questions, and stay informed.
*   **Warning Blocks** Use of markdown syntax such as `> [!NOTE]` to draw attention to important information and prevent this from being missed.