# NautilusTrader: High-Performance Algorithmic Trading Platform

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader-logo.png" width="500" alt="NautilusTrader Logo">

**Unlock the power of high-performance, Python-native algorithmic trading with NautilusTrader!** Dive into a robust and versatile platform designed for both backtesting and live deployment.

[![codecov](https://codecov.io/gh/nautechsystems/nautilus_trader/branch/master/graph/badge.svg?token=DXO9QQI40H)](https://codecov.io/gh/nautechsystems/nautilus_trader)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/nautechsystems/nautilus_trader)
![pythons](https://img.shields.io/pypi/pyversions/nautilus_trader)
![pypi-version](https://img.shields.io/pypi/v/nautilus_trader)
![pypi-format](https://img.shields.io/pypi/format/nautilus_trader?color=blue)
[![Downloads](https://pepy.tech/badge/nautilus-trader)](https://pepy.tech/project/nautilus-trader)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/NautilusTrader)

**Key Features:**

*   **High-Performance:** Built with Rust for speed and efficiency.
*   **Reliable & Safe:** Leverages Rust's type and thread safety.
*   **Backtesting & Live Trading:** Use the same strategy code for both.
*   **Modular & Flexible:** Easily integrate with any API via modular adapters.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows.
*   **AI-First:** Designed for developing and deploying algorithmic trading strategies within a performant Python-native environment.
*   **Multi-Venue Support:** Facilitates market-making and statistical arbitrage strategies.

**Get started today:** [Explore the NautilusTrader Repository](https://github.com/nautechsystems/nautilus_trader)

---

## Introduction

NautilusTrader is an open-source, production-grade algorithmic trading platform designed for quantitative traders. It empowers you to:

*   Backtest automated trading strategies on historical data using an event-driven engine.
*   Deploy those same strategies live, without code modifications.

The platform prioritizes correctness, safety, and performance, providing a robust Python-native environment for mission-critical trading system workloads. It is asset-class-agnostic and supports high-frequency trading across FX, equities, futures, options, crypto, and betting.

<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png" width="500" alt="NautilusTrader in Action">

## Why Choose NautilusTrader?

*   **Blazing-fast execution:** Core components written in Rust.
*   **Code Parity:** Identical strategy code for backtesting and live trading, eliminating reimplementation.
*   **Reduced Risk:** Enhanced risk management and type safety for more reliable trading.
*   **Highly Extensible:** Customize the platform with custom components, actors, data, and adapters.

## Technical Highlights

NautilusTrader's architecture combines the strengths of both Python and Rust:

*   **Python:** Provides a familiar environment for strategy development, benefiting from the vast ecosystem of data science and AI libraries.
*   **Rust:** Powers the performance-critical components, ensuring speed, memory safety, and concurrency.

> [!NOTE]
> **MSRV:** NautilusTrader relies heavily on improvements in the Rust language and compiler.
> As a result, the Minimum Supported Rust Version (MSRV) is generally equal to the latest stable release of Rust.

## Integrations

NautilusTrader offers modular *adapters* for seamless integration with various trading venues and data providers.

**Currently Supported Integrations:**

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

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

### Status Definitions

*   `building`: Under construction.
*   `beta`: Minimally working and in beta testing.
*   `stable`: Stabilized feature set and API, tested by developers and users.

## Versioning and Releases

NautilusTrader follows a **bi-weekly release schedule**.

*   `master`: Reflects the latest released version; recommended for production use.
*   `nightly`: Daily snapshots of the `develop` branch.
*   `develop`: Active development branch.

> [!NOTE]
> A **stable API for version 2.x** (likely after the Rust port) is targeted.

## Precision Mode

NautilusTrader offers two precision modes:

*   **High-precision:** 128-bit integers with up to 16 decimals (default for Linux/macOS).
*   **Standard-precision:** 64-bit integers with up to 9 decimals (default for Windows).

See the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation) for more details.

**Rust feature flag:** To enable high-precision mode in Rust, add the `high-precision` feature to your Cargo.toml:

```toml
[dependencies]
nautilus_model = { version = "*", features = ["high-precision"] }
```

## Installation

Install NautilusTrader easily using `pip`:

```bash
pip install -U nautilus_trader
```

Choose from:

1.  Pre-built binary wheels from PyPI or Nautech Systems package index.
2.  Build from source.

> [!TIP]
> We highly recommend installing using the [uv](https://docs.astral.sh/uv) package manager with a "vanilla" CPython.
>
> Conda and other Python distributions *may* work but aren’t officially supported.

### Installation Commands

#### From PyPI

```bash
pip install -U nautilus_trader
```

#### From the Nautech Systems package index

```bash
pip install -U nautilus_trader --index-url=https://packages.nautechsystems.io/simple
```
```bash
pip install -U nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```
```bash
pip install nautilus_trader==1.208.0a20241212 --index-url=https://packages.nautechsystems.io/simple
```

## Redis

Redis is optional and only needed for using the [cache](https://nautilustrader.io/docs/latest/concepts/cache) database or [message bus](https://nautilustrader.io/docs/latest/concepts/message_bus).

## Makefile

The provided `Makefile` simplifies build and development tasks, including:

*   `make install` (install with release mode)
*   `make build` (build in release mode)
*   `make test-performance` (performance tests)

## Examples

Explore examples for indicators and strategies in Python and Cython:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py)
*   [indicator](/nautilus_trader/indicators/)
*   [strategy](/nautilus_trader/examples/strategies/)
*   [backtest](/examples/backtest/)

## Docker

Use pre-built Docker images:

*   `nautilus_trader:latest`: Latest release.
*   `nautilus_trader:nightly`: Head of `nightly` branch.
*   `jupyterlab:latest`: Latest release + JupyterLab.
*   `jupyterlab:nightly`: Head of `nightly` branch + JupyterLab.

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
```
```bash
docker pull ghcr.io/nautechsystems/jupyterlab:nightly --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

Refer to the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development guidance.

## Testing with Rust

Use [cargo-nextest](https://nexte.st) for reliable Rust testing.

## Contributing

Contributions are welcome! Open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your ideas and follow the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines.

> [!NOTE]
> Pull requests should target the `develop` branch.

## Community

Join the [Discord](https://discord.gg/NautilusTrader) community for support and updates.

> [!WARNING]
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">
```
Key improvements and SEO optimizations:

*   **Clear, Concise Title:** "NautilusTrader: High-Performance Algorithmic Trading Platform"
*   **SEO-Friendly Description:**  The one-sentence hook is strong, and the description now includes keyword phrases: "Python-native algorithmic trading," "backtesting," and "live deployment".  Important features are also listed, using keywords.
*   **Keyword Optimization:** Used keywords like "algorithmic trading platform," "backtesting," "live trading," "Python," "Rust," "high-performance," "modular," etc. throughout the document.
*   **Structured Headings:** Clean and clear headings (Introduction, Why Choose NautilusTrader, Technical Highlights, Integrations, etc.) for readability and SEO benefit.
*   **Bulleted Lists:**  Highlights key features for easy scanning and readability.  Includes a summary of the integrations.
*   **Concise Language:**  Removed unnecessary words and phrases to improve clarity.
*   **Strong Call to Action:**  "Get started today" with a direct link.
*   **Concise Overview:** Reorganized the information for better flow and conciseness.
*   **Directly Answers User Questions:** The revised text directly answers the "Why NautilusTrader" question.
*   **Improved Formatting:** Use of `alt` text on images.
*   **Clarity and Flow:** Improved the readability and logical flow of the information.
*   **Warnings and Tips:** Use of `[!NOTE]` and `[!TIP]` for improved clarity.
*   **Removed Duplication:** Streamlined installation steps.
*   **Expanded Integration Information:** More details in a table format.
*   **Removed Irrelevant Content:** Reduced verbosity.
*   **Combined Similar Sections:** Consolidating information to avoid repetition.
*   **Corrected Markdown Formatting:** Ensuring proper formatting.
*   **Clear Versioning:** Provided a better explanation of the branching strategy.
*   **Simplified Installation Instructions**
*   **Expanded Community Information**
*   **Improved the description of the platform**

This improved README provides a more informative, user-friendly, and SEO-optimized introduction to NautilusTrader.  It highlights the key benefits, features, and how to get started.