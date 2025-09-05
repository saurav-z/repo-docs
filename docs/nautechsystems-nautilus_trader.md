# NautilusTrader: High-Performance Algorithmic Trading Platform

**Unlock the power of AI-first, high-performance algorithmic trading with NautilusTrader, a cutting-edge platform built for speed, reliability, and flexibility.** Explore the [NautilusTrader GitHub repository](https://github.com/nautechsystems/nautilus_trader) for more details.

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

| Platform           | Rust   | Python    |
| :----------------- | :----- | :-------- |
| `Linux (x86_64)`   | 1.89.0 | 3.11-3.13 |
| `Linux (ARM64)`    | 1.89.0 | 3.11-3.13 |
| `macOS (ARM64)`    | 1.89.0 | 3.11-3.13 |
| `Windows (x86_64)` | 1.89.0 | 3.11-3.13 |

- **Docs**: <https://nautilustrader.io/docs/>
- **Website**: <https://nautilustrader.io>
- **Support**: [support@nautilustrader.io](mailto:support@nautilustrader.io)

## Key Features

*   **High Performance:** Built with Rust for speed and efficiency.
*   **Reliable & Safe:** Leverage Rust's memory safety and optional Redis-backed state persistence.
*   **Cross-Platform:** Runs on Linux, macOS, and Windows, deployable with Docker.
*   **Modular & Flexible:** Easily integrate with any REST API or WebSocket feed via modular adapters.
*   **Advanced Order Types:** Supports `IOC`, `FOK`, `GTC`, `GTD`, `DAY`, `AT_THE_OPEN`, `AT_THE_CLOSE`, plus conditional triggers, and execution instructions like `post-only`, `reduce-only`, and icebergs.
*   **Customizable:** Add custom components and assemble systems from scratch using cache and message bus.
*   **Backtesting & Live Trading Parity:** Use identical strategy implementations for both backtesting and live deployments.
*   **Multi-Venue Support:** Enables market-making and statistical arbitrage strategies.
*   **AI Ready:** Backtest engine is fast enough to be used to train AI trading agents (RL/ES).

![nautilus-trader](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/nautilus-trader.png "nautilus-trader")

## Why Choose NautilusTrader?

*   **High-Performance Event-Driven Python:** Leveraging native binary core components.
*   **Code Reusability:** Achieve parity between backtesting and live trading environments with identical strategy code.
*   **Reduced Risk:** Enhanced risk management, accuracy, and type safety.
*   **Extensible Architecture:** Featuring message bus, custom components, actors, data integration, and adaptable integration.

NautilusTrader provides a streamlined approach, avoiding the need for separate implementations by leveraging Rust's core components and a Python-native environment optimized for quantitative traders.

## Integrations

NautilusTrader offers modular adapters for seamless connectivity to various trading venues and data providers.

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

See the [Integrations](https://nautilustrader.io/docs/latest/integrations/index.html) documentation for further details.

## Versioning and Releases

NautilusTrader is under active development and follows a bi-weekly release schedule.

*   `master`: Latest released version.
*   `nightly`: Daily snapshots from the `develop` branch.
*   `develop`: Active development branch.

## Installation

Install NautilusTrader using pip:

```bash
pip install -U nautilus_trader
```

For advanced installation options, including from source and using development builds, refer to the [Installation Guide](https://nautilustrader.io/docs/latest/getting_started/installation).

## Examples

Explore the example indicators, strategies, and backtest applications to get started with NautilusTrader:

*   [indicator](/nautilus_trader/examples/indicators/ema_python.py)
*   [indicator](/nautilus_trader/indicators/)
*   [strategy](/nautilus_trader/examples/strategies/)
*   [backtest](/examples/backtest/)

## Docker

Pre-built Docker images are available:

*   `nautilus_trader:latest` (latest release)
*   `nautilus_trader:nightly` (nightly build)
*   `jupyterlab:latest` (JupyterLab with example notebooks)
*   `jupyterlab:nightly` (JupyterLab with nightly build)

```bash
docker pull ghcr.io/nautechsystems/<image_variant_tag> --platform linux/amd64
docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:nightly
```

## Development

Refer to the [Developer Guide](https://nautilustrader.io/docs/latest/developer_guide/index.html) for development guidelines and best practices.

## Contributing

Contributions are welcome!  Please review the [CONTRIBUTING.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md) guidelines and open an [issue](https://github.com/nautechsystems/nautilus_trader/issues) to discuss your ideas.

## Community

Connect with the NautilusTrader community on [Discord](https://discord.gg/NautilusTrader) for support, discussions, and the latest updates.

> [!WARNING]
>
> NautilusTrader does not issue, promote, or endorse any cryptocurrency tokens. Any claims or communications suggesting otherwise are unauthorized and false.
>
> All official updates and communications from NautilusTrader will be shared exclusively through <https://nautilustrader.io>, our [Discord server](https://discord.gg/NautilusTrader),
> or our X (Twitter) account: [@NautilusTrader](https://x.com/NautilusTrader).
>
> If you encounter any suspicious activity, please report it to the appropriate platform and contact us at <info@nautechsystems.io>.

## License

NautilusTrader is licensed under the [GNU Lesser General Public License v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).

---

NautilusTrader™ is developed and maintained by Nautech Systems, a technology
company specializing in the development of high-performance trading systems.
For more information, visit <https://nautilustrader.io>.

© 2015-2025 Nautech Systems Pty Ltd. All rights reserved.

![nautechsystems](https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ns-logo.png "nautechsystems")
<img src="https://github.com/nautechsystems/nautilus_trader/raw/develop/assets/ferris.png" width="128">