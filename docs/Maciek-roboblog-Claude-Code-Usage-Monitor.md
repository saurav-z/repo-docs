# 🚀 Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

Tired of running out of tokens mid-session? **Monitor your Claude AI token usage in real-time with the Claude Code Usage Monitor**, providing advanced analytics, ML-based predictions, and a beautiful terminal UI to keep you informed.  Track your token consumption, understand your burn rate, and get intelligent predictions about your session limits.

[View the Original Repository on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **Real-time Monitoring:** Track token usage, cost, and burn rate with configurable refresh rates (0.1-20Hz).
*   **AI-Powered Predictions:** Machine learning algorithms provide P90 percentile calculations and intelligent session limit detection.
*   **Advanced Rich UI:** Enjoy a beautiful, color-coded terminal UI with progress bars, tables, and layouts.
*   **Smart Auto-Detection:** Automatic plan switching and custom limit discovery to optimize your usage.
*   **Comprehensive Plan Support:** Updated support for Pro, Max5, Max20, and custom plans.
*   **Advanced Warning System:** Receive multi-level alerts with cost and time predictions.
*   **Professional Architecture:** Modular design with Single Responsibility Principle (SRP) compliance.
*   **Intelligent Theming:** Automatic terminal background detection for optimal readability.
*   **Configuration and Logging:** Save your settings, with optional file logging, and multiple log levels for debugging and monitoring.

## Installation

Choose the installation method that suits your needs:

### ⚡ Recommended: Modern Installation with `uv`

`uv` offers the easiest and most reliable way to install and manage dependencies.

#### Install uv

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

#### Install from PyPI

```bash
# Install directly from PyPI with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

#### Install from Source

```bash
# Clone and install from source
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
uv tool install .

# Run from anywhere
claude-monitor
```

### 📦 Alternative: Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

> **⚠️ PATH Setup**:  If you see `WARNING: The script claude-monitor is installed in '/home/username/.local/bin' which is not on PATH`, follow the `export PATH` command.
>
> **⚠️ Important**: On modern Linux distributions, consider using `uv`, `pipx`, or a virtual environment to avoid "externally-managed-environment" errors (see Troubleshooting).

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within a conda environment)

## Usage

### Basic Usage

```bash
# Run with default settings (Custom plan)
claude-monitor

# Or use command aliases
cmonitor  # short alias
ccmonitor  # short alternative
ccm       # shortest alias
```

Press `Ctrl+C` to exit the monitor.

### Configuration Options

```bash
# Show help information
claude-monitor --help
```

#### Common Usage Examples

*   **Specify a Plan:** `claude-monitor --plan pro` (Pro plan) or `claude-monitor --plan max5` (Max5 plan) or `claude-monitor --plan custom` (Custom plan)
*   **Custom Token Limit:** `claude-monitor --plan custom --custom-limit-tokens 100000`
*   **Set a Reset Hour:** `claude-monitor --reset-hour 3` (Resets at 3 AM)
*   **View Options:** `claude-monitor --view daily` (Daily view) or `claude-monitor --view monthly` (Monthly view)
*   **Timezone Configuration:** `claude-monitor --timezone America/New_York`
*   **Logging and Debugging:** `claude-monitor --debug` or `claude-monitor --log-file ~/.claude-monitor/logs/monitor.log`

#### Key CLI Parameters

| Parameter                | Type      | Default    | Description                                                                 |
| ------------------------ | --------- | ---------- | --------------------------------------------------------------------------- |
| `--plan`                 | `string`  | `custom`   | Plan type: `pro`, `max5`, `max20`, or `custom`                              |
| `--custom-limit-tokens`  | `int`     | `None`     | Token limit for custom plan                                                 |
| `--view`                 | `string`  | `realtime` | View type: `realtime`, `daily`, or `monthly`                                 |
| `--timezone`             | `string`  | `auto`     | Timezone (auto-detected)                                                    |
| `--time-format`          | `string`  | `auto`     | Time format: `12h`, `24h`, or `auto`                                      |
| `--theme`                | `string`  | `auto`     | Display theme: `light`, `dark`, `classic`, or `auto`                         |
| `--refresh-rate`         | `int`     | `10`       | Data refresh rate in seconds (1-60)                                           |
| `--refresh-per-second`   | `float`   | `0.75`     | Display refresh rate in Hz (0.1-20.0)                                          |
| `--reset-hour`           | `int`     | `None`     | Daily reset hour (0-23)                                                       |
| `--log-level`            | `string`  | `INFO`     | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                |
| `--log-file`             | `path`    | `None`     | Log file path                                                               |
| `--debug`                | `flag`    | `False`    | Enable debug logging                                                        |
| `--version, -v`           | `flag`    | `False`    | Show version information                                                     |
| `--clear`                | `flag`    | `False`    | Clear saved configuration                                                  |

#### Plan Options

| Plan       | Token Limit   | Description                                                                 |
|------------|---------------|-----------------------------------------------------------------------------|
| `custom`   | P90-based     | Auto-detection with machine learning analysis (Default)                  |
| `pro`      | ~19,000       | Claude Pro subscription                                                    |
| `max5`     | ~88,000       | Claude Max5 subscription                                                   |
| `max20`    | ~220,000      | Claude Max20 subscription                                                  |

## What's New in v3.0.0

*   **Complete Architecture Rewrite:** Modular design, SRP compliance, Pydantic validation, and comprehensive testing.
*   **Enhanced Functionality:** Machine learning-based limit detection using P90 percentile calculations, updated plan limits, and cost analytics.
*   **New CLI Options:** Configurable display refresh, automatic time format detection, and advanced logging capabilities.
*   **Breaking Changes:** Package name change (from `claude-usage-monitor` to `claude-monitor`) and minimum Python version increase.

## Features & How It Works

*   **Real-time Monitoring**
*   **Rich UI Components**
*   **Multiple Usage Views**
*   **Machine Learning Predictions**
*   **Intelligent Auto-Detection**

## Troubleshooting

See the [Troubleshooting Section](#troubleshooting) for solutions to common installation and runtime issues.

## Contact

For questions, suggestions, or collaboration, contact [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## License

[MIT License](LICENSE)

## Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>