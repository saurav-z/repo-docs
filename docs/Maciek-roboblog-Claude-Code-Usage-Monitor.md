# Claude Code Usage Monitor: Stay in Control of Your Claude AI Token Usage

Tired of guessing how much you're spending on Claude AI?  **Claude Code Usage Monitor** is the essential tool to track and manage your token consumption, offering real-time monitoring, intelligent predictions, and a beautiful terminal UI.  [Check out the original repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

**Key features**
*   **Real-Time Monitoring:** Track your token usage in real-time with configurable refresh rates.
*   **ML-Based Predictions:**  Get intelligent session limit detection and cost projections.
*   **Rich, Intuitive UI:**  Visualize your data with color-coded progress bars and tables.
*   **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **Detailed Analytics:** Analyze burn rate, cost, and time predictions.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Table of Contents

*   [Key Features](#key-features)
*   [Installation](#installation)
    *   [Modern Installation with `uv` (Recommended)](#modern-installation-with-uv-recommended)
    *   [Installation with `pip`](#installation-with-pip)
    *   [Other Package Managers](#other-package-managers)
*   [Usage](#usage)
    *   [Command Line Options](#command-line-options)
    *   [Available Plans](#available-plans)
    *   [Usage Examples](#usage-examples)
*   [Understanding Sessions](#understanding-sessions)
*   [Features & How It Works](#features--how-it-works)
*   [Development Installation](#development-installation)
*   [Troubleshooting](#troubleshooting)
*   [Contact](#contact)
*   [License](#license)
*   [Contributors](#contributors)
*   [Acknowledgements](#acknowledgements)

## Key Features

*   **Real-time Monitoring:** Configurable refresh rates with dynamic display updates.
*   **Advanced UI:** Color-coded progress bars, tables, and WCAG-compliant contrast.
*   **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **Comprehensive Plan Support:** Pro, Max5, Max20 and P90 Custom plans.
*   **Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   **Cost Analytics:** Model-specific pricing with cache token calculations.
*   **ML-Based Predictions:** P90 percentile calculations and intelligent session limit detection.
*   **Comprehensive Testing:** Extensive test suite with 100+ test cases and full coverage.

## Installation

### Modern Installation with `uv` (Recommended)

`uv` offers the fastest and easiest installation, managing isolated environments and eliminating Python version issues.

```bash
# Install with uv
uv tool install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

#### First-time uv users
If you don't have uv installed yet, get it with one command:

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```
### Installation with `pip`

```bash
# Install
pip install claude-monitor

# (if necessary) Add to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

### Other Package Managers

```bash
# pipx
pipx install claude-monitor

# conda/mamba
pip install claude-monitor
```

## Usage

### Command Line Options

Get help: `claude-monitor --help`

**Key Parameters:**

*   `--plan`:  Plan type (`pro`, `max5`, `max20`, or `custom`). Default: `custom`.
*   `--view`: View type (`realtime`, `daily`, or `monthly`). Default: `realtime`.
*   `--timezone`: Timezone (e.g., `UTC`, `America/New_York`).  Default: `auto`.
*   `--time-format`: Time format (`12h`, `24h`, or `auto`).
*   `--refresh-rate`: Data refresh rate in seconds (1-60).
*   `--refresh-per-second`: Display refresh rate in Hz (0.1-20.0).
*   `--reset-hour`: Daily reset hour (0-23).
*   `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   `--clear`: Clear saved configuration.

### Available Plans

| Plan      | Token Limit   | Description                     |
|-----------|---------------|---------------------------------|
| `custom`  | P90 auto-detect | Intelligent limit detection     |
| `pro`     | 19,000        | Claude Pro subscription         |
| `max5`    | 88,000        | Claude Max5 subscription        |
| `max20`   | 220,000       | Claude Max20 subscription       |

### Usage Examples

```bash
# Run with the custom plan (default)
claude-monitor

# Run with the Pro plan
claude-monitor --plan pro

# Set the daily reset hour
claude-monitor --reset-hour 9

# Show the daily usage
claude-monitor --view daily
```

## Understanding Sessions

Claude Code operates on a 5-hour rolling session window system.

## Features & How It Works

*   **Architecture:** Modular design following the Single Responsibility Principle.
*   **Data Flow:**  Claude Config Files → Data Layer → Analysis Engine → UI Components → Terminal Display

## Development Installation

```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
pip install -e .
python -m claude_monitor
```

## Troubleshooting

Refer to the Troubleshooting section in the original README for common issues.

## Contact

*   **Email:** maciek@roboblog.eu

## License

[MIT License](LICENSE)

## Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## Acknowledgements

### Sponsors

*   **Ed** - *Buy Me Coffee Supporter*

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>