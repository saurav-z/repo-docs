# Claude Code Usage Monitor: Stay Ahead of Your AI Token Usage 🚀

**Effortlessly track and optimize your Anthropic Claude AI token usage with a beautiful, real-time terminal monitor.**

[View the Original Repository](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

Tired of unexpected AI costs?  This powerful terminal tool provides real-time token tracking, cost analysis, and intelligent session limit predictions for Anthropic's Claude AI.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **Real-time Monitoring:** Track token consumption, cost, and burn rate in real-time.
*   **Intelligent Predictions:**  ML-powered session limit predictions and multi-level alerts.
*   **Advanced UI:**  Rich, color-coded progress bars, tables, and adaptive layouts.
*   **Customizable Plans:** Supports various Claude plans, including a smart, auto-detecting "Custom" plan.
*   **Performance Optimized:** Efficient data processing and advanced caching.
*   **ML-based Predictions:** P90 percentile calculations and intelligent session limit detection
*   **Smart Auto-detection:** Automatic plan switching with custom limit discovery
*   **Cost Analytics:** Model-specific pricing with cache token calculations
*   **Comprehensive Logging & Debugging:** Optional file logging and Sentry integration for production monitoring.

## Installation

Choose your preferred method:

### ⚡ Modern Installation with `uv` (Recommended)

`uv` simplifies installation and avoids common Python environment issues.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install claude-monitor
uv tool install claude-monitor

# Run the monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within your conda environment)

## Usage

### Basic Usage

```bash
# Run with default settings (Custom plan with auto-detection)
claude-monitor

# Alternative commands
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# Exit the monitor
# Press Ctrl+C to gracefully exit
```

### Configuration Options

Customize the monitor to your needs with command-line parameters.

*   `--plan`: Specify your Claude plan (e.g., `pro`, `max5`, `max20`, `custom`).
*   `--custom-limit-tokens`:  Set a custom token limit for the "custom" plan.
*   `--view`: Select a view (e.g., `realtime`, `daily`, `monthly`).
*   `--timezone`: Set your timezone (e.g., `America/New_York`, `UTC`).
*   `--time-format`: Set time format, 12h or 24h (auto-detect by default).
*   `--theme`: Choose a theme (e.g., `light`, `dark`, `classic`, `auto`).
*   `--refresh-rate`: Set the data refresh rate (seconds).
*   `--refresh-per-second`: Set display refresh rate in Hz.
*   `--reset-hour`: Set the daily reset hour.
*   `--log-level`: Set logging level (e.g., `DEBUG`, `INFO`).
*   `--log-file`: Specify a log file path.
*   `--debug`: Enable debug logging.
*   `--clear`: Clear saved configuration.

For detailed help, run `claude-monitor --help`.

## Custom Plan: The Default

The "Custom" plan is now the default.  It intelligently adapts to your usage by:

*   Analyzing your past sessions (last 8 days).
*   Calculating personalized limits based on your actual token usage.
*   Providing accurate predictions and warnings tailored to your workflow.

## Development

For developers, see the [Development Installation](#-development-installation) section.

## Troubleshooting

See the [Troubleshooting](#troubleshooting) section for solutions to common installation and runtime issues.

## Additional Information

*   [Documentation (Roadmap)](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [License](LICENSE)
*   [Acknowledgments](Acknowledgements)

## Contact

Need help or have suggestions?  Contact [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## Contributors

Special thanks to:

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">
  **⭐  Enjoying the project?  Star this repo!  ⭐**
</div>