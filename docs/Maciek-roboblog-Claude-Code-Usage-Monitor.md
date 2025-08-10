# 🚀 Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

**Tired of unexpected limits?** Claude Code Usage Monitor provides a beautiful, real-time terminal interface to track your Claude AI token usage, predict session limits, and optimize your workflow.  [View the original repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

This powerful tool offers:

*   **Real-time Monitoring:** Track token consumption, burn rate, and cost analysis.
*   **AI-Powered Predictions:** Intelligent session limit detection and warnings.
*   **Advanced Rich UI:**  Color-coded progress bars, tables, and WCAG-compliant themes.
*   **Customizable Plans:** Support for Claude Pro, Max5, Max20, and the default, ML-driven Custom plan.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **🔮 ML-based predictions:** P90 percentile calculations and intelligent session limit detection.
*   **🔄 Real-time monitoring:** Configurable refresh rates (0.1-20 Hz) with intelligent display updates.
*   **📊 Advanced Rich UI:** Beautiful color-coded progress bars, tables, and layouts with WCAG-compliant contrast.
*   **🤖 Smart auto-detection:** Automatic plan switching with custom limit discovery.
*   **📋 Enhanced plan support:** Updated limits: Pro (19k), Max5 (88k), Max20 (220k), Custom (P90-based).
*   **⚠️ Advanced warning system:** Multi-level alerts with cost and time predictions.
*   **🎨 Intelligent theming:** Scientific color schemes with automatic terminal background detection.
*   **⏰ Advanced scheduling:** Auto-detected system timezone and time format preferences.
*   **📈 Cost analytics:** Model-specific pricing with cache token calculations.
*   **📝 Comprehensive logging:** Optional file logging with configurable levels.
*   **🧪 Extensive testing:** 100+ test cases with full coverage.
*   **⚡ Performance optimized:** Advanced caching and efficient data processing.

### The Custom Plan: Your Smart Default

The Custom plan is the default, designed for optimal Claude Code sessions. It monitors token usage, message count, and cost, adapting to your patterns by analyzing your last 8 days of usage to provide accurate, personalized limits and warnings.

## Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is a lightning-fast Python package and virtual environment manager.

```bash
# Install with uv
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

**If you don't have `uv` installed:**

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within a conda environment)

## Usage

### Basic Usage

```bash
# Run the monitor with the default (custom) plan
claude-monitor
# or
cmonitor
# or
ccmonitor
# or
ccm
# To exit - Ctrl+C
```

### Configuration Options

Customize your monitoring with command-line arguments:

*   `--plan`:  `pro`, `max5`, `max20`, or `custom` (default).
*   `--custom-limit-tokens`: Token limit for custom plan.
*   `--view`:  `realtime` (default), `daily`, or `monthly`.
*   `--timezone`: Your timezone (e.g., `America/New_York`, `UTC`).
*   `--time-format`: `12h`, `24h`, or `auto`.
*   `--theme`: `light`, `dark`, `classic`, or `auto`.
*   `--refresh-rate`: Data refresh interval (seconds).
*   `--refresh-per-second`: Display refresh rate (Hz).
*   `--reset-hour`: Daily reset hour (0-23).
*   `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   `--log-file`: Log file path.
*   `--debug`: Enable debug logging.
*   `--version, -v`: Show version information.
*   `--clear`: Clear saved configuration.

#### Examples

```bash
# Run with pro plan, dark theme and NY time
claude-monitor --plan pro --theme dark --timezone "America/New_York"

# Reset your tokens at 1 AM
claude-monitor --reset-hour 1
```

### Plan Options

| Plan       | Token Limit | Best For                                      |
|------------|-------------|-----------------------------------------------|
| **custom** | P90-based  | Intelligent limit detection (default)       |
| **pro**    | ~19,000     | Claude Pro subscription                     |
| **max5**   | ~88,000     | Claude Max5 subscription                    |
| **max20**  | ~220,000    | Claude Max20 subscription                   |

## What's New in v3.0.0

This release features a complete architecture rewrite for improved performance, accuracy, and a richer user experience. Key improvements include:

*   **Complete Architecture Rewrite:** SRP-compliant modular design, Pydantic-based configuration, Sentry integration.
*   **Enhanced Functionality:** P90 analysis for limit detection, updated plan limits, cost analytics.
*   **New CLI Options:**  Configurable refresh rates, time formats, and more.
*   **Breaking Changes:** Package name change, default plan change, Python 3.9+ requirement.

## Troubleshooting

*   **"externally-managed-environment" error**:  Use `uv`, `pipx`, or a virtual environment.
*   **Command not found:** Ensure `~/.local/bin` is in your PATH.
*   **Runtime Issues**: If you are not seeing a new session, start with 2 messages to see if the session will load.  You can then also specify a custom config path with this command: `CLAUDE_CONFIG_DIR=~/.config/claude ./claude_monitor.py`

## 📞 Contact

For questions, suggestions, or collaboration, contact [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

MIT License

## 🤝 Contributors

A big thanks to [@adawalli](https://github.com/adawalli), [@taylorwilsdon](https://github.com/taylorwilsdon), and [@moneroexamples](https://github.com/moneroexamples) for their contributions!

## 🙏 Acknowledgments

Special thanks to **Ed** for his support!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>