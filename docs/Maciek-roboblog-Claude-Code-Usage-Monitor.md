# 🚀 Claude Code Usage Monitor: Stay in Control of Your Claude AI Token Usage

**Tired of unexpectedly hitting your Claude AI token limits?**  Get real-time insights into your token consumption, predict session expirations, and optimize your AI usage with the **Claude Code Usage Monitor**! [Check out the original repository!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

**This powerful, real-time terminal monitoring tool provides:**

*   📊 **Real-time Monitoring:** Track token usage, burn rate, and cost with configurable refresh rates.
*   🔮 **ML-Based Predictions:**  Intelligent session limit detection and burn rate forecasting.
*   🎨 **Rich Terminal UI:**  Beautiful, color-coded progress bars, tables, and themes for optimal readability.
*   🤖 **Smart Auto-Detection:** Automatic plan switching, timezone, and time format detection.
*   💼 **Plan Support:** Supports multiple plans including Pro, Max5, Max20, and a custom plan.
*   📈 **Cost Analytics:** Model-specific pricing and cost calculation.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 🔑 Key Features

*   **Real-time Tracking:** Monitor your token usage in real-time.
*   **Burn Rate Analysis:** Analyze token consumption patterns to understand your usage velocity.
*   **Predictive Analytics:** Get intelligent session limit predictions.
*   **Customizable Views:** Choose between "realtime", "daily", or "monthly" views for detailed insights.
*   **Flexible Configuration:** Configure refresh rates, timezones, time formats, and more.
*   **Automated Plan Switching:** Automatically switch plans based on your usage.
*   **Detailed Cost Analysis:** Understand the cost of your Claude AI usage.
*   **Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   **Custom Plan:** Auto-adapting custom plan for optimal token usage.
*   **Easy Installation:** Simple installation options using `uv`, `pip`, `pipx`, and `conda`.

## 🚀 Installation

Choose your preferred method for installation:

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is the easiest, fastest, and recommended way to install the monitor.

```bash
# Install uv (if you don't have it)
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install claude-monitor
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
# Install
pip install claude-monitor

# Add to PATH (if needed)
# Check the pip install output for the warning message
#  e.g., WARNING: The script claude-monitor is installed in '/home/username/.local/bin' which is not on PATH
# If this message appears, add this to your ~/.bashrc or ~/.zshrc:
#   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # or ~/.zshrc
#   source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

#### pipx (Isolated Environments)

```bash
pipx install claude-monitor
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```

#### conda/mamba

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

## 📖 Usage

### ⚙️ Configuration Options

Configure the monitor to suit your needs with these command-line parameters:

| Parameter              | Type    | Default | Description                                                                                   |
| ---------------------- | ------- | ------- | --------------------------------------------------------------------------------------------- |
| `--plan`               | string  | `custom` | Plan type: `pro`, `max5`, `max20`, or `custom`                                                 |
| `--custom-limit-tokens` | int     | None    | Token limit for custom plan (must be > 0)                                                     |
| `--view`               | string  | `realtime` | View type: `realtime`, `daily`, or `monthly`                                                  |
| `--timezone`           | string  | `auto`  | Timezone (auto-detected). Examples: `UTC`, `America/New_York`, `Europe/London`                  |
| `--time-format`        | string  | `auto`  | Time format: `12h`, `24h`, or `auto`                                                           |
| `--theme`              | string  | `auto`  | Display theme: `light`, `dark`, `classic`, or `auto`                                             |
| `--refresh-rate`       | int     | 10      | Data refresh rate in seconds (1-60)                                                             |
| `--refresh-per-second` | float   | 0.75    | Display refresh rate in Hz (0.1-20.0)                                                           |
| `--reset-hour`         | int     | None    | Daily reset hour (0-23)                                                                         |
| `--log-level`          | string  | `INFO`  | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                                  |
| `--log-file`           | path    | None    | Log file path                                                                                 |
| `--debug`              | flag    | False   | Enable debug logging                                                                          |
| `--version`, `-v`      | flag    | False   | Show version information                                                                     |
| `--clear`              | flag    | False   | Clear saved configuration                                                                     |

### 💡 Key Features in v3.0.0

*   **Complete Architecture Rewrite:** Modular design with Single Responsibility Principle (SRP).
*   **Enhanced Functionality:**  P90 analysis, updated plan limits, cost analytics, and Rich UI.
*   **New CLI Options:**  Flexible settings for time format, token limits, logging, and configuration resets.
*   **Comprehensive Test Suite:** 100+ test cases for full coverage.

### 🚀 Usage Examples

```bash
# Run with default custom plan
claude-monitor

# Monitor your Pro plan (approx. 19,000 tokens)
claude-monitor --plan pro

# Monitor your Max5 plan (approx. 88,000 tokens)
claude-monitor --plan max5

# Monitor your Max20 plan (approx. 220,000 tokens)
claude-monitor --plan max20

# Set a custom token limit
claude-monitor --plan custom --custom-limit-tokens 150000

# Show daily usage
claude-monitor --view daily

# Set timezone
claude-monitor --timezone "America/Los_Angeles"

# Reset daily at a specific hour (e.g., 8 AM)
claude-monitor --reset-hour 8

# Force the dark theme
claude-monitor --theme dark

# Clear saved settings
claude-monitor --clear
```

### 🔄 Available Plans

| Plan        | Token Limit     | Best For                      |
| ----------- | --------------- | ----------------------------- |
| **custom**  | P90 auto-detect | Intelligent limit detection (default) |
| **pro**     | ~19,000         | Claude Pro subscription     |
| **max5**    | ~88,000         | Claude Max5 subscription     |
| **max20**   | ~220,000        | Claude Max20 subscription    |

## 🔧 Development Installation

[See the Development Installation section in the original README.](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor#-development-installation)

## 🤝 Contributing

We welcome contributions!  See our [Contributing Guide](CONTRIBUTING.md).

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

Have questions?  Contact [maciek@roboblog.eu](mailto:maciek@roboblog.eu).