# 🚀 Claude Code Usage Monitor: Stay Ahead of Your Anthropic AI Token Usage

Tired of hitting those AI token limits?  **Claude Code Usage Monitor** ([Original Repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)) is the ultimate real-time terminal tool for tracking your Claude AI token consumption. Get advanced analytics, machine learning-based predictions, and a beautiful, informative Rich UI, all designed to keep you productive.

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 📌 Key Features

*   ✅ **Real-time Monitoring:** Track your token usage, burn rate, and cost in real-time with configurable refresh rates (0.1-20 Hz).
*   📊 **Advanced UI:**  Enjoy a beautiful Rich UI with color-coded progress bars, tables, and WCAG-compliant contrast for optimal readability.
*   🔮 **ML-Powered Predictions:** Get intelligent session limit detection and accurate predictions about remaining tokens and session expirations.
*   🤖 **Smart Auto-Detection:** Automatic plan switching and custom limit discovery based on your usage patterns.
*   📈 **Cost Analytics:** Detailed model-specific pricing with cache token calculations to help you manage your spending.
*   ✨ **Comprehensive Logging:**  Optional file logging with configurable levels for in-depth analysis and troubleshooting.
*   📦 **Easy Installation:** Modern installation using `uv` for simplified setup and management.

---

## 🛠️ Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` simplifies installation, resolves Python version conflicts, and creates isolated environments automatically.

```bash
# Install uv (one-time setup - Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (one-time setup - Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal

# Install claude-monitor from PyPI using uv
uv tool install claude-monitor

# Run from anywhere
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

### 🐳 Other Package Managers

#### pipx (Isolated Environments)
```bash
# Install with pipx
pipx install claude-monitor

# Run from anywhere
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```


#### conda/mamba
```bash
# Install with pip in conda environment
pip install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

---

## 📖 Usage

### Basic Commands

```bash
# Run the monitor with default settings (Custom plan with auto-detection)
claude-monitor

# View help information
claude-monitor --help

# Press Ctrl+C to gracefully exit
```

### Configuration Options

Control the monitor's behavior with various command-line parameters:

| Parameter              | Type    | Default | Description                                                   |
| ---------------------- | ------- | ------- | ------------------------------------------------------------- |
| `--plan`               | string  | custom  | Plan type: `pro`, `max5`, `max20`, or `custom`                |
| `--custom-limit-tokens` | int     | None    | Token limit for custom plan (must be > 0)                     |
| `--view`               | string  | realtime | View type: `realtime`, `daily`, or `monthly`                    |
| `--timezone`           | string  | auto    | Timezone (auto-detected). Examples: `UTC`, `America/New_York` |
| `--time-format`        | string  | auto    | Time format: `12h`, `24h`, or `auto`                          |
| `--theme`              | string  | auto    | Display theme: `light`, `dark`, `classic`, or `auto`            |
| `--refresh-rate`       | int     | 10      | Data refresh rate in seconds (1-60)                           |
| `--refresh-per-second` | float   | 0.75    | Display refresh rate in Hz (0.1-20.0)                         |
| `--reset-hour`         | int     | None    | Daily reset hour (0-23)                                       |
| `--log-level`          | string  | INFO    | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file`           | path    | None    | Log file path                                                 |
| `--debug`              | flag    | False   | Enable debug logging                                          |
| `--version, -v`        | flag    | False   | Show version information                                      |
| `--clear`              | flag    | False   | Clear saved configuration                                     |

### Plan Options

Select your Anthropic Claude plan:

| Plan        | Token Limit     | Best For                         |
| ----------- | --------------- | -------------------------------- |
| **custom**  | P90 auto-detect | Intelligent limit detection (default) |
| **pro**     | ~19,000         | Claude Pro subscription          |
| **max5**    | ~88,000         | Claude Max5 subscription         |
| **max20**   | ~220,000        | Claude Max20 subscription        |

---

## ✨ Features Deep Dive

### 🔄 Real-time Monitoring & Rich UI

*   **Configurable Refresh Rates:** Fine-tune the update intervals to match your needs.
*   **High-Precision Display Refresh:**  Control the display refresh rate (0.1-20 Hz) for optimal performance and responsiveness.
*   **Intelligent Change Detection:** Minimize CPU usage by only updating when necessary.
*   **Adaptive UI Themes:**  The monitor automatically detects your terminal background and provides optimized themes.
*   **Multiple Views:** Switch between real-time, daily, and monthly views for comprehensive analysis.

### 🔮 Machine Learning Predictions

*   **P90 Calculator:** Uses 90th percentile analysis for intelligent limit detection and accurate forecasts.
*   **Burn Rate Analytics:** Tracks your consumption patterns across multiple sessions.
*   **Cost Projections:** Model-specific pricing with cache token calculations.
*   **Session Forecasting:** Predicts when your sessions will expire based on your usage patterns.

### 🤖 Intelligent Auto-Detection

*   **Automatic Plan Switching:** The monitor can automatically switch plans based on your usage.
*   **Limit Discovery:**  Analyzes historical data to find your actual token limits.

---

## 🚀 Usage Examples

*   **Basic Usage:** `claude-monitor` (starts with the Custom plan)
*   **Specify Your Plan:** `claude-monitor --plan pro` (for Pro subscribers)
*   **Custom Plan with Limit:** `claude-monitor --plan custom --custom-limit-tokens 150000`
*   **Daily View:** `claude-monitor --view daily`
*   **Set Timezone:** `claude-monitor --timezone America/Los_Angeles`
*   **Enable Debugging:** `claude-monitor --debug`

---

## 🤝 Contributing

We welcome contributions!  See our [Contributing Guide](CONTRIBUTING.md) to learn how you can help.

---

## 📚 Additional Resources

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

Special thanks to our sponsors and contributors:

*   **Ed** - *Buy Me Coffee Supporter*
*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>