# 🚀 Claude Code Usage Monitor: Stay in Control of Your Claude AI Token Usage

Tired of exceeding your Claude AI token limits? **This real-time terminal monitor provides advanced analytics, intelligent predictions, and a beautiful Rich UI to help you effortlessly track and optimize your token consumption.**.  Check out the original repo [here](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)!

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 🔑 Key Features

*   **🔮 ML-Based Predictions:** Get intelligent session limit detection with P90 percentile calculations.
*   **🔄 Real-Time Monitoring:** Track token usage with configurable refresh rates (0.1-20 Hz).
*   **📊 Advanced Rich UI:** Enjoy color-coded progress bars, sortable data tables, and responsive design with WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:**  Automatic plan switching and custom limit discovery.
*   **📋 Enhanced Plan Support:** Updated limits for Pro, Max5, Max20, and a flexible Custom plan.
*   **⚠️ Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   **🎨 Intelligent Theming:** Scientific color schemes with automatic terminal background detection.
*   **⏰ Advanced Scheduling:** Auto-detected system timezone and time format preferences.
*   **📈 Cost Analytics:** Model-specific pricing with cache token calculations.
*   **📝 Comprehensive Logging:** Optional file logging with configurable levels.
*   **⚡ Performance Optimized:** Efficient data processing and caching.

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` provides the easiest and fastest installation and is recommended for all users.

```bash
uv tool install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

See detailed instructions for `uv` installation from source and first-time `uv` users in the original README.

### 📦 Installation with `pip`

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```
**Important**: Ensure your `~/.local/bin` is in your `PATH`.

### 🛠️ Alternative Installation Methods

*   **pipx:**  `pipx install claude-monitor`
*   **conda/mamba:**  `pip install claude-monitor` within your conda environment.

## 📖 Usage

### ⚙️ Configuration Options

Customize the monitor to fit your needs.  All options can be saved for future sessions.

*   **--plan:** (default: `custom`) Specify your Claude plan (pro, max5, max20, custom).
*   **--custom-limit-tokens:** Set a token limit for the custom plan.
*   **--view:** (default: `realtime`) Choose your display view (realtime, daily, monthly).
*   **--timezone:** (default: `auto`) Set your timezone.
*   **--time-format:** (default: `auto`) Select time format (12h, 24h).
*   **--theme:** (default: `auto`) Choose display theme (light, dark, classic).
*   **--refresh-rate:** (default: 10) Data refresh rate (seconds).
*   **--refresh-per-second:** (default: 0.75) Display refresh rate (Hz).
*   **--reset-hour:** Set daily reset hour (0-23).
*   **--log-level:** (default: `INFO`) Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   **--log-file:** Set the log file path.
*   **--debug:** Enable debug logging.
*   **--clear:** Clear saved configuration.

### 💻 Basic Usage

```bash
claude-monitor  # Run with default settings
claude-code-monitor # (full name)
cmonitor  # (short)
ccmonitor # (short alternative)
ccm # (shortest)

# Example with plan and theme
claude-monitor --plan pro --theme dark
```

### ℹ️ Get Help

```bash
claude-monitor --help
```

## ✨ Features & How It Works

### 🔍 Understanding Claude Sessions

*   Claude Code sessions operate on a 5-hour rolling window.
*   The monitor accurately calculates burn rates and predicts session expiration.

### 📊 Token Limits by Plan (v3.0.0)

| Plan           | Token Limit | Best For                  |
|----------------|-------------|---------------------------|
| **custom**       | P90-based   | Intelligent limit detection (default) |
| **pro**          | ~19,000     | Claude Pro subscription |
| **max5**         | ~88,000     | Claude Max5 subscription |
| **max20**        | ~220,000    | Claude Max20 subscription |

### 🤖 Smart Detection Features

*   **Automatic Plan Switching**: Automatically switches to a custom plan if usage exceeds limits.
*   **Limit Discovery**:  Analyzes your usage history to find actual token limits.

## 🚀 Usage Examples (Common Scenarios)

*   **Morning Developer:** Configure reset times to align with your workday.
*   **Night Owl Coder:** Use flexible reset scheduling for late-night sessions.
*   **Heavy User with Variable Limits:** Let auto-detection find your actual limits.
*   **International User:** Set your specific timezone.
*   **Quick Check:** Just run the default command.
*   **Usage Analysis Views:** View daily and monthly token usage patterns.

## 🔧 Development Installation

For developers and contributors, detailed setup, testing, and contribution guidelines are available.

## 📞 Contact

For questions, suggestions, or collaboration, reach out to [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

MIT License

## 🤝 Contributors

See the [Contributors](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/graphs/contributors) page for a full list.

## 🙏 Acknowledgments

Thanks to Ed and other supporters!

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>