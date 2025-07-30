# 🤖 Claude Code Usage Monitor: Real-time AI Token Tracking and Prediction

**Tired of unexpected AI usage limits?** Monitor your Claude AI token consumption with the **Claude Code Usage Monitor**, a powerful terminal tool that provides real-time tracking, advanced analytics, and intelligent predictions, available on [GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)!

## 🌟 Key Features

*   **Real-Time Monitoring:** Track token usage, cost, and messages in real-time with configurable refresh rates.
*   **ML-Powered Predictions:** Intelligent session limit detection and P90 percentile calculations for accurate forecasting.
*   **Rich Terminal UI:** Beautiful, color-coded progress bars, tables, and layouts with WCAG-compliant contrast for optimal readability.
*   **Automated Plan Switching:** Automatically adjusts your plan based on usage, with custom limit discovery.
*   **Comprehensive Plan Support:** Supports Pro, Max5, Max20 and Custom plans with updated limits.
*   **Detailed Cost Analytics:** Model-specific pricing and cache token calculations.
*   **Customizable Views:** Real-time, daily, and monthly usage views.
*   **Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   **Configuration Saving:** Preferences are automatically saved to avoid re-specifying them on each run.
*   **Robust Testing:**  Extensive testing with 100+ test cases and full coverage.

## 🚀 Installation

Choose your preferred installation method:

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is the fastest and most reliable way to install and manage the monitor.

```bash
# Install uv (if you don't have it) - Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (if you don't have it) - Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install from PyPI using uv
uv tool install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor, ccm
```

### 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# If command not found, add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Run
claude-monitor  # or cmonitor, ccmonitor, ccm
```

### 🛠️ Other Package Managers

*   **pipx (Isolated Environments):** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` within your conda environment

## 📖 Usage

Get started by typing `claude-monitor` in your terminal.

### Key Command-Line Options

| Parameter            | Type   | Default   | Description                                                                                                                                                                                          |
| :------------------- | :----- | :-------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--plan`             | string | `custom`  | Plan type: `pro`, `max5`, `max20`, or `custom`.                                                                                                                                                       |
| `--custom-limit-tokens` | int    | `None`    | Token limit for `custom` plan.                                                                                                                                                                       |
| `--view`             | string | `realtime`| View type: `realtime`, `daily`, or `monthly`.                                                                                                                                                          |
| `--timezone`         | string | `auto`    | Timezone (e.g., `UTC`, `America/New_York`).                                                                                                                                                           |
| `--time-format`      | string | `auto`    | Time format: `12h`, `24h`, or `auto`.                                                                                                                                                                |
| `--theme`            | string | `auto`    | Display theme: `light`, `dark`, `classic`, or `auto`.                                                                                                                                                  |
| `--refresh-rate`     | int    | `10`      | Data refresh rate in seconds (1-60).                                                                                                                                                                   |
| `--refresh-per-second` | float  | `0.75`    | Display refresh rate in Hz (0.1-20.0).                                                                                                                                                                  |
| `--reset-hour`       | int    | `None`    | Daily reset hour (0-23).                                                                                                                                                                              |
| `--log-level`        | string | `INFO`    | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.                                                                                                                                         |
| `--log-file`         | path   | `None`    | Log file path.                                                                                                                                                                                         |
| `--debug`            | flag   | `False`   | Enable debug logging.                                                                                                                                                                                  |
| `--version`, `-v`    | flag   | `False`   | Show version information.                                                                                                                                                                               |
| `--clear`            | flag   | `False`   | Clear saved configuration.                                                                                                                                                                               |

### Basic Examples:

*   **Run with default settings (Custom plan):** `claude-monitor`
*   **Pro Plan:** `claude-monitor --plan pro`
*   **Daily View:** `claude-monitor --view daily`
*   **Set Timezone:** `claude-monitor --timezone America/Los_Angeles`

## ✨ Features & How It Works

This project provides a real-time terminal monitor for tracking your Claude AI token usage.  It leverages machine learning to predict limits and burn rates, and offers a range of customizable features:

*   **[Full feature list here](#-key-features)**
*   **[Understand how Claude Sessions Work](#understanding-claude-sessions)**
*   **[Token Limits by Plan](#token-limits-by-plan)**
*   **[Smart Detection Features](#smart-detection-features)**

## 🔧 Development Installation

See [Development Installation](#-development-installation) for contributing and testing.

## 🤝 Contributors

See [Contributors](#-contributors) for the full list.

## 📚 Additional Resources

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

MIT License - see the [License](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>