# 🚀 Claude Code Usage Monitor: Real-time AI Token Tracking and Prediction

**Effortlessly monitor and manage your Anthropic Claude AI token usage with this powerful terminal tool.** ([View on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor))

This README provides a comprehensive guide to help you understand, install, and utilize the Claude Code Usage Monitor, offering real-time insights, advanced analytics, and smart predictions to optimize your AI token consumption.

## 🔑 Key Features

*   **Real-time Monitoring:** Track token usage, cost, and burn rate with configurable refresh intervals.
*   **ML-Based Predictions:** Intelligent session limit detection and cost projections.
*   **Advanced UI:** Rich, color-coded terminal interface with WCAG-compliant contrast.
*   **Automated Plan Switching:** Smart plan detection based on your usage patterns.
*   **Customizable Plans:** Support for Pro, Max5, Max20, and custom limits.
*   **Comprehensive Analytics:** Daily and monthly usage views, cost analysis, and burn rate.
*   **User-Friendly Configuration:** Save and override preferences easily via CLI.
*   **Robust Architecture:** Modular design, comprehensive testing, and error handling.

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` offers the easiest and most reliable installation:

*   **Automatic Isolation:** Prevents conflicts and system issues.
*   **Python Version Agnostic:** Handles Python versions seamlessly.
*   **Effortless Updates:** Easy to update and uninstall.
*   **Cross-Platform Compatibility:** Works on all major platforms.

1.  **Install `uv`:** (Follow platform-specific instructions below)

    **Linux/macOS:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    **Windows:**

    ```powershell
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Install Claude Monitor:**

    ```bash
    uv tool install claude-monitor
    ```

3.  **Run the Monitor:**

    ```bash
    claude-monitor  # Or cmonitor, ccmonitor for short
    ```

### 📦 Installation with `pip`

```bash
pip install claude-monitor

# If command not found, add to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc # or restart terminal

claude-monitor # Run
```

>   **Important:**  Modern Linux users might encounter "externally-managed-environment" errors.  Consider `uv`, a virtual environment, or `pipx` for best results. See [Troubleshooting](#troubleshooting)

### 🛠️ Other Package Managers

*   **pipx:**  `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within your conda env)

## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Basic Usage

```bash
claude-monitor  # Default: Custom plan, real-time view
```

### Configuration Options

*   **`--plan`**: pro, max5, max20, custom
*   **`--custom-limit-tokens`**:  Token limit for custom plan
*   **`--view`**: realtime, daily, monthly
*   **`--timezone`**:  Timezone (e.g., UTC, America/New_York). Auto-detected by default
*   **`--time-format`**:  12h, 24h, or auto
*   **`--theme`**:  light, dark, classic, or auto
*   **`--refresh-rate`**:  Data refresh rate (seconds)
*   **`--refresh-per-second`**: Display refresh rate in Hz
*   **`--reset-hour`**: Daily reset hour (0-23)
*   **`--log-level`**:  DEBUG, INFO, WARNING, ERROR, CRITICAL
*   **`--log-file`**: Log file path
*   **`--debug`**: Enable debug logging
*   **`--version, -v`**:  Show version
*   **`--clear`**: Clear saved configuration

### Plan Options

| Plan        | Token Limit | Best For                               |
| ----------- | ----------- | -------------------------------------- |
| `custom`    | P90-based   | Intelligent limit detection (default) |
| `pro`       | ~19,000     | Claude Pro subscription               |
| `max5`      | ~88,000     | Claude Max5 subscription              |
| `max20`     | ~220,000    | Claude Max20 subscription             |

## 🚀 What's New in v3.0.0

*   **Complete Architecture Rewrite:** Modular design, Pydantic validation, SRP principles.
*   **ML-Powered Limit Detection:**  P90 percentile analysis.
*   **Updated Plan Limits:** Accurate limits for Claude's plans.
*   **Enhanced UI:** Richer terminal interface with configurable refresh and display settings.
*   **New CLI Options**:  Customizable logging and configuration.
*   **Breaking Changes**: Package renamed to `claude-monitor`, minimum Python 3.9.

## ✨ Features & How It Works

*   **Real-time Monitoring:** Configurable refresh rates and intelligent display updates for efficient resource usage.
*   **Advanced UI**: Rich color-coded progress bars, tables, and layouts with WCAG contrast compliance.
*   **Plan Switching**:  Automatic switching with custom limit discovery.
*   **Cost Analytics**: Model-specific pricing with token calculations.
*   **P90 Analysis**:  Machine learning-based limit detection with 90th percentile calculations.
*   **Burn Rate Analysis**: Multi-session consumption analysis
*   **Understanding Claude Sessions**:  Guidance on Claude session mechanics and optimal usage.

## 🚀 Usage Examples

*   **Basic Usage**: `claude-monitor`
*   **Specify Plan**: `claude-monitor --plan max5`
*   **Configure View**: `claude-monitor --view daily`
*   **Set Timezone**: `claude-monitor --timezone America/New_York`
*   **Custom Reset Times**: `claude-monitor --reset-hour 9`
*   **Daily and Monthly Usage**: View token usage patterns, trends, and budgets with daily and monthly views.

## 🔧 Development Installation

For contributors and developers:

1.  **Clone the repository:** `git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git`
2.  **Navigate to the directory:** `cd Claude-Code-Usage-Monitor`
3.  **Install in development mode**: `pip install -e .`
4.  **Run from source**: `python -m claude_monitor`
5.  **Testing**: Comprehensive test suite available for all components. Use `pytest` for execution.

See [DEVELOPMENT.md](DEVELOPMENT.md) for additional information.

## Troubleshooting

*   **"externally-managed-environment" Error:** Use `uv`, `pipx`, or a virtual environment.
*   **Command Not Found:**  Ensure the correct PATH setup.  See [Installation](#installation).
*   **Runtime Issues:** Review error messages for specific solutions.
*   **No active session found:** Ensure Claude Code has sent at least two messages to the API.

## 📞 Contact

*   **Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

*   **Sponsors:** Special thanks to **Ed** for their support.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>