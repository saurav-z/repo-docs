# 🚀 Claude Code Usage Monitor: Stay Ahead of Your AI Token Limits

**Tired of unexpected limits in Claude AI?** This terminal-based tool provides real-time monitoring, intelligent predictions, and advanced analytics to help you optimize your Claude AI usage. [View on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## 🌟 Key Features

*   **Real-time Monitoring:** Track token consumption, burn rate, and cost in real-time with configurable refresh rates.
*   **Advanced Analytics:** Visualize usage with color-coded progress bars, tables, and customizable views (Realtime, Daily, Monthly).
*   **ML-Powered Predictions:** Get intelligent session limit detection and predictions based on machine learning.
*   **Smart Auto-Detection:**  Automatically switches between custom and other plans to optimize your usage.
*   **Customizable Plans:** Supports Claude Pro, Max5, Max20, and custom plans, including advanced P90 auto-detection.
*   **Cost Analytics:** Monitor model-specific costs with cache token calculations.
*   **Configuration:** Save your preferences for a consistent and easy-to-use experience.

## 📦 Installation

### 🚀 **Recommended: Modern Installation with `uv` (Fastest & Easiest)**

`uv` automatically creates isolated environments, avoiding conflicts and Python version issues.

```bash
# Install with uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Claude Monitor
uv tool install claude-monitor

# Run
claude-monitor
```

### 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# Run
claude-monitor
```

### 🛠️ Other Installation Methods

*   **`pipx`:** `pipx install claude-monitor` (isolated environments)
*   **conda/mamba:** `pip install claude-monitor` (within conda environment)

## 📖 Usage

### Basic Commands
```bash
claude-monitor # Run with default custom plan
cmonitor  # Short alias
ccmonitor # Short alternative
ccm       # Shortest alias
```

### Command-line Parameters

*   **`--plan`:** (`pro`, `max5`, `max20`, `custom`) - Sets the plan. Default: `custom`.
*   **`--custom-limit-tokens`:**  (Integer) - Token limit for the custom plan.
*   **`--view`:** (`realtime`, `daily`, `monthly`) - Sets the view. Default: `realtime`.
*   **`--timezone`:** (Timezone string) -  Sets the timezone (e.g., `America/New_York`, `UTC`). Auto-detected by default.
*   **`--time-format`:** (`12h`, `24h`) - Time format. Auto-detected by default.
*   **`--theme`:** (`light`, `dark`, `classic`, `auto`) -  Theme. Default: `auto`.
*   **`--refresh-rate`:** (Integer) - Refresh rate in seconds (1-60).
*   **`--refresh-per-second`:** (Float) - Display refresh rate in Hz (0.1-20.0)
*   **`--reset-hour`:** (Integer) - Daily reset hour (0-23).
*   **`--log-level`:** (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) - Logging level. Default: `INFO`.
*   **`--log-file`:** (Path) - Log file path.
*   **`--debug`:** (Flag) - Enable debug logging.
*   **`--version, -v`:** (Flag) - Show version information.
*   **`--clear`:** (Flag) - Clear saved configuration.

### Plan Options and Limits
| Plan | Token Limit | Description |
|---|---|---|
| **custom** | P90 auto-detect | Intelligent limit detection (default) |
| **pro** | ~19,000 | Claude Pro subscription |
| **max5** | ~88,000 | Claude Max5 subscription |
| **max20** | ~220,000 | Claude Max20 subscription |

### Configuration Examples
```bash
# Run Pro plan, dark theme, and set time zone
claude-monitor --plan pro --theme dark --timezone "America/New_York"

# Subsequent runs will restore the settings
claude-monitor --plan pro
```
### Clear all saved preferences
```bash
claude-monitor --clear
```

## ✨ Features & How It Works

###  Real-Time Monitoring

Track token usage, burn rate, cost, and receive warnings as you approach your limits.

###  Advanced Features

*   **Automatic Plan Switching:** Auto-detects usage patterns and suggests optimal plans.
*   **ML-Based Predictions:** Uses machine learning for accurate token limit predictions.
*   **Rich UI:** The terminal interface features  WCAG-compliant color schemes with scientific contrast ratios

### Architecture Overview

v3.0.0 is built with a modular, Single Responsibility Principle (SRP) architecture.

*   **User Interface Layer**
*   **Monitoring Orchestrator**
*   **Foundation Layer**

## 🚀 Example Usage

### Quick Start
```bash
claude-monitor
```
### Monitor your usage in different time zone
```bash
claude-monitor --view daily --timezone America/New_York
```

### Usage View Configuration
```bash
# Real-time monitoring with live updates (Default)
claude-monitor --view realtime

# Daily token usage aggregated in table format
claude-monitor --view daily

# Monthly token usage aggregated in table format
claude-monitor --view monthly
```

## 🔧 Development Installation

1.  Clone the repository: `git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git`
2.  `cd Claude-Code-Usage-Monitor`
3.  Install:
    *   **Recommended**:  `pip install -e .` (after activating a virtual environment)
    *   Or `python3 -m venv venv && source venv/bin/activate && pip install -e .`
4.  Run: `python -m claude_monitor`

## 📞 Contact

For questions, suggestions, or collaboration, contact:

*   **Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

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

Special thanks to our supporters!

---
<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>