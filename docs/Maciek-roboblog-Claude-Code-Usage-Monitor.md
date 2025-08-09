# 🚀 Claude Code Usage Monitor: Stay Ahead of Your Claude AI Token Usage

**Tired of guessing your Claude AI token usage?**  Monitor your token consumption in real-time with the **Claude Code Usage Monitor** ([Original Repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)), a powerful terminal tool packed with advanced analytics, machine learning-based predictions, and a beautiful Rich UI.  Track your token consumption, burn rate, cost analysis, and receive intelligent predictions about your session limits.

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 🔑 Key Features: Your Real-Time Claude AI Companion

*   **Real-time Monitoring:**  Track token usage, burn rate, and costs as your sessions progress.
*   **ML-Powered Predictions:** Get intelligent predictions about session limits using P90 percentile calculations.
*   **Advanced Rich UI:** Beautiful color-coded progress bars, tables, and layouts for intuitive monitoring.
*   **Smart Auto-Detection:** Automatically recognizes your plan, customizes your limits, and adapts to your usage.
*   **Customizable and Versatile:** Configure refresh rates, time zones, themes, and logging options.
*   **Cost Analytics:**  Model-specific pricing with cache token calculations for accurate budgeting.

### 🚀  Key Benefits:

*   **Stay Within Limits:** Prevent unexpected overages and manage your Claude AI budget effectively.
*   **Optimize Your Usage:**  Understand your token consumption patterns and make data-driven decisions.
*   **Enhance Productivity:**  Maximize your Claude AI session time with informed resource management.
*   **Effortless Setup:**  Get started quickly with modern installation options.

## 📦 Installation Guide

### ⚡ Option 1: Modern Installation with `uv` (Recommended)

`uv` offers the easiest and most reliable installation experience, handling virtual environments automatically.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Claude Monitor
uv tool install claude-monitor

# Run the Monitor
claude-monitor
```

### 📦 Option 2: Installation with `pip`

```bash
pip install claude-monitor
```

If you encounter issues with the `claude-monitor` command, add the `~/.local/bin` directory to your `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal
```

### 🛠️ Other Installation Options:

*   **`pipx` (Isolated Environments):** `pipx install claude-monitor`
*   **`conda/mamba`:** Install with `pip` in your conda environment.

## 📖  Getting Started: Using the Monitor

###  Basic Usage:

```bash
claude-monitor  # or cmonitor, ccmonitor for short
```

*   Press `Ctrl+C` to exit the monitor.

### ⚙️ Configuration Options:

*   **`--plan`**:  Choose your plan (e.g., `pro`, `max5`, `max20`, `custom`).
*   **`--custom-limit-tokens`**: Set a specific token limit for the `custom` plan.
*   **`--view`**:  Switch between `realtime`, `daily`, or `monthly` views.
*   **`--timezone`**: Set your timezone (e.g., `America/New_York`).
*   **`--time-format`**: Set time format: 12h or 24h.
*   **`--theme`**: Choose a theme: `light`, `dark`, `classic`, or `auto`.
*   **`--refresh-rate`**: Set data refresh rate (seconds).
*   **`--log-file`**: Specify a log file path.
*   **`--log-level`**: Set log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
*   **`--clear`**:  Clear your saved configuration.

### 📄 Plan Options:

| Plan        | Token Limit (approx.) | Best For                            |
|-------------|-----------------------|-------------------------------------|
| **custom**  | P90 auto-detect       | Intelligent limit detection (default) |
| **pro**     | ~19,000               | Claude Pro subscription             |
| **max5**    | ~88,000               | Claude Max5 subscription            |
| **max20**   | ~220,000              | Claude Max20 subscription           |

## ✨ New in v3.0.0: A Complete Rewrite

*   **Major Architecture Rewrite:** Modern, modular design following the Single Responsibility Principle.
*   **Machine Learning for Limit Detection:**  Utilizing P90 analysis for intelligent, personalized limits.
*   **Updated Plan Limits:**  Support for the latest Claude AI plan offerings.
*   **Enhanced Functionality:**  Advanced cost analytics, Rich UI improvements.
*   **User-Friendly Features:** Improved CLI options for customization and control.

## 🛠️ Development & Contribution

For developers and contributors:

### Development Installation

```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
pip install -e .
python -m claude_monitor
```

### Comprehensive Testing

v3.0.0 includes a robust testing suite:

```bash
cd src/
python -m pytest
# Run with coverage
python -m pytest --cov=claude_monitor --cov-report=html

```

### Helpful Links

*   **[Development Roadmap](DEVELOPMENT.md)** - ML features, PyPI package, Docker plans
*   **[Contributing Guide](CONTRIBUTING.md)** - How to contribute, development guidelines
*   **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

A special thanks to **Ed** for his generous support!

## 📞 Contact

Questions or suggestions?  Reach out to: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## ⭐ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐  Show your support!  Star this repo if you find it helpful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>