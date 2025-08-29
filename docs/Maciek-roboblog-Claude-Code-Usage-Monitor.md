# 🚀 Claude Code Usage Monitor: Real-Time Token Tracking & AI-Powered Predictions

**Tired of exceeding your Claude AI token limits?**  [Monitor your Claude AI token usage in real-time](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor) with advanced analytics, machine learning-based predictions, and a beautiful terminal UI to optimize your usage and costs.

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## 🔑 Key Features

*   **Real-Time Monitoring:** Configurable refresh rates with intelligent display updates to see your token usage at a glance.
*   **ML-Powered Predictions:** Get accurate session limit detection and intelligent predictions based on your usage patterns.
*   **Advanced Rich UI:** Beautiful, color-coded progress bars, tables, and layouts optimized for WCAG-compliant contrast.
*   **Smart Auto-Detection:** Automatic plan switching with custom limit discovery to maximize your Claude AI plan.
*   **Cost Analytics:** Model-specific pricing and cache token calculations to track your spending.
*   **Customizable Plans:**  Supports Pro, Max5, Max20, and a dynamic "Custom" plan for personalized limits.
*   **Comprehensive Logging & Debugging:**  Optional file logging and Sentry integration for production monitoring.

## 🚀 Installation

Choose the best installation method for your needs:

### ⚡ Recommended: Modern Installation with `uv`

`uv` is a blazing fast Python package and virtual environment manager, making installation a breeze.

```bash
# Install with uv
uv tool install claude-monitor

# Run
claude-monitor
```

### 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# Add to PATH if needed (See Troubleshooting)
# Run from anywhere
claude-monitor # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   **`pipx`:** `pipx install claude-monitor`
*   **`conda/mamba`:** `pip install claude-monitor` (inside a conda environment)

---

## 📖 Usage

### 📚 Get Help

```bash
claude-monitor --help
```

### 🛠️ Basic Usage

```bash
# Run the monitor with default settings
claude-monitor

# Specify your Claude AI plan
claude-monitor --plan pro # for pro plan
claude-monitor --plan max5 # for max5 plan
claude-monitor --plan max20 # for max20 plan
claude-monitor --plan custom # for custom plan
```

### ⚙️ Configuration Options

*   **`--plan`**:  `pro`, `max5`, `max20`, or `custom` (default).
*   **`--custom-limit-tokens`**: Custom token limit for the `custom` plan.
*   **`--view`**:  `realtime` (default), `daily`, `monthly`.
*   **`--timezone`**: Your timezone (auto-detected).
*   **`--time-format`**: `12h`, `24h`, or `auto`.
*   **`--theme`**: `light`, `dark`, `classic`, or `auto`.
*   **`--refresh-rate`**: Refresh rate in seconds (1-60).
*   **`--refresh-per-second`**: Display refresh rate in Hz (0.1-20.0).
*   **`--reset-hour`**: Daily reset hour (0-23).
*   **`--log-level`**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **`--log-file`**: Log file path.
*   **`--debug`**: Enable debug logging.
*   **`--clear`**: Clear saved configuration.

### 📊 Available Plans

| Plan       | Token Limit     | Best For                          |
|------------|-----------------|-----------------------------------|
| **custom** | P90 auto-detect | Intelligent limit detection       |
| **pro**    | ~19,000         | Claude Pro subscription          |
| **max5**   | ~88,000         | Claude Max5 subscription         |
| **max20**  | ~220,000        | Claude Max20 subscription        |

## ✨ Features & How It Works

This tool provides a comprehensive overview of your Claude AI usage, including real-time monitoring, trend analysis, and predictive capabilities.  Key features include:

*   **Real-time Dashboard:** See your token usage, burn rate, cost, and estimated session duration at a glance.
*   **Advanced Analytics:** Analyze daily and monthly token usage to identify usage patterns and potential cost savings.
*   **Intelligent Predictions:**  ML-powered algorithms predict when your sessions will expire and suggest optimal plans.
*   **Automatic Plan Switching:** The tool intelligently detects when you are approaching your Pro plan limits and suggests a custom plan if it improves limits.

## 💻 Development Installation

If you're interested in contributing or modifying the code, follow these steps:

```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
pip install -e .
python -m claude_monitor
```

## 🐛 Troubleshooting

*   **Installation Issues:**  Consult the Troubleshooting section in the original README for solutions to common installation problems, especially related to "externally-managed-environment" errors and PATH issues.
*   **No active session found:** Verify that you've interacted with Claude AI to start a session before running the monitor.

## 📞 Contact

Have questions, suggestions, or want to contribute?  Contact the developer at [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributors

Thanks to our contributors:

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

Special thanks to our supporters!

**Ed** - *Buy Me Coffee Supporter*

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>