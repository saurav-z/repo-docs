# 🚀 Claude Code Usage Monitor: Real-time AI Token Tracking and Analytics

**Tired of guessing your Claude AI token usage?** Stay in control with the Claude Code Usage Monitor, a powerful terminal tool for real-time monitoring, intelligent predictions, and cost analysis.  [Check it out on GitHub!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

<img src="https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png" alt="Claude Token Monitor Screenshot" width="700"/>

---

## ✨ Key Features

*   **📊 Real-time Monitoring:** Track your Claude AI token consumption, burn rate, and cost analysis in real-time.
*   **🔮 ML-Based Predictions:**  Intelligent session limit detection, including P90 percentile calculations, helps you stay ahead.
*   **🎨 Rich Terminal UI:** Enjoy a beautiful, color-coded, and WCAG-compliant interface for easy monitoring.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery for optimal usage.
*   **📈 Advanced Cost Analytics:** Model-specific pricing with detailed token calculations to manage your AI spending.
*   **🚀 Comprehensive Plan Support:** Supports Pro, Max5, Max20, and a customizable custom plan.

## 🚀 Installation

Choose your preferred installation method:

### ⚡ Modern Installation with `uv` (Recommended)

`uv` provides the fastest and most isolated installation.

```bash
# Install with uv
uv tool install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor
claude-monitor # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   **pipx:**  `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor`

## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Basic Usage

```bash
claude-monitor # Default custom plan with auto-detection
```

### Configuration Options

*   **Plan:**  `--plan pro`, `--plan max5`, `--plan max20`, `--plan custom` (default)
*   **Custom Token Limit:** `--custom-limit-tokens [tokens]` (for custom plan)
*   **View:** `--view realtime` (default), `--view daily`, `--view monthly`
*   **Timezone:** `--timezone [timezone]` (e.g., `America/New_York`, `UTC`)
*   **Time Format:** `--time-format 12h`, `--time-format 24h`, `--time-format auto`
*   **Theme:** `--theme light`, `--theme dark`, `--theme classic`, `--theme auto`
*   **Refresh Rate:** `--refresh-rate [seconds]` (1-60)
*   **Display Refresh Rate:** `--refresh-per-second [Hz]` (0.1-20.0)
*   **Reset Hour:** `--reset-hour [hour]` (0-23)
*   **Logging:** `--log-level DEBUG`, `--log-file [path]`
*   **Clear Saved Config:** `--clear`

## 🧠 Understanding Claude Sessions

Claude Code sessions operate on a 5-hour rolling window. The monitor tracks token usage, messages, and cost within each session. The custom plan uses your usage history to calculate P90 limits, providing accurate predictions.

## 💰 Available Plans

| Plan        | Token Limit   | Best For                                     |
|-------------|---------------|----------------------------------------------|
| **custom**  | P90 auto-detect | Intelligent limit detection (default)       |
| **pro**     | ~19,000       | Claude Pro subscription                    |
| **max5**    | ~88,000       | Claude Max5 subscription                   |
| **max20**   | ~220,000      | Claude Max20 subscription                  |

## 🚀 What's New in v3.0.0

*   **Complete Architecture Rewrite:** Modular design, Pydantic validation, SRP compliance.
*   **Enhanced Functionality:** P90 Analysis, updated plan limits, richer UI and CLI Options.
*   **Performance Optimized**: Advanced caching and efficient data processing
*   **Comprehensive testing**: 100+ test cases with full coverage
*   **Error reporting**: Sentry integration for production monitoring

## 🔧 Development Installation

For contributing:

```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
pip install -e .
python -m claude_monitor
```

## 📞 Contact

*   **Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

*   **Sponsors:** Special thanks to our supporters.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>