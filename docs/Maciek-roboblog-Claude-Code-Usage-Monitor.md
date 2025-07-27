# 🚀 Claude Code Usage Monitor: Stay Ahead of Your AI Token Usage!

**Tired of unexpectedly hitting those Claude AI limits?** 🛑 Track your token consumption in real-time with the **Claude Code Usage Monitor**! Get intelligent predictions, advanced analytics, and a beautiful terminal UI. Manage your AI costs and stay focused on your work.  Check out the original repo [here](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

<br>

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

<br>

## ✨ Key Features

*   **🔮 ML-based Predictions:** Get 90th percentile calculations and intelligent session limit detection.
*   **🔄 Real-time Monitoring:** Track token usage with configurable refresh rates (0.1-20 Hz).
*   **📊 Advanced Rich UI:** Enjoy a beautiful terminal UI with color-coded progress bars and WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Automatic plan switching with custom limit discovery.
*   **📋 Enhanced Plan Support:** Updated limits for Pro, Max5, Max20, and Custom plans.
*   **⚠️ Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   **📈 Cost Analytics:** Model-specific pricing with cache token calculations.

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

**`uv` offers isolated environments, no Python version issues, easy updates, and works on all platforms.**

```bash
# Install with uv
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

## 📖 Usage

### Basic Usage

```bash
# Run the monitor with default settings (Custom plan)
claude-monitor
```

### Configuration Options

*   **`--plan`**: pro, max5, max20, custom (default)
*   **`--view`**: realtime (default), daily, monthly
*   **`--timezone`**: auto (default), or specify a timezone (e.g., `America/New_York`)
*   **`--theme`**: auto (default), light, dark, classic
*   **`--refresh-rate`**:  (1-60 seconds, default: 10)
*   **`--log-level`**:  DEBUG, INFO, WARNING, ERROR, CRITICAL
*   **`--clear`**: Clear saved configuration

**Key Features:**
- ✅ Automatic parameter persistence between sessions
- ✅ CLI arguments always override saved settings
- ✅ Atomic file operations prevent corruption
- ✅ Graceful fallback if config files are damaged
- ✅ Plan parameter never saved (must specify each time)

### Example Commands:

*   `claude-monitor --plan pro --theme dark --timezone "America/New_York"`
*   `claude-monitor --view daily`
*   `claude-monitor --clear`

## 📊 Available Plans

| Plan       | Token Limit | Best For                    |
|------------|-------------|-----------------------------|
| **custom** | P90 auto-detect   | Intelligent limit detection (default) |
| **pro**    | ~19,000       | Claude Pro subscription     |
| **max5**   | ~88,000       | Claude Max5 subscription    |
| **max20**  | ~220,000      | Claude Max20 subscription   |

## 🚀 What's New in v3.0.0

*   **Complete Architecture Rewrite**: Modular design following Single Responsibility Principle (SRP), Pydantic-based configuration, and comprehensive testing.
*   **Enhanced Functionality**: P90 analysis, updated plan limits, cost analytics, and a rich terminal UI.
*   **New CLI Options**:  `--refresh-per-second`, `--time-format`, `--custom-limit-tokens`, `--log-file`, `--log-level`, and `--clear`.
*   **Breaking Changes**: Package name change, default plan changed to custom, and minimum Python version increase.

## 🔧 Development Installation

```bash
# Clone the repository
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor

# Install in development mode
pip install -e .

# Run from source
python -m claude_monitor
```

## 📞 Contact

Reach out with questions, suggestions, or collaborations!

**📧 Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📚 Additional Documentation

-   [Development Roadmap](DEVELOPMENT.md)
-   [Contributing Guide](CONTRIBUTING.md)
-   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

[MIT License](LICENSE) - use and modify as needed.

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

### Sponsors

A special thanks to our supporters who help keep this project going:

**Ed** - *Buy Me Coffee Supporter*
> "I appreciate sharing your work with the world. It helps keep me on track with my day. Quality readme, and really good stuff all around!"

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>