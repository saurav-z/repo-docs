# 🚀 Claude Code Usage Monitor: Stay Ahead of Your Claude AI Token Usage

**Tired of running out of tokens mid-session?** 🤖 Monitor and optimize your Anthropic Claude AI token usage with this powerful and intuitive terminal tool. Get real-time insights, advanced analytics, and intelligent predictions to maximize your productivity and minimize unexpected costs.  [Check out the original repo for more details!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## Key Features

*   **Real-time Monitoring:** Track token consumption, burn rate, and session limits in real-time with configurable refresh rates (0.1-20 Hz).
*   **ML-Powered Predictions:** Intelligent session limit detection and P90 percentile calculations for accurate insights.
*   **Advanced Rich UI:** Beautiful, color-coded progress bars, data tables, and layouts with WCAG-compliant contrast for optimal readability.
*   **Automatic Plan Switching:** Smart auto-detection automatically switches between plans (Pro, Max5, Max20, Custom) based on usage patterns.
*   **Custom Plan Flexibility:**  Use the "Custom" plan to monitor token, message, and cost usage tailored to your individual Claude Code sessions.
*   **Cost Analytics:** Model-specific pricing with cache token calculations for detailed cost analysis.
*   **Configurable Logging & Debugging:**  Optional file logging with configurable levels and Sentry integration for production monitoring.
*   **Performance Optimized:** Advanced caching and efficient data processing for smooth performance.

---

## 🚀 Installation

Choose your preferred method to install the Claude Code Usage Monitor:

### ⚡ Modern Installation with `uv` (Recommended)

`uv` offers the simplest and most reliable installation method.

#### Install from PyPI

```bash
uv tool install claude-monitor  # Simplest
```

#### Install from Source

```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
uv tool install .
```

#### First-Time `uv` Users

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart your terminal after installation.
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor
```

**Important:** If `claude-monitor` command not found, add `~/.local/bin` to your PATH.

**Important Notes:** On modern Linux distros, consider `uv` or a virtual environment to avoid "externally-managed-environment" errors. See the troubleshooting section.

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within a conda environment)

---

## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Basic Usage

*   Simply run: `claude-monitor` (or use command aliases like `cmonitor`, `ccmonitor`, or `ccm`).  Press Ctrl+C to exit gracefully.

### Configuration Options

*   **Plan Selection:** `claude-monitor --plan <pro|max5|max20|custom>`
*   **View Type:** `claude-monitor --view <realtime|daily|monthly>`
*   **Theme:** `claude-monitor --theme <light|dark|classic|auto>`
*   **Timezone:** `claude-monitor --timezone <Your Timezone>` (e.g., `America/New_York`, `UTC`)
*   **Refresh Rate:** `claude-monitor --refresh-rate <seconds>` & `--refresh-per-second <Hz>`
*   **Logging & Debugging:** `--log-level <DEBUG|INFO|WARNING|ERROR|CRITICAL>`, `--log-file <path>`, `--debug`
*   **Clear Configuration:** `claude-monitor --clear`

**Key Features:**
- ✅ Automatic parameter persistence between sessions
- ✅ CLI arguments always override saved settings
- ✅ Atomic file operations prevent corruption
- ✅ Graceful fallback if config files are damaged
- ✅ Plan parameter never saved (must specify each time)

---

## 🚀 What's New in v3.0.0

*   **Complete Architecture Rewrite:** Modular design, Pydantic validation, comprehensive testing.
*   **Enhanced Features:** ML-based limit detection, updated plan limits, and improved UI.
*   **New CLI Options:** `--refresh-per-second`, `--time-format`, `--custom-limit-tokens`, `--log-file`, and command aliases.
*   **Breaking Changes:** Package name change, default plan change (to `custom`), Python 3.9+ minimum.

---

## 💻 Development Installation

For contributing or development:

1.  **Clone:** `git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git`
2.  **Install:** `pip install -e .` (inside a virtual environment)
3.  **Run:** `python -m claude_monitor`

---

## Troubleshooting

[See detailed solutions and common issues in the Troubleshooting section of the original README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor#troubleshooting).

---

## 📞 Contact

Reach out with questions or suggestions:  [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

---

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

---

## 📝 License

[MIT License](LICENSE)

---

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

---
<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>