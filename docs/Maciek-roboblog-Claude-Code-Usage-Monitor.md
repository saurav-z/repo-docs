# ⏱️ Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

Tired of hitting token limits?  **[Claude Code Usage Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)** is your real-time terminal companion, providing advanced analytics and AI-powered predictions to optimize your Claude AI usage.

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 🚀 Key Features

*   **Real-time Monitoring:** Track token consumption, burn rate, and cost in real-time with configurable refresh rates.
*   **AI-Powered Predictions:** Machine learning provides intelligent session limit detection, and burn rate analysis.
*   **Advanced Rich UI:**  A beautiful, color-coded terminal UI with WCAG-compliant contrast for optimal readability.
*   **Smart Auto-Detection:** Automatic plan switching and intelligent plan limit discovery tailored to your usage.
*   **Configurable & Customizable:** Set custom reset times, timezone, theme, and refresh rates.
*   **Cost Analytics:** Detailed model-specific pricing and cache token calculations.
*   **Complete Architecture Rewrite (v3.0.0):**  Improved stability, modular design, and enhanced features.

## 📦 Installation

Choose your preferred installation method:

### ⚡ Modern Installation with uv (Recommended)

**Why uv?**  Easy, isolated environments, no Python version conflicts, and simple updates.

```bash
# Install from PyPI with uv
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

Or, from source:

```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
uv tool install .
claude-monitor
```
If you don't have uv installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# or
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
```

### 📦 Installation with pip

```bash
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

**Troubleshooting:**
*   **PATH Setup:**  If you see a warning about the script installation location, follow the `export PATH` command.
*   **"externally-managed-environment" Error:** On modern Linux distributions, consider using `uv`, virtual environments, or `pipx` (see detailed solutions in the original README).

### 🛠️ Other Package Managers (pipx & conda/mamba)

Install with pipx:

```bash
pipx install claude-monitor

# Run from anywhere
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```

Or install with conda/mamba:

```bash
pip install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```
## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Basic Usage

```bash
# Default (Custom plan with auto-detection)
claude-monitor

# Alternative commands
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# Exit the monitor
# Press Ctrl+C to gracefully exit
```

### Configuration Options

Customize your monitoring experience with various command-line options:

*   **`--plan`:**  Select your Claude plan: `pro`, `max5`, `max20`, or `custom` (default).
*   **`--custom-limit-tokens`:** Set a custom token limit for the `custom` plan.
*   **`--timezone`:**  Set your timezone (e.g., `America/New_York`, `UTC`).
*   **`--time-format`:**  Choose time format: `12h`, `24h`, or `auto`.
*   **`--theme`:**  Choose a display theme: `light`, `dark`, `classic`, or `auto`.
*   **`--refresh-rate`:** Set data refresh rate (seconds).
*   **`--refresh-per-second`:** Set display refresh rate (Hz).
*   **`--reset-hour`:**  Set the daily reset hour (0-23).
*   **`--log-level`:** Set the log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **`--log-file`:** Specify a log file path.
*   **`--debug`:** Enable debug logging.
*   **`--clear`:** Clear saved configuration.

**Key Features:**
- ✅ Automatic parameter persistence between sessions
- ✅ CLI arguments always override saved settings
- ✅ Atomic file operations prevent corruption
- ✅ Graceful fallback if config files are damaged
- ✅ Plan parameter never saved (must specify each time)

#### Plan Options

| Plan        | Token Limit     | Description                                   |
| ----------- | --------------- | --------------------------------------------- |
| `pro`       | ~19,000         | Claude Pro subscription                     |
| `max5`      | ~88,000         | Claude Max5 subscription                    |
| `max20`     | ~220,000        | Claude Max20 subscription                   |
| `custom`    | P90-based      | Auto-detection with ML analysis (default)  |

### Available Plans

| Plan       | Token Limit     | Best For                     |
|------------|-----------------|------------------------------|
| **custom** | P90 auto-detect | Intelligent limit detection (default) |
| **pro**    | ~19,000         | Claude Pro subscription      |
| **max5**   | ~88,000         | Claude Max5 subscription     |
| **max20**  | ~220,000        | Claude Max20 subscription    |

### 🚀 What's New in v3.0.0

The latest update features a complete architecture rewrite, offering:

*   **Modular Design:**  Single Responsibility Principle (SRP) compliance.
*   **Type-Safe Configuration:** Pydantic-based configuration for robust validation.
*   **ML-Powered Limit Detection:** 90th percentile calculations for intelligent limit detection.
*   **Enhanced Rich UI:**  WCAG-compliant themes and auto-detection of terminal background.
*   **New Command Line Options:**  Configurable display refresh rates, time formats, and more.
*   **Updated Plan Limits:**  Pro (44k), Max5 (88k), Max20 (220k) tokens.

## ✨ Features & How It Works

### v3.0.0 Architecture Overview

The new version features a complete rewrite with modular architecture following Single Responsibility Principle (SRP):

### 🖥️ User Interface Layer

| Component            | Description           |
| -------------------- | --------------------- |
| **CLI Module**       | Pydantic-based        |
| **Settings/Config**  | Type-safe             |
| **Error Handling**   | Sentry-ready          |
| **Rich Terminal UI** | Adaptive Theme        |

---

### 🎛️ Monitoring Orchestrator

| Component                | Key Responsibilities                                             |
| ------------------------ | ---------------------------------------------------------------- |
| **Central Control Hub**  | Session Mgmt · Real-time Data Flow · Component Coordination      |
| **Data Manager**         | Cache Mgmt · File I/O · State Persist                           |
| **Session Monitor**      | Real-time · 5 hr Windows · Token Track                           |
| **UI Controller**        | Rich Display · Progress Bars · Theme System                     |
| **Analytics**            | P90 Calculator · Burn Rate · Predictions                        |

---

### 🏗️ Foundation Layer

| Component           | Core Features                                           |
| ------------------- | ------------------------------------------------------- |
| **Core Models**     | Session Data · Config Schema · Type Safety             |
| **Analysis Engine** | ML Algorithms · Statistical · Forecasting              |
| **Terminal Themes** | Auto-detection · WCAG Colors · Contrast Opt            |
| **Claude API Data** | Token Tracking · Cost Calculator · Session Blocks      |

---

**🔄 Data Flow:**
Claude Config Files → Data Layer → Analysis Engine → UI Components → Terminal Display

### Current Features

*   **Advanced Real-time Monitoring:** Configurable update intervals, high-precision display refresh, intelligent change detection.
*   **Rich UI Components:** Progress bars, data tables, and adaptive layouts.
*   **Machine Learning Predictions:**  P90 analysis, burn rate analytics, cost projections, and session forecasting.
*   **Intelligent Auto-Detection:** Automatic theme detection, system integration, plan recognition, and limit discovery.

### Smart Detection Features

*   **Automatic Plan Switching:**  Monitor detects usage, analyzes sessions, and switches to `custom_max` if needed.
*   **Limit Discovery Process:**  Scans your history, finds peak usage, validates data, and sets limits based on machine learning.

### Understanding Claude Sessions

Claude Code operates on a **5-hour rolling session window system** where token limits apply within each session.

## 🔧 Development Installation

For development and contributing, follow the development installation instructions in the original README.

## 📞 Contact

For questions, suggestions, or collaboration, contact [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## 📚 Additional Documentation

*   **[Development Roadmap](DEVELOPMENT.md)** - ML features, PyPI package, Docker plans
*   **[Contributing Guide](CONTRIBUTING.md)** - How to contribute, development guidelines
*   **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

## 📝 License

[MIT License](LICENSE) - Use and modify freely.

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

Want to contribute? Check out our [Contributing Guide](CONTRIBUTING.md)!

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