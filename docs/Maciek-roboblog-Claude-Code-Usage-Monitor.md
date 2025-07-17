# 🚀 Claude Code Usage Monitor: Stay Ahead of Your Token Limits 

**Tired of unexpected Claude AI session cutoffs?**  This real-time terminal monitor gives you complete control over your token usage with advanced analytics, intelligent predictions, and a beautiful Rich UI.  Check out the original repo for details: [Maciek-roboblog/Claude-Code-Usage-Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **Real-time Monitoring:** Track token consumption, burn rate, and session limits in real-time with configurable refresh rates.
*   **ML-Powered Predictions:** Get intelligent session limit predictions based on machine learning, including P90 percentile calculations and smart detection.
*   **Advanced Rich UI:** Enjoy a beautiful, color-coded terminal interface with progress bars, tables, and WCAG-compliant contrast.
*   **Automated Plan Management:** Smart auto-detection of plans and limits for optimal efficiency, automatically switching to a 'custom' plan.
*   **Detailed Cost Analytics:** Understand your spending with model-specific pricing and cache token calculations.
*   **Configuration & Logging:** Type-safe Pydantic-validated configuration, comprehensive logging options, and optional Sentry integration.

---

## Installation

### ⚡ Modern Installation with uv (Recommended)

**Why uv is the best choice:**
- ✅ Creates isolated environments automatically (no system conflicts)
- ✅ No Python version issues
- ✅ No "externally-managed-environment" errors
- ✅ Easy updates and uninstallation
- ✅ Works on all platforms

```bash
# Install directly from PyPI with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with pip

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers (pipx, conda/mamba)

Refer to the original README for detailed instructions.

---

## Usage

### Basic Commands

*   `claude-monitor`: Runs the monitor with default settings (Custom plan).
*   `claude-monitor --help`:  Shows a comprehensive list of command-line options.

### Configuration Options

*   **Plan Selection**:  Choose from `pro`, `max5`, `max20`, or `custom`.
*   **Custom Limits:** Set custom token limits for the `custom` plan.
*   **Timezone/Time Format:** Configure your preferred timezone and time display format.
*   **Theme Selection:** Customize the UI theme (light, dark, classic, or auto-detect).
*   **Refresh Rates:** Adjust the data refresh rate and display refresh rate for optimal performance.
*   **Logging**: Enable logging to a file for debugging.

### Available Plans

| Plan        | Token Limit     | Description                                    |
| ----------- | --------------- | ---------------------------------------------- |
| **custom**  | P90 auto-detect | Intelligent limit detection (default)          |
| **pro**     | ~19,000         | Claude Pro subscription                        |
| **max5**    | ~88,000         | Claude Max5 subscription                       |
| **max20**   | ~220,000        | Claude Max20 subscription                      |

---

## What's New in v3.0.0

*   **Complete Architecture Rewrite:** Modular design for maintainability, Pydantic validation, and comprehensive testing.
*   **Enhanced Functionality:** P90-based limit detection, model-specific cost analytics, and Rich UI improvements.
*   **New CLI Options:** Control over display refresh rate, time format, logging, and saved configurations.
*   **Breaking Changes:** Changed package name, default plan, and minimum Python version.

---

## Understanding How It Works

The Claude Code Usage Monitor is built with a modular design which is split up into three layers:

*   **🖥️ User Interface Layer**: Handles CLI, Settings, Errors, and Rich Terminal UI
*   **🎛️ Monitoring Orchestrator**: Manages central control, data flow, and analytics
*   **🏗️ Foundation Layer**: Contains Core Models, Analysis Engine, and Terminal Themes

**Data Flow:** Claude Config Files → Data Layer → Analysis Engine → UI Components → Terminal Display

### Key Features:

- Configurable update intervals with high-precision display refresh
- Multi-threaded orchestration with callback system
- Progress Bars and Data Tables
- Intelligent auto-detection terminal themes

---

## Development

See [Development Installation](#-development-installation) for setting up a development environment.  This includes testing with a test suite of **100+ test cases**.

---

## Contributing

Contributions are welcome! See the [Contributing Guide](CONTRIBUTING.md) for details.

---

## Contact

**📧 Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

---

## Acknowledgments

### Sponsors

A special thanks to our supporters who help keep this project going:

**Ed** - *Buy Me Coffee Supporter*
> "I appreciate sharing your work with the world. It helps keep me on track with my day. Quality readme, and really good stuff all around!"

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>