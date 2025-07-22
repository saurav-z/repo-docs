# 🤖 Claude Code Usage Monitor: Real-time Token Tracking with Advanced Analytics

Tired of guessing your Claude AI token usage? **Claude Code Usage Monitor** is your go-to terminal tool for real-time tracking, helping you stay within your limits. [Check out the original repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

<br/>
<p align="center">
    <img src="https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png" alt="Claude Token Monitor Screenshot" width="800">
</p>
<br/>

## 🔑 Key Features

*   ✅ **Real-time Monitoring:** Track token consumption with configurable refresh rates (0.1-20 Hz).
*   🔮 **ML-Powered Predictions:** Get intelligent session limit detection and burn rate analysis.
*   📊 **Rich Terminal UI:**  Enjoy a beautiful, color-coded interface with WCAG-compliant contrast.
*   🤖 **Smart Auto-Detection:** Automatic plan switching and custom limit discovery for optimized usage.
*   📈 **Advanced Analytics:**  Cost analysis, model-specific pricing, and cache token calculations.
*   🛠️ **Easy Installation:**  Install quickly with `uv`, `pip`, `pipx`, or other package managers.
*   ⚙️ **Customizable:** Configure timezone, time format, themes, and logging options.

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is the fastest and most reliable way to install and manage this tool. It automatically creates isolated environments, preventing system conflicts.

```bash
# Install with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor
# if you get a warning message add ~/.local/bin to PATH as mentioned in the Installation section
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   `pipx`
*   `conda/mamba`

*(See the original README for specific installation instructions using each package manager.)*

## 📖 Usage

```bash
claude-monitor --help # Get help
```

### Key Command-Line Parameters

| Parameter | Type   | Default | Description                                    |
| :---------- | :----- | :------ | :--------------------------------------------- |
| `--plan`    | string | custom  | Plan type: pro, max5, max20, or custom        |
| ...         | ...    | ...     | ... (See original README for full list)       |

### Available Plans

| Plan        | Token Limit     | Best For                     |
| :---------- | :-------------- | :--------------------------- |
| **custom**  | P90 auto-detect | Intelligent limit detection (default) |
| **pro**     | ~19,000         | Claude Pro subscription      |
| **max5**    | ~88,000         | Claude Max5 subscription     |
| **max20**   | ~220,000        | Claude Max20 subscription    |

*(See the original README for complete information on configuration options, command aliases, and usage examples.)*

## 🚀 What's New in v3.0.0

This major update features a complete architecture rewrite, enhancing functionality and user experience.

### Major Updates

*   **Complete Architecture Rewrite**
*   **Enhanced Functionality:** ML-based limit detection, updated plan limits, cost analytics
*   **New CLI Options:** Configuration of refresh rate, time format, and logging
*   **Breaking Changes:** Default plan changed to custom

*(See the original README for detailed information on the new architecture, features, and breaking changes.)*

## ✨ Features & How It Works

### Overview

This tool provides a modular architecture to monitor your token usage with real-time updates and machine-learning powered features.

### Core Components

*   **User Interface Layer:** Pydantic-based CLI, settings, Rich Terminal UI.
*   **Monitoring Orchestrator:** Real-time data flow, session management, and UI controller.
*   **Foundation Layer:** Core models, analysis engine, Claude API data.

### Smart Detection Features

*   **Automatic Plan Switching:**  Automatically switches to the optimal plan.
*   **Limit Discovery Process:**  Analyzes past usage to determine limits.

*(See the original README for a comprehensive overview of the architecture and data flow.)*

## 🔧 Development Installation

For developers wanting to contribute or test the code:

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor

# Install in development mode
pip install -e .

# Run from source
python -m claude_monitor
```

### Testing

```bash
cd src/
python -m pytest
```

*(See the original README for prerequisites, virtual environment setup, and detailed testing information.)*

## 📞 Contact

For questions, suggestions, or collaboration:

**📧 Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📚 Additional Documentation

*   **[Development Roadmap](DEVELOPMENT.md)**
*   **[Contributing Guide](CONTRIBUTING.md)**
*   **[Troubleshooting](TROUBLESHOOTING.md)**

## 📝 License

[MIT License](LICENSE) - Use and modify freely.

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

*(See the original README for a full list of contributors.)*

## 🙏 Acknowledgments

A special thanks to our supporters who help keep this project going:
**Ed** - *Buy Me Coffee Supporter*

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>