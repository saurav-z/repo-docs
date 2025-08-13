# 🚀 Claude Code Usage Monitor: Stay Ahead of Your AI Token Usage

**Tired of surprises with your Claude AI token usage?** 🤖 Get real-time insights, smart predictions, and detailed analytics with the Claude Code Usage Monitor!  [Check out the original repo for more details.](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

This powerful, real-time terminal monitoring tool provides advanced analytics, machine learning-based predictions, and a rich, user-friendly UI for Claude AI token usage. Track your token consumption, analyze your burn rate, monitor your costs, and receive intelligent predictions to avoid session limit surprises.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   ✅ **Real-time Monitoring:** Configurable refresh rates (0.1-20 Hz) with intelligent display updates.
*   📊 **Advanced Rich UI:** Beautiful color-coded progress bars, tables, and WCAG-compliant contrast.
*   🤖 **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   🔮 **ML-Based Predictions:** P90 percentile calculations and intelligent session limit detection.
*   📈 **Cost Analytics:** Model-specific pricing and cache token calculations.
*   ⚠️ **Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   🔄 **Enhanced Plan Support:** Updated limits: Pro (19k), Max5 (88k), Max20 (220k), and Custom (P90-based).
*   💼 **Professional Architecture:** Modular design with Single Responsibility Principle (SRP) compliance.

### v3.0.0: Major Update Highlights

*   **Complete Architecture Rewrite**: Enhanced functionality and testability.
*   **Updated Plan Limits**: Pro (19k), Max5 (88k), Max20 (220k).
*   **Custom plan**: Default option with intelligent limit detection using P90 analysis based on past usage patterns
*   **Machine Learning Based Predictions**: Estimate and track usage patterns accurately based on history

## Installation

### ⚡ Modern Installation with uv (Recommended)

**Why uv?**
*   Creates isolated environments automatically.
*   No Python version conflicts.
*   Easy updates and uninstallation.
*   Works on all platforms.

#### Install from PyPI

```bash
# Install directly from PyPI with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

#### Install from Source

```bash
# Clone and install from source
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
uv tool install .

# Run from anywhere
claude-monitor
```

#### First-time uv users
If you don't have uv installed yet, get it with one command:

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
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

> **⚠️ PATH Setup**: If you see WARNING: The script claude-monitor is installed in '/home/username/.local/bin' which is not on PATH, follow the export PATH command above.
>
> **⚠️ Important**: On modern Linux distributions, using `pip install --break-system-packages` is NOT recommended. Consider using `uv`, a virtual environment, or `pipx`.  See [Troubleshooting](#troubleshooting) for more details.

### 🛠️ Other Package Managers

*   **pipx**: `pipx install claude-monitor`
*   **conda/mamba**: `pip install claude-monitor` (within your conda environment)

## Usage

### Basic Usage

```bash
# Run with default settings (Custom plan)
claude-monitor
# Or use short command alternatives
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# Exit the monitor
# Press Ctrl+C to gracefully exit
```

### Configuration Options

*   **Plan:**  `--plan pro`, `--plan max5`, `--plan max20`, `--plan custom`
*   **View:** `--view realtime`, `--view daily`, `--view monthly`
*   **Timezone:** `--timezone America/New_York`, `--timezone UTC`, etc.  (Auto-detects by default.)
*   **Theme:** `--theme dark`, `--theme light`, `--theme auto`
*   **Refresh Rate:** `--refresh-rate 5` (seconds) or `--refresh-per-second 1.0` (Hz)
*   **Custom Reset Hour:** `--reset-hour 3` (0-23)
*   **Logging:** `--log-level DEBUG`, `--log-file /path/to/log.txt`
*   **Clear Saved Settings:** `--clear`

### Plan Options

| Plan        | Token Limit     | Best For                               |
| ----------- | --------------- | -------------------------------------- |
| **custom**  | P90 auto-detect | Intelligent limit detection (default) |
| **pro**     | ~19,000         | Claude Pro subscription                |
| **max5**    | ~88,000         | Claude Max5 subscription               |
| **max20**   | ~220,000        | Claude Max20 subscription              |

**The custom plan intelligently adapts to your usage patterns, offering personalized limits based on your session history.**

## What's New in v3.0.0

*   **Complete Architecture Rewrite**: Improves code maintainability and testability.
*   **P90 Analysis**: Machine Learning based limit detection using 90th percentile calculations.
*   **Updated Plan Limits**: Support for the latest Claude AI plans.
*   **New CLI Options**: Configure refresh rates, themes, time formats, and custom limits.
*   **Command Aliases**:  Easier to use with `cmonitor`, `ccmonitor`, and `ccm`.

## Understanding Claude Sessions

Claude Code operates on a **5-hour rolling session window system**:

*   **Session Start**: Begins with your first message to Claude
*   **Session Duration**: Lasts exactly 5 hours from that first message
*   **Token Limits**: Apply within each 5-hour session window
*   **Multiple Sessions**: Can have several active sessions simultaneously
*   **Rolling Windows**: New sessions can start while others are still active

## Contact

Need help, have suggestions, or want to collaborate?

*   **Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md) - Future ML features, PyPI package, and Docker plans.
*   [Contributing Guide](CONTRIBUTING.md) - How to contribute and development guidelines.
*   [Troubleshooting](TROUBLESHOOTING.md) - Common issues and their solutions.

## License

[MIT License](LICENSE)

## Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## Acknowledgments

### Sponsors

A special thanks to our supporters who help keep this project going:

**Ed** - *Buy Me Coffee Supporter*

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>