# 🚀 Claude Code Usage Monitor: Real-time Tracking & AI-Powered Predictions

**Effortlessly monitor and optimize your Anthropic Claude AI token usage with this powerful, real-time terminal monitoring tool.  [View the original repository here](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).**

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

Are you tired of guessing your Claude AI token usage? This tool provides beautiful real-time monitoring, advanced analytics, machine learning-based predictions, and a rich terminal UI to track your token consumption, burn rate, cost analysis, and session limit predictions.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **🔮 ML-Based Predictions:**  Intelligent session limit detection and P90 percentile calculations.
*   **🔄 Real-time Monitoring:** Configurable refresh rates (0.1-20 Hz) with smart display updates.
*   **📊 Advanced Rich UI:**  Beautiful color-coded progress bars, tables, and WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:**  Automatic plan switching and custom limit discovery.
*   **📋 Enhanced Plan Support:** Updated limits for Pro, Max5, Max20, and Custom plans.
*   **⚠️ Advanced Warning System:**  Multi-level alerts with cost and time predictions.
*   **📈 Cost Analytics:**  Model-specific pricing with cache token calculations.
*   **⚡ Performance Optimized:** Advanced caching and efficient data processing.
*   **📝 Comprehensive Logging:** Optional file logging with configurable levels for detailed usage tracking.

## Installation

### ⚡ Recommended: Modern Installation with `uv`

`uv` offers the fastest and easiest installation experience, creating isolated environments automatically.

**Install from PyPI:**
```bash
uv tool install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

**Install from Source:**
```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
uv tool install .
claude-monitor
```

**First-time uv users**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh #Linux/macOS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" #Windows
#Restart your terminal after installing uv
```

### 📦 Alternative: Installation with `pip`

```bash
pip install claude-monitor
# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Alternative: Installation with Package Managers
```bash
#pipx (Isolated Environments)
pipx install claude-monitor

#conda/mamba
pip install claude-monitor
```

## Usage

### Get Help
```bash
claude-monitor --help
```
### Basic Usage
```bash
claude-monitor # with default (custom plan with auto-detection)
```
### Configuration Options

*   **Plan Selection:** `--plan pro|max5|max20|custom`
*   **Custom Limit:** `--custom-limit-tokens <int>`
*   **View Options:** `--view realtime|daily|monthly`
*   **Timezone:** `--timezone <timezone>` (e.g., UTC, America/New_York)
*   **Time Format:** `--time-format 12h|24h|auto`
*   **Theme:** `--theme light|dark|classic|auto`
*   **Refresh Rate:** `--refresh-rate <seconds>` & `--refresh-per-second <Hz>`
*   **Reset Hour:** `--reset-hour <0-23>`
*   **Logging:** `--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL` & `--log-file <path>`

### Saved Configuration
The monitor automatically saves your preferences for future sessions. The preferences that are saved include:
- View type (--view)
- Theme preferences (--theme)
- Timezone settings (--timezone)
- Time format (--time-format)
- Refresh rates (--refresh-rate, --refresh-per-second)
- Reset hour (--reset-hour)
- Custom token limits (--custom-limit-tokens)

The configuration file path is: ~/.claude-monitor/last_used.json

### Plan Options

| Plan        | Token Limit | Description                                                                    |
| ----------- | ----------- | ------------------------------------------------------------------------------ |
| **custom**  | P90-based   | Intelligent limit detection (default)                                          |
| **pro**     | ~19,000     | Suitable for Claude Pro subscription                                           |
| **max5**    | ~88,000     | Designed for Claude Max5 subscription                                           |
| **max20**   | ~220,000    | Optimized for Claude Max20 subscription                                          |

## 🚀 What's New in v3.0.0

Major updates include a complete architecture rewrite for improved performance and reliability, enhanced functionality with machine learning predictions, added CLI options for fine-tuning the monitoring experience, and comprehensive testing for a robust and stable tool.

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>