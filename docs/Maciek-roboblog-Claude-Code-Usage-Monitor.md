# 🚀 Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

Tired of exceeding your Claude AI token limits?  [Track and analyze your Claude usage](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor) with this powerful terminal tool, featuring real-time monitoring, machine learning-based predictions, and a beautiful Rich UI.

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

<p align="center">
    <img src="https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png" alt="Claude Token Monitor Screenshot" width="80%">
</p>

---

## 🔑 Key Features

*   ✅ **Real-time Monitoring:** Track token usage, burn rate, and costs in real-time. Configurable refresh rates and intelligent display updates.
*   🔮 **AI-Powered Predictions:** Get machine learning-based predictions for session limits, burn rate analysis and estimated session expiration.
*   📊 **Advanced Rich UI:**  Beautiful, color-coded progress bars, sortable tables, and WCAG-compliant contrast for optimal readability.
*   🤖 **Smart Auto-Detection:**  Automatic plan switching with custom limit discovery and terminal theme detection.
*   📈 **Cost Analytics:**  Model-specific pricing with cache token calculations.
*   🚀 **Custom Plan Support:** Personalized token limits tailored to your usage.
*   ⚠️ **Advanced Warning System:** Multi-level alerts with cost and time predictions to avoid surprises.

## 🚀 Installation

Choose your preferred method:

### ⚡ Modern Installation with `uv` (Recommended)

`uv` provides the easiest and most reliable installation.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install claude-monitor
uv tool install claude-monitor

# Run the monitor
claude-monitor # or cmonitor, ccmonitor, ccm
```

### 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# Run the monitor
claude-monitor # or cmonitor, ccmonitor, ccm
```

>   **Important**: If the `claude-monitor` command is not found after pip install, ensure your `~/.local/bin` directory is in your `PATH`.

### 🛠️ Alternative Installation Methods

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within a conda environment)

## 📖 Usage

### Basic Usage

```bash
claude-monitor # Launches the monitor with default settings
```

### ⚙️ Configuration

Customize the monitor's behavior with command-line arguments:

| Parameter           | Description                                   | Default      |
| :------------------ | :-------------------------------------------- | :----------- |
| `--plan`            | Subscription plan (pro, max5, max20, custom)   | `custom`     |
| `--custom-limit-tokens` | Token limit for custom plan                 | *None*       |
| `--view`            | View type (realtime, daily, monthly)        | `realtime`   |
| `--timezone`        | Timezone (e.g., UTC, America/New_York)       | `auto`       |
| `--time-format`     | Time format (12h, 24h, auto)                  | `auto`       |
| `--theme`           | Display theme (light, dark, classic, auto)   | `auto`       |
| `--refresh-rate`    | Data refresh rate (seconds)                 | `10`         |
| `--refresh-per-second` | Display refresh rate (Hz)                  | `0.75`       |
| `--reset-hour`      | Daily reset hour (0-23)                     | *None*       |
| `--log-level`       | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO`       |
| `--log-file`        | Log file path                                 | *None*       |
| `--debug`           | Enable debug logging                         | `False`      |
| `--clear`           | Clear saved configuration                    | `False`      |

#### Example:
```bash
claude-monitor --plan pro --theme dark --timezone "America/Los_Angeles" --refresh-rate 5 --debug
```

### 📊 Available Plans

| Plan       | Token Limit       | Best For                         |
| :--------- | :---------------- | :------------------------------- |
| **custom** | P90 Auto-Detect   | Intelligent limit detection     |
| **pro**    | ~19,000           | Claude Pro subscription        |
| **max5**   | ~88,000           | Claude Max5 subscription       |
| **max20**  | ~220,000          | Claude Max20 subscription      |

### 💡 Command Aliases

For quicker access, use any of these commands:

*   `claude-monitor` (Primary)
*   `claude-code-monitor` (Full name)
*   `cmonitor` (Short)
*   `ccmonitor` (Short alternative)
*   `ccm` (Shortest)

## ✨ Features & How It Works

### 🔄 Real-time Monitoring

*   Configurable update intervals (1-60 seconds)
*   High-precision display refresh (0.1-20 Hz)
*   Intelligent change detection to minimize CPU usage
*   Multi-threaded orchestration with callback system

### 🔮 ML-Based Predictions

*   **P90 Calculator**:  90th percentile analysis for intelligent limit detection
*   **Burn Rate Analytics**: Multi-session consumption pattern analysis
*   **Cost Projections**:  Model-specific pricing with cache token calculations
*   **Session Forecasting**: Predicts session expiration based on usage patterns

### 🤖 Smart Auto-Detection

*   **Background Detection**:  Automatically determines terminal theme (light/dark)
*   **System Integration**:  Auto-detects timezone and time format preferences
*   **Plan Recognition**:  Analyzes usage patterns to suggest optimal plans
*   **Limit Discovery**:  Scans historical data to find actual token limits

### ⚙️ Understanding Claude Sessions

Claude Code operates on a **5-hour rolling session window system**:

*   **Session Start**: Begins with your first message to Claude
*   **Session Duration**: Lasts exactly 5 hours from that first message
*   **Token Limits**: Apply within each 5-hour session window
*   **Multiple Sessions**: Can have several active sessions simultaneously
*   **Rolling Windows**: New sessions can start while others are still active

## 🔧 Development Installation

Follow these steps to contribute or modify the code:

1.  **Clone the Repository:** `git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git`
2.  **Navigate to the Directory:** `cd Claude-Code-Usage-Monitor`
3.  **Create and Activate a Virtual Environment:**  `python3 -m venv venv` (or similar) and then `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
4.  **Install Dependencies:** `pip install -e .`
5.  **Run tests:** `python -m pytest tests`
6.  **Run the monitor** `python -m claude_monitor`

## 📞 Contact

For questions, suggestions, or collaboration:  [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

*   **Ed:** *Buy Me Coffee Supporter* - "I appreciate sharing your work with the world...Quality readme, and really good stuff all around!"

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">
    **⭐  If you find this project helpful, please star the repository! ⭐**
    <br>
    [Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)
</div>