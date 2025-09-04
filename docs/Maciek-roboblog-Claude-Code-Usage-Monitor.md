# 🚀 Claude Code Usage Monitor: Track & Optimize Your AI Token Usage

Are you tired of exceeding your Claude AI token limits unexpectedly?  **Claude Code Usage Monitor** is your solution for real-time token tracking and smart AI usage optimization.  Get proactive insights and prevent costly overages with our intuitive terminal-based tool!  [Explore the original repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

Track your Claude AI token consumption with a beautiful, real-time terminal monitor that provides advanced analytics, machine learning-based predictions, and a rich user interface.  Get insights into your token burn rate, cost analysis, and receive intelligent predictions about session limits to help you optimize your usage.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **📊 Real-time Monitoring:** Configurable refresh rates with dynamic display updates.
*   **🔮 ML-Powered Predictions:** Intelligent session limit detection & cost projections.
*   **✅ Smart Plan Auto-Detection:** Automatically adjusts to your usage with custom limit discovery.
*   **🎨 Rich Terminal UI:** Color-coded progress bars, sortable tables, and WCAG-compliant themes.
*   **📈 Cost Analytics:** Model-specific pricing with detailed token calculations.
*   **🚀 Customizable:** Adjust timezone, themes, and reset hours with CLI options.
*   **💪 Robust Architecture:** Modular design for easy customization and maintenance.

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` provides the fastest and easiest installation, ensuring isolated environments and seamless updates:

1.  **Install `uv`**:

    *   **Linux/macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
    *   **Windows:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

2.  **Install the Monitor**:

    ```bash
    uv tool install claude-monitor
    ```

3.  **Run**:

    ```bash
    claude-monitor
    ```

### 📦 Installation with `pip`

```bash
pip install claude-monitor

#  If claude-monitor command not found, add ~/.local/bin to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

claude-monitor  # or cmonitor, ccmonitor
```

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:**  `pip install claude-monitor`  (within a conda environment)

## 📖 Usage

### Basic Usage

*   Run with default settings: `claude-monitor`
*   Use short aliases: `cmonitor`, `ccmonitor`, `ccm`

### Command-Line Parameters (Key Options)

*   `--plan`: (Default: `custom`) pro, max5, max20, or custom.
*   `--custom-limit-tokens`: Token limit for the `custom` plan.
*   `--view`: realtime, daily, or monthly usage views.
*   `--timezone`: Auto-detect, or specify (e.g., UTC, America/New_York).
*   `--time-format`: 12h, 24h, or auto.
*   `--theme`: light, dark, classic, or auto.
*   `--refresh-rate`: Data refresh rate in seconds (1-60).
*   `--debug`: Enable debugging mode.

### Available Plans & Limits

| Plan          | Token Limit  | Description                                                                                                                                                               |
| :------------ | :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **custom**    | P90-based    | Adaptive limits based on your usage history, leveraging 90th percentile calculations. (Default)                                                                           |
| **pro**       | ~19,000      | For the Claude Pro subscription                                                                                                                                             |
| **max5**      | ~88,000      | For the Claude Max5 subscription                                                                                                                                            |
| **max20**     | ~220,000     | For the Claude Max20 subscription                                                                                                                                           |

## ⚙️  Configuration & Customization

*   **Save Preferences**:  The monitor saves your settings (view, theme, timezone, refresh rates, reset hour) for subsequent runs.
*   **Override Saved Settings**:  Use CLI arguments to adjust settings on each run.
*   **Clear Saved Settings**:  Use the `--clear` flag to reset your configuration.
*   **Timezone:**  Specify your local time for accurate monitoring using `--timezone`.

## 🔍 Understanding Claude Sessions

*   **5-Hour Rolling Window**: Claude Code sessions operate within a 5-hour window from your first message.
*   **Session Management**:  Monitor allows for multiple active sessions.
*   **Burn Rate Calculation**: The monitor estimates token consumption per minute.

## 🔧 Development Installation

1.  **Clone the repository:** `git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git`
2.  **Navigate to the directory:** `cd Claude-Code-Usage-Monitor`
3.  **Install in development mode:** `pip install -e .`
4.  **Run the monitor:** `python -m claude_monitor`
5. **Testing:** Use python -m pytest to run the test suite.

## 📞 Get in Touch

*   **Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📝 License

MIT License - Use and modify freely.

## 🙏 Acknowledgments

Thank you to all contributors and Ed (Buy Me Coffee Supporter).

---

<div align="center">
**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)
</div>