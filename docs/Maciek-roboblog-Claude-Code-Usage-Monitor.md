# 🚀 Claude Code Usage Monitor: Real-time Token Tracking and AI-Powered Predictions

**Effortlessly monitor and analyze your Claude AI token usage with real-time insights and intelligent predictions.** [View the original repository](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

This powerful, open-source terminal tool provides a beautiful real-time view of your Claude AI token consumption, burn rate, and cost analysis.  It goes beyond basic tracking by leveraging machine learning to provide intelligent predictions about session limits and offers a user-friendly Rich UI.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **🔮 ML-Based Predictions:** Intelligent session limit detection based on the 90th percentile of your usage.
*   **🔄 Real-time Monitoring:** Configurable refresh rates (0.1-20 Hz) with smart UI updates.
*   **📊 Rich Terminal UI:**  Visually appealing, color-coded progress bars, tables, and layouts for easy understanding.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery to maximize your efficiency.
*   **📋 Enhanced Plan Support:**  Supports Claude Pro, Max5, Max20 plans, and a customizable plan.
*   **⚠️ Advanced Warning System:** Multi-level alerts with cost and time predictions to prevent unexpected overages.
*   **🎨 Intelligent Theming:** Automatic terminal background detection for optimal readability and a consistent experience.
*   **📈 Cost Analytics:** Model-specific pricing with cache token calculations for accurate cost tracking.
*   **📝 Comprehensive Logging:** Optional file logging with configurable levels for detailed usage analysis.
*   **⚡ Performance Optimized:** Advanced caching and efficient data processing for a smooth, responsive experience.
*   **v3.0.0 Architecture Rewrite:** Modular design with Single Responsibility Principle (SRP) compliance, Pydantic validation, and extensive testing.

## Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is the fastest and easiest way to install and use the monitor.

```bash
# Install from PyPI with uv
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

For first-time `uv` users, install `uv` with a single command:

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

### 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

>   **⚠️ PATH Setup:** If you see a PATH warning after install, follow the `export PATH` command.
>
>   **⚠️ Important:** On modern Linux distributions, avoid `--break-system-packages` and consider using `uv`, a virtual environment, or `pipx` instead.

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` within a conda environment.

## Usage

### Basic Usage

```bash
# Run the monitor with default settings
claude-monitor
```

Press `Ctrl+C` to exit gracefully.

### Configuration Options

*   **`--plan`**: `pro`, `max5`, `max20`, or `custom` (default).
*   **`--custom-limit-tokens`**:  Token limit for the custom plan.
*   **`--view`**: `realtime` (default), `daily`, or `monthly`.
*   **`--timezone`**: Timezone (auto-detected).  Examples: `UTC`, `America/New_York`.
*   **`--time-format`**: `12h`, `24h`, or `auto`.
*   **`--theme`**: `light`, `dark`, `classic`, or `auto`.
*   **`--refresh-rate`**: Data refresh rate in seconds (1-60).
*   **`--refresh-per-second`**: Display refresh rate in Hz (0.1-20.0).
*   **`--reset-hour`**: Daily reset hour (0-23).
*   **`--log-level`**: Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **`--log-file`**: Log file path.
*   **`--debug`**: Enable debug logging.
*   **`--clear`**: Clear saved configuration.

### Available Plans and Limits

*   **custom:** P90 auto-detect, intelligent limit detection (default)
*   **pro:** ~19,000 tokens, best for Claude Pro subscription
*   **max5:** ~88,000 tokens, best for Claude Max5 subscription
*   **max20:** ~220,000 tokens, best for Claude Max20 subscription

### Saved Preferences

The monitor automatically saves your settings (view, theme, timezone, refresh rates, custom limits) to avoid re-specifying them each time.  Use `--clear` to reset. Configuration is located in `~/.claude-monitor/last_used.json`.

## Key Features & Functionality Breakdown

This version features a complete rewrite with modular architecture following the Single Responsibility Principle (SRP):

### Core Components

*   **UI Layer:** Command-line interface with Pydantic-based settings, and a Rich Terminal UI.
*   **Monitoring Orchestrator:** Central control hub for session management and real-time data flow.
*   **Foundation Layer:** Core models, analysis engine with ML algorithms, terminal themes, and Claude API data integration.

###  Real-Time Monitoring

-   **🔄 Advanced Real-time Monitoring**: Configurable update intervals (1-60 seconds) for session tracking.
-   **📊 Rich UI Components**: Optimized color-coded progress bars, data tables, and a theme system.
-   **📈 Multiple Usage Views**:  Realtime, daily, and monthly views for analysis.
-   **🔮 Machine Learning Predictions**: P90 Calculator for 90th percentile analysis, Burn Rate Analytics for consumption patterns, cost projections.
-   **🤖 Intelligent Auto-Detection**: Automatic plan switching.

### How It Works: Data Flow

Claude Config Files → Data Layer → Analysis Engine → UI Components → Terminal Display

### Understanding Claude Sessions

-   Sessions operate on a 5-hour rolling window system.
-   Token limits apply within each 5-hour window.
-   Multiple sessions can run simultaneously.

### Token Limits by Plan (v3.0.0)

| Plan          | Limit (Tokens) | Cost Limit | Messages | Algorithm        |
|---------------|----------------|------------|----------|------------------|
| **Claude Pro** | 19,000         | $18.00     | 250      | Fixed limit      |
| **Claude Max5** | 88,000         | $35.00     | 1,000    | Fixed limit      |
| **Claude Max20** | 220,000        | $140.00    | 2,000    | Fixed limit      |
| **Custom**     | P90-based      | (default) $50.00 | 250+     | Machine learning |

### Smart Detection Features

-   **Automatic Plan Switching**:  The monitor can detect when you're exceeding Pro limits and switch to the custom plan.
-   **Limit Discovery**: Auto-detection system finds your highest token usage and adapts.

## 🚀 Usage Examples

### Common Scenarios

*   **Morning Developer:** `claude-monitor --reset-hour 9 --timezone US/Eastern`
*   **Night Owl Coder:**  `claude-monitor --reset-hour 0`
*   **Heavy User with Variable Limits:** `claude-monitor --plan custom`
*   **International User:** `claude-monitor --timezone Europe/London`
*   **Quick Check:** `claude-monitor`
*   **Usage Analysis Views:** `claude-monitor --view daily`

### Plan Selection Strategies

*   Start with the default (`claude-monitor`), to let it auto-detect and switch.
*   Use the `max5` or `max20` plan flags, if you know your subscription.
*   Use the custom flag, if you are unsure of your limits.

### Best Practices

*   **Start early** in your sessions for accurate tracking.
*   **Use `uv`** for installation.
*   **Monitor burn rate** and plan strategically.
*   **Use timezones.**

## 🔧 Development Installation

For developers and contributors:

```bash
# Clone the repository
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor

# Install in development mode
pip install -e .

# Run from source
python -m claude_monitor
```

## Troubleshooting

*   **Installation Errors:** Follow solutions provided in the "Installation Issues" section, especially those concerning "externally-managed-environment" errors. If issues persist, check the Troubleshooting section.
*   **Runtime Issues**: If you see the "No active session found" error, try the following: send at least two messages to Claude in a new session.

## 📞 Contact

*   **Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

### Sponsors

*   **Ed** - *Buy Me Coffee Supporter*

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>