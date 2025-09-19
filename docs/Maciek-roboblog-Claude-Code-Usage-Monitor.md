# 🚀 Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

**Keep tabs on your Claude AI token usage with ease!**  [Check out the original repo here](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

This powerful terminal tool provides real-time monitoring of your Claude AI token consumption, along with advanced analytics, machine learning-based predictions, and a rich, customizable user interface. Track your token usage, burn rate, and costs, plus receive intelligent predictions about your session limits.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 🔑 Key Features

*   **🔮 ML-Based Predictions:** Get accurate token limit predictions based on your usage patterns.
*   **🔄 Real-time Monitoring:** Customizable refresh rates (0.1-20 Hz) with intelligent display updates.
*   **📊 Advanced Rich UI:** Beautifully designed, color-coded progress bars, tables, and layouts with WCAG-compliant contrast for optimal readability.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery to optimize your experience.
*   **📋 Enhanced Plan Support:** Updated limits for Pro (19k), Max5 (88k), Max20 (220k), and Custom plans.
*   **⚠️ Advanced Warning System:** Multi-level alerts with cost and time predictions to avoid overspending.
*   **💼 Professional Architecture:**  Built with a modular design that follows the Single Responsibility Principle (SRP).
*   **🎨 Intelligent Theming:** Scientific color schemes and automatic terminal background detection.
*   **⏰ Advanced Scheduling:** Auto-detected system timezone and time format preferences.
*   **📈 Cost Analytics:** Model-specific pricing and cache token calculations.
*   **📝 Comprehensive Logging:** Optional file logging with configurable levels for in-depth analysis.
*   **🧪 Extensive Testing:** Over 100 test cases with full coverage for robust reliability.
*   **🎯 Error Reporting:** Optional Sentry integration for proactive production monitoring.
*   **⚡ Performance Optimized:** Advanced caching and efficient data processing for minimal resource consumption.

## 🚀 Installation

### ⚡ Modern Installation with uv (Recommended)

**Why uv is the best choice:**

*   ✅ Creates isolated environments automatically (no system conflicts)
*   ✅ No Python version issues
*   ✅ No "externally-managed-environment" errors
*   ✅ Easy updates and uninstallation
*   ✅ Works on all platforms

The fastest and easiest way to install and use the monitor:

[![PyPI](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)

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
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

> **⚠️ PATH Setup**: If you see WARNING: The script claude-monitor is installed in '/home/username/.local/bin' which is not on PATH, follow the export PATH command above.
> **⚠️ Important**: On modern Linux distributions (Ubuntu 23.04+, Debian 12+, Fedora 38+), you may encounter an "externally-managed-environment" error. Instead of using --break-system-packages, we strongly recommend:
> 1. **Use uv instead** (see above) - it's safer and easier
> 2. **Use a virtual environment** - python3 -m venv myenv && source myenv/bin/activate
> 3. **Use pipx** - pipx install claude-monitor
> See the Troubleshooting section for detailed solutions.

### 🛠️ Other Package Managers

#### pipx (Isolated Environments)

```bash
# Install with pipx
pipx install claude-monitor

# Run from anywhere
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```

#### conda/mamba

```bash
# Install with pip in conda environment
pip install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

## 📖 Usage

### Get Help

```bash
# Show help information
claude-monitor --help
```

### Configuration Options

The monitor offers a variety of configuration options for a personalized experience:

*   **--plan**:  Choose your plan (pro, max5, max20, or custom)
*   **--custom-limit-tokens**: Set a custom token limit for the custom plan.
*   **--view**: Select the display view (realtime, daily, or monthly).
*   **--timezone**: Set your timezone (auto-detected by default).
*   **--time-format**: Choose your preferred time format (12h or 24h).
*   **--theme**: Customize the display theme (light, dark, classic, or auto).
*   **--refresh-rate**: Set the data refresh rate in seconds (1-60).
*   **--refresh-per-second**: Adjust the display refresh rate in Hz (0.1-20.0).
*   **--reset-hour**: Set the daily reset hour (0-23).
*   **--log-level**: Configure the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   **--log-file**: Specify a log file path for detailed logging.
*   **--debug**: Enable debug logging for troubleshooting.
*   **--version, -v**: Display version information.
*   **--clear**: Clear saved configuration settings.

### Basic Usage

1.  **uv Installation (Recommended)**

```bash
# Run the monitor using the default settings
claude-monitor

# Or using short aliases
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# To Exit the monitor
# Press Ctrl+C to gracefully exit
```

2.  **Development mode**

```bash
If running from source, use python -m claude_monitor from the src/ directory.
```

## 🤝 Contributors

-   [@adawalli](https://github.com/adawalli)
-   [@taylorwilsdon](https://github.com/taylorwilsdon)
-   [@moneroexamples](https://github.com/moneroexamples)

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

*   **Ed** - *Buy Me Coffee Supporter*: "I appreciate sharing your work with the world. It helps keep me on track with my day. Quality readme, and really good stuff all around!"

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>