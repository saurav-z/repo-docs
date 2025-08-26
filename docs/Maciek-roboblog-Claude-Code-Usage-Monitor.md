# 🚀 Claude Code Usage Monitor: Stay Ahead of Your Claude AI Token Limits

**Tired of running out of tokens unexpectedly?** Claude Code Usage Monitor is a powerful terminal tool that provides real-time monitoring, advanced analytics, and intelligent predictions for your Claude AI token usage.  **[Check out the original repository](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)** to start optimizing your Claude AI sessions today!

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

**Key Features:**

*   **Real-time Monitoring:** Track token consumption, burn rate, and session limits.
*   **ML-Powered Predictions:**  Intelligent session limit detection and cost analysis.
*   **Advanced Rich UI:** Beautiful, color-coded terminal interface with WCAG-compliant contrast.
*   **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **Cost Analytics:** Model-specific pricing and cache token calculations.
*   **Configurable:** Set refresh rates, timezones, themes, and logging options.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## 🔑 Core Features & Benefits

*   **P90 Percentile Analysis:** Leverage machine learning to understand your token usage patterns and receive the most accurate predictions.
*   **Custom Plan Adaptability:** The default "Custom" plan learns from your usage, providing tailored limits and warnings.
*   **Real-time Performance:** Enjoy configurable refresh rates (0.1-20 Hz) for up-to-the-second monitoring.
*   **Multiple Usage Views:** Monitor your token consumption with both real-time and aggregated daily and monthly stats.
*   **Optimized Cost Efficiency:**  Track cost consumption alongside token usage, so you can budget and allocate accordingly.
*   **Customizable Settings:** Easily configure your preferences with command-line options for time zones, themes, and log levels.
*   **Flexible Installation:** Install via `uv`, `pip`, `pipx`, or `conda/mamba` to suit your development environment.

## 💻 Installation

### ⚡ Modern Installation with uv (Recommended)

`uv` is the easiest and fastest way to install and use the monitor, automatically creating isolated environments.

1.  **Install `uv` (if you don't have it):**

    *   **Linux/macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
    *   **Windows:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

2.  **Install Claude Monitor:**

    ```bash
    # Install directly from PyPI with uv
    uv tool install claude-monitor

    # Run from anywhere
    claude-monitor  # or cmonitor, ccmonitor for short
    ```

3.  **Install from Source:**

    ```bash
    # Clone and install from source
    git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
    cd Claude-Code-Usage-Monitor
    uv tool install .

    # Run from anywhere
    claude-monitor
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

**Important Notes:**

*   **PATH Setup:**  If you see a warning about the script not being on your PATH, follow the `export PATH` command.
*   **"externally-managed-environment" Errors:** Consider `uv` or a virtual environment if you encounter this on modern Linux distributions. See the Troubleshooting section for details.

### 🛠️ Other Package Managers

Install with `pipx` (isolated environments):
```bash
pipx install claude-monitor
claude-monitor
```

Install with `conda/mamba`:
```bash
pip install claude-monitor
claude-monitor
```

## 📖 Usage Guide

### Basic Commands

```bash
# Start the monitor with default settings (Custom plan)
claude-monitor

# Get help
claude-monitor --help

# Exit gracefully
# Press Ctrl+C to gracefully exit
```

### Configuration Options

Customize your monitoring with command-line flags:

*   **Plan Selection:** `--plan pro|max5|max20|custom`
*   **Custom Limit:** `--custom-limit-tokens [tokens]`
*   **View Type:** `--view realtime|daily|monthly`
*   **Timezone:** `--timezone [timezone]` (e.g., `America/New_York`, `UTC`)
*   **Time Format:** `--time-format 12h|24h`
*   **Theme:** `--theme light|dark|classic|auto`
*   **Refresh Rate:** `--refresh-rate [seconds]` & `--refresh-per-second [Hz]`
*   **Reset Hour:** `--reset-hour [hour]` (0-23)
*   **Logging:** `--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL` & `--log-file [path]`
*   **Debug Mode:** `--debug`
*   **Clear Configuration:** `--clear`
*   **Show version:** `--version, -v`

### Command Aliases

The tool can be invoked using any of these commands:

- claude-monitor (primary)
- claude-code-monitor (full name)
- cmonitor (short)
- ccmonitor (short alternative)
- ccm (shortest)

### Saving Preferences

The monitor saves your settings, so you don't have to re-enter them every time. Preferences are stored in `~/.claude-monitor/last_used.json`. Use the `--clear` option to reset saved configuration.

## ⚙️ Understanding Plans & Limits

*   **Custom (Default):** P90-based auto-detection for intelligent limit setting based on your usage patterns.
*   **pro:**  ~19,000 tokens
*   **max5:** ~88,000 tokens
*   **max20:** ~220,000 tokens

## ✨ Key Features in v3.0.0

*   **Complete Architecture Rewrite:**  Modular design for maintainability.
*   **P90 Analysis:** Machine learning-based limit detection.
*   **Updated Plan Limits:** Pro (19k), Max5 (88k), Max20 (220k).
*   **Cost Analytics:** Model-specific pricing.
*   **Rich UI:** Adaptive themes and improved display refresh.
*   **Enhanced Configuration:** New CLI options.
*   **More robust Testing:** More than 100 test cases

## 🛠️ Troubleshooting

*   **"externally-managed-environment" Error:** Use `uv`, `pipx`, or a virtual environment.
*   **Command Not Found:**  Verify your PATH or use the full path to the script.
*   **Python Version Conflicts:** Use the correct Python version (e.g., `python3.11 -m ...`).

## 📞 Contact

For questions, suggestions, or collaboration, contact:

**📧 Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📚 Additional Resources

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)
*   [LICENSE](LICENSE)
*   [Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

### Sponsors

A special thanks to our supporters who help keep this project going:

**Ed** - *Buy Me Coffee Supporter*
> "I appreciate sharing your work with the world. It helps keep me on track with my day. Quality readme, and really good stuff all around!"

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>