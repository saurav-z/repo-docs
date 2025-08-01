# 🚀 Claude Code Usage Monitor: Real-time Tracking for AI Token Consumption

**Stay in control of your Claude AI token usage with the Claude Code Usage Monitor!** This powerful, open-source tool provides real-time monitoring, advanced analytics, and intelligent predictions to help you optimize your AI workflow.  [Check out the original repo!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

The Claude Code Usage Monitor provides a beautiful, real-time terminal interface to track your Claude AI token consumption, burn rate, and costs.  Leveraging machine learning, it provides intelligent predictions about session limits, empowering you to work efficiently and avoid unexpected charges.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **Real-time Monitoring:** Track token usage, burn rate, and cost in real-time.
*   **ML-Powered Predictions:** Intelligent session limit detection and predictions based on your usage patterns.
*   **Advanced UI:** Beautiful, color-coded progress bars, tables, and WCAG-compliant contrast for easy readability.
*   **Customizable Plans:** Support for Claude Pro, Max5, Max20, and a custom plan with auto-detection.
*   **Cost Analytics:** Model-specific pricing and detailed cost breakdown.
*   **Plan Auto-Switching:** Automatically switch to the most appropriate plan based on your usage.
*   **Comprehensive Logging:** Detailed logging for debugging and analysis.
*   **Flexible Configuration:** Customize refresh rates, themes, time zones, and reset times.

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` offers the fastest and easiest installation experience, creating isolated environments automatically and resolving many common installation issues.

```bash
# Install with uv (Recommended):
uv tool install claude-monitor

# Run:
claude-monitor  # or cmonitor, ccmonitor, ccm
```

For first-time `uv` users:

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

# Run:
claude-monitor  # or cmonitor, ccmonitor, ccm
```

### 🛠️ Other Package Managers

*   **pipx:**  `pipx install claude-monitor`
*   **conda/mamba:**  `pip install claude-monitor` in your conda environment.

## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Core Commands

*   `claude-monitor`: Run the monitor with default settings (Custom plan).
*   `claude-code-monitor`, `cmonitor`, `ccmonitor`, `ccm`:  Short aliases for convenience.

### Key Command-Line Options

*   `--plan`:  Specify your Claude plan (`pro`, `max5`, `max20`, `custom`).  Defaults to `custom`.
*   `--custom-limit-tokens`:  Set a custom token limit for the custom plan.
*   `--view`: Choose your display view (`realtime`, `daily`, `monthly`). Defaults to `realtime`.
*   `--timezone`: Set your timezone (e.g., `America/New_York`, `UTC`).
*   `--time-format`: Set time format (12h or 24h), defaults to `auto`.
*   `--theme`: Choose your terminal theme (`light`, `dark`, `classic`, `auto`).
*   `--refresh-rate`: Set data refresh rate in seconds (1-60).
*   `--refresh-per-second`: Set display refresh rate in Hz (0.1-20.0).
*   `--reset-hour`: Set the daily reset hour (0-23).
*   `--log-level`: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   `--log-file`: Set log file path.
*   `--debug`: Enable debug logging.
*   `--clear`: Clear saved configuration.
*   `--version, -v`: Show version information.

### Examples

```bash
# Start with custom plan
claude-monitor

# Pro plan
claude-monitor --plan pro

# Max5 Plan
claude-monitor --plan max5

# Max20 Plan
claude-monitor --plan max20

# Override the saved settings
claude-monitor --plan pro --theme light

# Set the timezone
claude-monitor --timezone America/Los_Angeles

# Set the daily reset hour
claude-monitor --reset-hour 9
```

## 💰 Available Plans

*   **Custom (default):** P90 auto-detect based on your usage.
*   **pro:** Claude Pro subscription - ~19,000 tokens.
*   **max5:** Claude Max5 subscription - ~88,000 tokens.
*   **max20:** Claude Max20 subscription - ~220,000 tokens.

## ✨  What's New in v3.0.0

*   **Complete Architecture Rewrite:**  Improved modularity, testing, and error handling.
*   **ML-Based Limit Detection:**  90th percentile calculations for more accurate limits.
*   **Updated Plan Limits:** Accurate support for the latest plan limits.
*   **Enhanced UI:**  Improved theming and UI responsiveness.
*   **New CLI Options:**  More granular control over refresh rates and logging.
*   **Performance Optimizations:** Caching and efficiency improvements.

##  🔧 Development Installation

To contribute to the project:

```bash
# Clone the repository
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor

# Install in development mode
pip install -e .

# Run from source
python -m claude_monitor
```

Full details on development and testing are in the [Development Installation](#-development-installation) section.

## 📞 Contact

For questions, suggestions, or collaboration:

*   **Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📚  Additional Resources

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)
*   [LICENSE](LICENSE)

## 🙏 Acknowledgments

A special thanks to Ed and our other sponsors for their support!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>
```
Key improvements and optimization:

*   **SEO Focus:** Title and introduction are optimized for search terms.  Headings and formatting improve readability for both users and search engines.
*   **Concise Summary:**  The one-sentence hook is at the beginning, and the summary is direct.
*   **Key Features (Bulleted):**  Key features are now easily scannable.
*   **Installation Section:** Streamlined installation instructions using `uv` as the primary recommendation.  Pip, pipx, and conda installation covered in a clean way.
*   **Usage and Options:** Clearer examples and descriptions of command-line options.
*   **Plan Information:** Emphasis on default custom plan and plan selection strategies.
*   **What's New Section:** A concise summary of the latest changes.
*   **Development & Troubleshooting:** Kept these sections but made them concise and link to more detailed docs.
*   **Concise & Focused Language:** The text has been edited to be more direct and easier to read.
*   **Call to Actions:** Encourages starring the repo and provides direct links for reporting bugs and requesting features.
*   **Removed Redundancy:**  Removed redundant information to keep the README concise.
*   **Formatting Consistency:** Uniform use of bolding and formatting for a consistent user experience.
*   **HTML/Markdown Consistency:** Used consistent Markdown formatting for bullet points and headings.

This improved README is optimized for clarity, user experience, and search engine optimization.