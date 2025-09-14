# 🚀 Claude Code Usage Monitor: Stay Ahead of Your Claude AI Token Usage

**Tired of unexpected token limits?**  The Claude Code Usage Monitor provides real-time terminal monitoring, insightful analytics, and intelligent predictions for your Anthropic Claude AI usage, helping you optimize costs and maximize productivity. [Check out the original repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)!

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)
---
![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)
---
## Key Features

*   **Real-time Monitoring:** Track token consumption, burn rate, and session limits directly in your terminal with configurable refresh rates.
*   **Advanced Analytics:** Get detailed cost analysis, model-specific pricing, and comprehensive usage reports.
*   **Intelligent Predictions:** Leverage machine learning to predict session limits and provide proactive warnings.
*   **Rich User Interface:** Enjoy a beautiful, color-coded, and WCAG-compliant terminal UI for easy readability.
*   **Smart Plan Auto-Detection:** Automatically switch between plans based on your usage patterns for optimal efficiency.
*   **Customizable:** Tailor the monitor to your specific needs with a variety of configuration options.

## Installation

Choose your preferred installation method:

### ⚡ Modern Installation with `uv` (Recommended)

This is the easiest and fastest way to install, providing automatic isolated environments and eliminating many potential Python version conflicts.

#### Install `uv`

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

#### Install & Run

```bash
uv tool install claude-monitor
claude-monitor
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```
**Important**: If `claude-monitor` is not found, add `~/.local/bin` to your `PATH`.

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal
```

### 🛠️ Other Package Managers

*   **pipx:**  `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within your conda environment)

## Usage

Run the monitor with default settings:

```bash
claude-monitor
```

View help for available options:

```bash
claude-monitor --help
```

### Configuration Options

*   **`--plan`:**  Specify your Claude AI plan (e.g., `pro`, `max5`, `max20`, `custom` - which is default)
*   **`--custom-limit-tokens`:** Set a custom token limit for the custom plan.
*   **`--view`:** Choose the view type: `realtime`, `daily`, or `monthly`.
*   **`--timezone`:** Set your timezone (e.g., `America/New_York`, `UTC`).  Defaults to auto-detection.
*   **`--time-format`:** Specify time format `12h` or `24h` or leave as `auto`.
*   **`--theme`:** Select a theme: `light`, `dark`, `classic`, or `auto`.
*   **`--refresh-rate`:** Set data refresh rate in seconds (1-60).
*   **`--refresh-per-second`:** Set the display refresh rate in Hz (0.1-20.0).
*   **`--reset-hour`:** Set the daily reset hour (0-23).
*   **`--log-level`:** Set logging level `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **`--log-file`:**  Specify a file path for logging.
*   **`--debug`:** Enable debug logging.
*   **`--clear`:** Clear saved configuration.

## Available Plans

| Plan         | Token Limit (approx.) | Best For                      |
|--------------|-----------------------|-------------------------------|
| **custom**   | P90 auto-detect       | Intelligent limit detection (default) |
| **pro**      | 19,000                | Claude Pro subscription       |
| **max5**     | 88,000                | Claude Max5 subscription      |
| **max20**    | 220,000               | Claude Max20 subscription     |

##  🚀 What's New in v3.0.0

*   **Complete Architecture Rewrite**: Improved modularity, type safety, and testing.
*   **ML-Powered Predictions**:  90th percentile analysis for intelligent limit detection.
*   **Enhanced Functionality**:  Updated plan limits and advanced logging.
*   **Rich UI**:  WCAG-compliant themes.

## Troubleshooting

*   **"externally-managed-environment" Error**:  Use `uv`,  `pipx`, or a virtual environment. See detailed instructions in the Troubleshooting section of the original README for further help.
*   **Command Not Found**: Ensure `~/.local/bin` is in your `PATH` after `pip install`.
*   **No Active Session Found**: If the monitor doesn't show usage after sending a few messages, please follow the steps outlined in the original documentation under the "No Active Session Found" section to specify a custom config path.

## 📞 Contact

For questions, suggestions, or to contribute, contact maciek@roboblog.eu.

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

A special thanks to **Ed** (Buy Me Coffee Supporter)!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---
<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>
```
Key improvements and SEO optimizations:

*   **Clear, Concise Hook:** Starts with a direct benefit for users.
*   **Target Keywords:**  Uses terms like "Claude AI," "token usage," "real-time monitoring," and "cost optimization."
*   **Headline Structure:** Uses heading tags (`<h1>`, `<h2>`, etc.) to improve readability and SEO.
*   **Bulleted Lists:**  Clearly presents key features and options.
*   **Concise Descriptions:**  Avoids overly technical language.
*   **Actionable Language:**  Uses calls to action (e.g., "Check out the original repo").
*   **Links Back:** Links to the original repo and other relevant documentation.
*   **Emphasis on Benefits:** Highlights the advantages of using the tool.
*   **Organized Structure:**  Well-organized with clear sections for installation, usage, and troubleshooting.
*   **`uv` Emphasis:** The README now leads with the `uv` installation because it is the easiest way.
*   **"What's New" Section:** Highlights major updates for SEO purposes.
*   **Contributors and Star History:** Encourages user engagement.
*   **Clear Troubleshooting Section:** Directs users to the most common solutions.
*   **SEO-friendly:** Uses keyphrases (Claude Code, token usage monitor) throughout.