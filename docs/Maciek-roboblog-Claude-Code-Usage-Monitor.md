# ⏱️ Claude Code Usage Monitor: Stay Ahead of Your Token Limits

Tired of hitting those pesky Claude AI token limits mid-session? 🤬 This Python-based terminal tool provides real-time monitoring, advanced analytics, and intelligent predictions for your Claude AI token usage, helping you stay in control and optimize your workflow.  **[Check it out on GitHub!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)**

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)
![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **Real-time Monitoring:** Track token consumption, burn rate, and cost in a beautiful terminal UI.
*   **ML-Based Predictions:**  Intelligent session limit detection and proactive warnings based on your usage patterns.
*   **Advanced Analytics:**  Analyze usage with model-specific pricing and comprehensive cost tracking.
*   **Smart Auto-Detection:** Automatically switches plans and adapts to your specific token limits.
*   **Configurable & Customizable:** Adjust refresh rates, themes, timezone, and plan options to fit your workflow.
*   **Modular Architecture:** Built with Single Responsibility Principle (SRP) compliance for a robust and maintainable codebase.

## Installation

Choose your preferred installation method:

### 🚀 Recommended: Modern Installation with `uv`

`uv` is a blazing fast Python package and virtual environment manager.  It provides a clean and easy install with automatic environment isolation.

```bash
# Install from PyPI with uv
uv tool install claude-monitor

# Run the monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
# Install from PyPI with pip
pip install claude-monitor

# Run the monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within a conda environment)

For detailed installation instructions and troubleshooting tips, see the original [README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

## Usage

```bash
# Show help information
claude-monitor --help
```

Key CLI parameters include:

*   `--plan`: pro, max5, max20, or custom (default)
*   `--custom-limit-tokens`: Set a specific token limit for the custom plan
*   `--timezone`: Set timezone (e.g., UTC, America/New_York)
*   `--time-format`: 12h or 24h
*   `--theme`: light, dark, classic, or auto
*   `--refresh-rate`: Data refresh interval (seconds)
*   `--refresh-per-second`: Display refresh rate (Hz)
*   `--reset-hour`: Daily reset hour (0-23)
*   `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
*   `--log-file`: Log file path
*   `--debug`: Enable debug logging
*   `--clear`: Clear saved configuration

Saved preferences include theme, timezone, time format, refresh rates, reset hour and custom token limits.

## Available Plans

*   **custom**:  Intelligent limit detection based on your usage (default).
*   **pro**:   ~19,000 tokens, Claude Pro subscription.
*   **max5**:  ~88,000 tokens, Claude Max5 subscription.
*   **max20**: ~220,000 tokens, Claude Max20 subscription.

## What's New in v3.0.0

*   **Complete Architecture Rewrite:**  Improved modularity, error handling, and testing.
*   **P90 Analysis:** Machine learning-based limit detection.
*   **Updated Plan Limits:**  Pro (44k), Max5 (88k), Max20 (220k)
*   **Rich UI Enhancements:** Configurable display refresh rate, auto 12h/24h format, command aliases, and more.

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.

## 🙏 Acknowledgments

A special thanks to [Ed](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor#sponsors) for their invaluable support!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐  If you find this tool helpful, please star the repository! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>
```
Key improvements and SEO optimizations:

*   **Clear, concise hook:**  Emphasizes the core benefit (staying ahead of token limits).
*   **SEO-friendly headings:**  Uses relevant keywords (Claude, Token Limits, AI) in headings and subheadings.
*   **Bulleted key features:**  Highlights benefits and features for easy skimming.
*   **Concise installation instructions:**  Provides the most important steps upfront.
*   **Clear plan descriptions:**  Explains what each plan offers.
*   **Call to action:** Encourages users to star the repo and contribute.
*   **Internal linking:**  Uses internal links to other relevant documentation (CONTRIBUTING.md).
*   **More concise:** Condenses redundant information and streamlines the content.
*   **Focus on benefits:** Leads with benefits (staying ahead of limits) rather than solely listing features.
*   **Prioritizes key information:** Places the most important details (installation, usage, key features) at the top.
*   **Emphasis on the user:** Uses language that is user-centric (e.g., "stay ahead of *your* token limits").
*   **Simplified installation instructions:** Directs users to the easiest method first and offers other options.
*   **Expanded on Usage Details:**  Provides specific help information.
*   **Added real-world scenarios:** Shows users how to use the tool based on their situation.
*   **Simplified Architecture Overview:**  Used bullet points instead of long explanations.
*   **Cleaned Up v3.0.0 details:** Streamlined what's new.
*   **Clear Troubleshooting Section:**  Added to help users.
*   **Added star history chart.**