# 🚀 Claude Code Usage Monitor: Real-Time Token Tracking with Advanced Analytics

**Tired of unexpected Claude AI costs?** Claude Code Usage Monitor provides a beautiful, real-time terminal interface for tracking your Claude AI token usage, with machine learning-powered predictions and comprehensive cost analysis. [View on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## Key Features:

*   **🚀 Real-time Monitoring:** Track token usage, burn rate, and cost with configurable refresh rates.
*   **📊 Advanced Analytics:**  View daily and monthly usage, identify trends, and optimize your Claude AI usage.
*   **🔮 ML-Powered Predictions:** Get intelligent session limit detection and cost projections based on your usage patterns.
*   **🎨 Rich Terminal UI:** Enjoy a beautiful, color-coded interface with progress bars, tables, and WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery to optimize your costs.
*   **💼 Professional Architecture:** Modular design, SRP compliance, Pydantic validation and comprehensive testing.

## Installation:

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is the fastest and easiest way to install and manage the monitor.

```bash
# Install uv (if you don't have it):
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Claude Monitor
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# Run from anywhere (may need to add ~/.local/bin to your PATH)
claude-monitor  # or cmonitor, ccmonitor for short
```
### 🛠️ Other Package Managers

You can also use `pipx` and conda/mamba. See the original README for details.

## Usage:

```bash
# Run the monitor with default settings (custom plan)
claude-monitor

# Get help on command-line options
claude-monitor --help

# Examples of common configuration
claude-monitor --plan pro                # Use Pro plan (19k token limit)
claude-monitor --plan max5               # Use Max5 plan (88k token limit)
claude-monitor --plan custom --custom-limit-tokens 100000  # Set custom token limit
claude-monitor --view daily               # View daily token usage
claude-monitor --theme dark               # Use dark theme
claude-monitor --timezone America/New_York  # Set timezone
```

### Available Plans:

*   **custom**: (default)  ML-based, P90 auto-detect based.
*   **pro**: 19,000 token limit (approx)
*   **max5**: 88,000 token limit (approx)
*   **max20**: 220,000 token limit (approx)

## Key Improvements in v3.0.0

*   **Complete Architecture Rewrite**: Modular design with enhanced testing and error handling.
*   **P90 Analysis**:  Machine learning-based limit detection using 90th percentile calculations for the custom plan.
*   **Updated Plan Limits**: Pro (19k), Max5 (88k), Max20 (220k) tokens.
*   **Rich UI**: WCAG-compliant themes and auto-detection of terminal background.

## Development:

Detailed information on development installation, testing and contribution is available in the original README.

## Contact:

maciek@roboblog.eu

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>