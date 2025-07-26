# ⏱️ Claude Code Usage Monitor: Real-time Token Tracking & Analytics

**Stay in control of your Claude AI token usage with this powerful, real-time terminal monitor!** ([View on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor))

This tool provides advanced analytics, machine learning-based predictions, and a beautiful Rich UI to help you track your token consumption, estimate costs, and get intelligent predictions about your session limits.

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   ✅ **ML-Powered Predictions:**  Get intelligent session limit detection based on your usage patterns.
*   🔄 **Real-time Monitoring:**  Configure refresh rates for up-to-the-second token tracking.
*   📊 **Rich UI:**  Enjoy a beautiful, color-coded terminal display with WCAG-compliant contrast.
*   🤖 **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   💰 **Cost Analytics:** Track model-specific pricing and cache token calculations.
*   ⚙️ **Customizable:**  Tailor the monitor with configurable refresh rates, themes, and more.

## Installation

### ⚡ Modern Installation with uv (Recommended)

The fastest and easiest way to install:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install from PyPI
uv tool install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

For other install methods (pip, pipx, conda) please refer to the full README: [https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

## Usage

```bash
claude-monitor --help # See all the options
```

### Key Command Line Options

*   `--plan`: Choose your plan (pro, max5, max20, custom). Default: `custom`.
*   `--view`: View type: `realtime`, `daily`, or `monthly`.
*   `--timezone`: Set your timezone (e.g., `America/New_York`).
*   `--time-format`: Choose 12h or 24h format.
*   `--theme`: Choose a theme (light, dark, classic, auto).

## Available Plans

| Plan        | Token Limit     |
|-------------|-----------------|
| **custom** | P90 auto-detect |
| **pro**     | ~19,000         |
| **max5**    | ~88,000         |
| **max20**   | ~220,000        |

## Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for details.

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>