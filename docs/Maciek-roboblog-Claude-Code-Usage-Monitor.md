# Claude Code Usage Monitor: Stay in Control of Your Claude AI Token Usage

**Tired of running out of Claude AI tokens mid-session?** 🚀 The Claude Code Usage Monitor provides real-time, intelligent tracking and prediction, helping you optimize your Claude AI usage.  [Check out the original repo!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

This powerful terminal tool gives you a beautiful, real-time view of your Claude AI token consumption, burn rate, and session cost, including machine learning-based predictions to prevent running out of tokens.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   ✅ **Real-time Monitoring:** Track token usage, cost, and burn rate with configurable refresh rates.
*   🔮 **ML-Powered Predictions:** Intelligent session limit detection and forecasting using advanced machine learning.
*   📊 **Advanced Rich UI:** Beautiful, color-coded progress bars, tables, and layouts with WCAG-compliant contrast.
*   🤖 **Smart Auto-Detection:** Automatic plan switching and custom limit discovery based on your usage patterns.
*   ⚠️ **Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   📈 **Cost Analytics:** Model-specific pricing with cache token calculations.
*   🔧 **Pydantic Validation:** Type-safe configuration with automatic validation.
*   ⚙️ **Configurable:** Choose between Pro, Max5, Max20, or a Custom plan that learns from your usage.
*   🛠️ **Easy Installation:** Available via `uv` (recommended), `pip`, `pipx`, and `conda/mamba`.

## Installation

### Modern Installation with `uv` (Recommended)

`uv` offers the easiest and most reliable installation, creating isolated environments automatically.

```bash
# Install with uv
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

**Note:** If `claude-monitor` command is not found after `pip` install, make sure your `~/.local/bin` directory is in your `PATH`.

### Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` within a conda environment.

## Usage

### Basic Usage

Simply run `claude-monitor` (or one of the short aliases).  The monitor will use the **Custom plan** by default, with intelligent limits based on your recent usage.

### Configuration Options

*   **`--plan`**: Choose your plan: `pro`, `max5`, `max20`, or `custom` (default).
*   **`--custom-limit-tokens`**: Set a custom token limit for the `custom` plan.
*   **`--timezone`**: Set your timezone (e.g., `America/New_York`, `UTC`).
*   **`--time-format`**: Choose time format (`12h`, `24h`, or `auto`).
*   **`--theme`**: Set the display theme (`light`, `dark`, `classic`, or `auto`).
*   **`--refresh-rate`**: Set data refresh rate (seconds).
*   **`--refresh-per-second`**: Set display refresh rate in Hz.
*   **`--reset-hour`**: Configure daily reset time.
*   **`--log-level`**: Set the logging level.
*   **`--log-file`**: Specify a log file path.
*   **`--debug`**: Enable debug logging.
*   **`--clear`**: Clear saved configuration.

### Example Usage

```bash
# Start with auto-detected limits (Custom plan)
claude-monitor

# Track a Pro plan
claude-monitor --plan pro

# Reset the usage at 3 AM, and use dark theme
claude-monitor --reset-hour 3 --theme dark
```

## Available Plans

| Plan          | Token Limit     | Best For                      |
| ------------- | --------------- | ----------------------------- |
| **custom**    | P90 auto-detect | Intelligent limit detection (default) |
| **pro**       | ~19,000         | Claude Pro subscription        |
| **max5**      | ~88,000         | Claude Max5 subscription       |
| **max20**     | ~220,000        | Claude Max20 subscription      |

## ✨ v3.0.0 Major Update Highlights

*   **Complete Architecture Rewrite:** Modular, SRP-compliant design for improved maintainability.
*   **ML-Based Predictions:** Uses 90th percentile calculations for intelligent session limit detection.
*   **Advanced Rich UI:** Enhanced, WCAG-compliant terminal UI with configurable refresh rates.
*   **Smart Auto-Detection:** Automatically detects your terminal theme, time format, and plan.
*   **Enhanced Logging & Configuration:** Includes detailed logging options and saves user preferences.

## Troubleshooting

See the [Troubleshooting](TROUBLESHOOTING.md) section for common issues and solutions, including help with "externally-managed-environment" errors and command not found issues.

## 📞 Contact

Get help with questions, suggestions, or collaborations by contacting [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

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

Special thanks to **Ed**, a generous supporter, for their contributions!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>