# 🚀 Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

**Tired of guessing your Claude AI token usage?** Stay ahead of your limits with the Claude Code Usage Monitor, a powerful, real-time terminal tool that provides advanced analytics, intelligent session predictions, and a beautiful Rich UI. [Check it out on GitHub!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   ✅ **Real-time Monitoring:** Track token consumption, burn rate, and costs with configurable refresh rates.
*   📊 **Advanced Analytics:** View detailed usage, including daily and monthly trends.
*   🤖 **AI-Powered Predictions:** Get intelligent estimates of session limits and receive warnings.
*   🎨 **Rich Terminal UI:** Enjoy a beautiful, color-coded interface that adapts to your terminal.
*   ⚙️ **Customizable:** Set preferences, manage your plan, and configure logging for detailed analysis.
*   🚀 **Modern Installation:** Simplifies installation, particularly with the recommended `uv` package manager.

### Key Features Deep Dive (v3.0.0)

*   **🔮 ML-based predictions** - P90 percentile calculations and intelligent session limit detection
*   **🔄 Real-time monitoring** - Configurable refresh rates (0.1-20 Hz) with intelligent display updates
*   **📊 Advanced Rich UI** - Beautiful color-coded progress bars, tables, and layouts with WCAG-compliant contrast
*   **🤖 Smart auto-detection** - Automatic plan switching with custom limit discovery
*   **📋 Enhanced plan support** - Updated limits: Pro (44k), Max5 (88k), Max20 (220k), Custom (P90-based)
*   **⚠️ Advanced warning system** - Multi-level alerts with cost and time predictions
*   **💼 Professional Architecture** - Modular design with Single Responsibility Principle (SRP) compliance
*   **🎨 Intelligent theming** - Scientific color schemes with automatic terminal background detection
*   **⏰ Advanced scheduling** - Auto-detected system timezone and time format preferences
*   **📈 Cost analytics** - Model-specific pricing with cache token calculations
*   **🔧 Pydantic validation** - Type-safe configuration with automatic validation
*   **📝 Comprehensive logging** - Optional file logging with configurable levels
*   **🧪 Extensive testing** - 100+ test cases with full coverage
*   **🎯 Error reporting** - Optional Sentry integration for production monitoring
*   **⚡ Performance optimized** - Advanced caching and efficient data processing

## Installation

### ⚡ Recommended: Installation with `uv`

`uv` is the fastest, easiest, and most reliable way to install. It handles isolated environments and avoids common Python installation issues.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh # Linux/macOS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

# Install claude-monitor
uv tool install claude-monitor

# Run
claude-monitor # or cmonitor, ccmonitor
```

### 📦 Pip Installation

```bash
pip install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor
```

>   **Note:**  If `claude-monitor` command isn't found after `pip install`, add `~/.local/bin` to your `PATH` (see original README).

### 🛠️ Other Package Managers & Methods

See the original README for `pipx`, `conda/mamba` and development installation instructions.

## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Key Parameters

| Parameter               | Type      | Default       | Description                                  |
| ----------------------- | --------- | ------------- | -------------------------------------------- |
| `--plan`                | string    | `custom`      | Plan type (`pro`, `max5`, `max20`, `custom`) |
| `--custom-limit-tokens` | int       | `None`        | Token limit for `custom` plan                |
| `--view`                | string    | `realtime`    | View type (`realtime`, `daily`, `monthly`)    |
| `--timezone`            | string    | `auto`        | Timezone (e.g., `America/New_York`)          |
| `--time-format`         | string    | `auto`        | Time format (`12h`, `24h`)                     |
| `--theme`               | string    | `auto`        | Display theme (`light`, `dark`, `classic`)    |
| `--refresh-rate`        | int       | `10`          | Data refresh rate (seconds)                   |
| `--refresh-per-second`  | float     | `0.75`        | Display refresh rate (Hz)                      |
| `--reset-hour`          | int       | `None`        | Daily reset hour                             |
| `--log-level`           | string    | `INFO`        | Logging level                                |
| `--log-file`            | path      | `None`        | Log file path                                |
| `--debug`               | flag      | `False`       | Enable debug logging                         |
| `--version, -v`         | flag      | `False`       | Show version information                     |
| `--clear`               | flag      | `False`       | Clear saved configuration                    |

### Basic Usage Examples

```bash
# Run with default settings (Custom plan)
claude-monitor

# Specify your plan
claude-monitor --plan pro

# Get help
claude-monitor --help

# Override saved settings
claude-monitor --plan pro --theme light

# Clear configuration
claude-monitor --clear
```

## Available Plans

| Plan         | Token Limit | Best For                       |
| ------------ | ----------- | ------------------------------ |
| `custom`     | P90-based   | Intelligent limit detection    |
| `pro`        | ~19,000     | Claude Pro subscription        |
| `max5`       | ~88,000     | Claude Max5 subscription       |
| `max20`      | ~220,000    | Claude Max20 subscription      |

## 🚀 What's New in v3.0.0

*   **Complete Architecture Rewrite:** Improved modularity, testing and error handling.
*   **Enhanced Functionality**: ML-powered limit detection, updated plan limits.
*   **New CLI Options**: Customizable display and logging capabilities.
*   **Breaking Changes:** Renamed package, default plan changed to `custom`, Python 3.9+ requirement.

## 📚 Documentation & Resources

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributors

See original README.

## 🙏 Acknowledgments

A special thanks to our supporters!

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>