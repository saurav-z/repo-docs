# 🚀 Claude Code Usage Monitor: Real-time AI Token Tracking & Analytics

**Tired of guessing your Claude AI token usage?** This powerful terminal tool gives you real-time monitoring, advanced analytics, and intelligent predictions, ensuring you stay within your limits. Track token consumption, analyze costs, and get smart warnings about session limits. [Visit the original repo for more information.](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **🔮 ML-based Predictions:** Get intelligent session limit detection using P90 percentile calculations.
*   **🔄 Real-time Monitoring:** Configure refresh rates for dynamic and efficient tracking.
*   **📊 Advanced Rich UI:**  Beautiful, color-coded progress bars, tables, and layouts.
*   **🤖 Smart Auto-Detection:** Automatic plan switching with custom limit discovery.
*   **📋 Enhanced Plan Support:**  Updated limits for Pro, Max5, Max20, and a Custom plan.
*   **⚠️ Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   **💼 Professional Architecture:** Modular design adhering to the Single Responsibility Principle (SRP).
*   **🎨 Intelligent Theming:** Scientific color schemes with automatic terminal background detection.
*   **⏰ Advanced Scheduling:** Auto-detected system timezone and time format preferences.
*   **📈 Cost Analytics:** Model-specific pricing with cache token calculations.
*   **🔧 Pydantic Validation:** Type-safe configuration with automatic validation.
*   **📝 Comprehensive Logging:** Optional file logging with configurable levels.
*   **🧪 Extensive Testing:** 100+ test cases with full coverage.
*   **🎯 Error Reporting:** Optional Sentry integration for production monitoring.
*   **⚡ Performance Optimized:** Advanced caching and efficient data processing.

## Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` simplifies installation and creates isolated environments.  It avoids common Python version issues.

```bash
# Install with uv:
uv tool install claude-monitor

# Run:
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor

# If "claude-monitor" command not found, add your local bin to PATH
# Add this line to ~/.bashrc or ~/.zshrc and reload the shell or restart terminal
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

#### pipx

```bash
pipx install claude-monitor
claude-monitor # or other aliases
```

#### conda/mamba

```bash
pip install claude-monitor
claude-monitor # or other aliases
```

## Usage

###  Basic Usage
```bash
# Default: Custom plan (auto-detect)
claude-monitor
```

### Command Aliases

The tool can be invoked using any of these commands:
- claude-monitor (primary)
- claude-code-monitor (full name)
- cmonitor (short)
- ccmonitor (short alternative)
- ccm (shortest)

### Configuration Options

| Parameter           | Type    | Default | Description                                                                                                   |
|---------------------|---------|---------|---------------------------------------------------------------------------------------------------------------|
| `--plan`            | string  | custom  | Plan type: `pro`, `max5`, `max20`, or `custom`                                                               |
| `--custom-limit-tokens` | int     | None    | Token limit for custom plan (must be > 0)                                                               |
| `--view`            | string  | realtime| View type: `realtime`, `daily`, or `monthly`                                                                |
| `--timezone`        | string  | auto    | Timezone (auto-detected).  Examples: `UTC`, `America/New_York`, `Europe/London`                                  |
| `--time-format`     | string  | auto    | Time format: `12h`, `24h`, or `auto`                                                                          |
| `--theme`           | string  | auto    | Display theme: `light`, `dark`, `classic`, or `auto`                                                        |
| `--refresh-rate`    | int     | 10      | Data refresh rate in seconds (1-60)                                                                           |
| `--refresh-per-second` | float  | 0.75    | Display refresh rate in Hz (0.1-20.0)                                                                        |
| `--reset-hour`      | int     | None    | Daily reset hour (0-23)                                                                                    |
| `--log-level`       | string  | INFO    | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                                               |
| `--log-file`        | path    | None    | Log file path                                                                                               |
| `--debug`           | flag    | False   | Enable debug logging                                                                                          |
| `--version`, `-v`   | flag    | False   | Show version information                                                                                      |
| `--clear`           | flag    | False   | Clear saved configuration                                                                                     |


### Plan Options

| Plan       | Token Limit | Cost Limit        | Description                                      |
|------------|-------------|-------------------|--------------------------------------------------|
| pro        | 19,000      | $18.00            | Claude Pro subscription                        |
| max5       | 88,000      | $35.00            | Claude Max5 subscription                       |
| max20      | 220,000     | $140.00           | Claude Max20 subscription                      |
| custom     | P90-based   | (default) $50.00  | Auto-detection with ML analysis (recommended) |

## Custom Plan

The **Custom plan** is the default and auto-adapts to your usage, analyzing your sessions over the last 8 days to calculate personalized limits.

## Troubleshooting

*   **"externally-managed-environment" Error:** (Linux) Use `uv`, `pipx` or a virtual environment.
*   **Command Not Found:** Ensure your local bin is on your PATH.
*   **No active session found:** Start a conversation with Claude before running the monitor. Specify your Claude config path.

## Additional Resources

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)
*   [License](LICENSE)
*   [Contact](mailto:maciek@roboblog.eu)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>