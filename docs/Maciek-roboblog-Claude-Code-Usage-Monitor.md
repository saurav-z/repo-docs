# 🚀 Claude Code Usage Monitor: Track and Optimize Your Claude AI Token Usage

**Tired of guessing your Claude AI token usage?** This powerful, real-time terminal tool provides advanced analytics, machine-learning based predictions, and a rich user interface to help you monitor and optimize your token consumption.  [Check it out on GitHub!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 🔑 Key Features

*   **🔮 ML-Powered Predictions:**  Intelligent session limit detection and P90 percentile calculations.
*   **🔄 Real-Time Monitoring:** Configurable refresh rates (0.1-20 Hz) with dynamic display updates.
*   **📊 Rich Terminal UI:**  Beautiful, color-coded progress bars, tables, and WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **📋 Enhanced Plan Support:**  Updated limits for Pro, Max5, Max20, and Custom plans.
*   **⚠️ Advanced Warning System:**  Multi-level alerts with cost and time predictions.
*   **🎨 Intelligent Theming:**  Scientific color schemes with automatic terminal background detection.
*   **📈 Cost Analytics:** Model-specific pricing with cache token calculations.
*   **📝 Comprehensive Logging:** Optional file logging with configurable levels.
*   **🧪 Extensive Testing:** 100+ test cases with full coverage.
*   **🎯 Error Reporting:**  Optional Sentry integration for production monitoring.
*   **⚡ Performance Optimized:** Advanced caching and efficient data processing.

## 🚀 Installation

### 1. ⚡ Modern Installation with `uv` (Recommended)

**Why `uv`?**

*   ✅ Automatic isolated environments (no system conflicts)
*   ✅ No Python version issues
*   ✅ Easy updates and uninstallation
*   ✅ Works on all platforms

#### Install & Run with `uv`

```bash
# Install with uv:
uv tool install claude-monitor

# Run from anywhere:
claude-monitor  # or cmonitor, ccmonitor, ccm
```

#### First-time `uv` Users

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
### 2. 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 3. 🛠️ Other Package Managers

#### `pipx` (Isolated Environments)

```bash
# Install with pipx
pipx install claude-monitor

# Run from anywhere
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```

#### `conda`/`mamba`

```bash
# Install with pip in conda environment
pip install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

## 📖 Usage

### 💡 Get Help

```bash
claude-monitor --help
```

### ⚙️ Command-Line Parameters

| Parameter             | Type     | Default     | Description                                                                                             |
| --------------------- | -------- | ----------- | ------------------------------------------------------------------------------------------------------- |
| `--plan`              | string   | `custom`    | Plan type: `pro`, `max5`, `max20`, or `custom`                                                             |
| `--custom-limit-tokens` | int      | `None`      | Token limit for custom plan (must be > 0)                                                              |
| `--view`              | string   | `realtime`  | View type: `realtime`, `daily`, or `monthly`                                                             |
| `--timezone`          | string   | `auto`      | Timezone (auto-detected). Examples: `UTC`, `America/New_York`, `Europe/London`                            |
| `--time-format`       | string   | `auto`      | Time format: `12h`, `24h`, or `auto`                                                                     |
| `--theme`             | string   | `auto`      | Display theme: `light`, `dark`, `classic`, or `auto`                                                     |
| `--refresh-rate`      | int      | `10`        | Data refresh rate in seconds (1-60)                                                                      |
| `--refresh-per-second`| float    | `0.75`      | Display refresh rate in Hz (0.1-20.0)                                                                    |
| `--reset-hour`        | int      | `None`      | Daily reset hour (0-23)                                                                                  |
| `--log-level`         | string   | `INFO`      | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                                           |
| `--log-file`          | path     | `None`      | Log file path                                                                                           |
| `--debug`             | flag     | `False`     | Enable debug logging                                                                                     |
| `--version, -v`       | flag     | `False`     | Show version information                                                                                 |
| `--clear`             | flag     | `False`     | Clear saved configuration                                                                                |

### 💰 Available Plans

| Plan         | Token Limit   | Best For                       |
| ------------ | ------------- | ------------------------------ |
| **custom**   | P90 auto-detect | Intelligent limit detection (default) |
| **pro**      | ~19,000       | Claude Pro subscription        |
| **max5**     | ~88,000       | Claude Max5 subscription       |
| **max20**    | ~220,000      | Claude Max20 subscription      |

### 🕹️ Usage Examples

```bash
# Run with default settings (Custom plan)
claude-monitor

# Start with Pro Plan and dark theme.  Settings will be saved for future runs.
claude-monitor --plan pro --theme dark

# Override saved settings for a single run
claude-monitor --plan pro --theme light

# See daily usage
claude-monitor --view daily

# Clear your saved settings
claude-monitor --clear
```

## 📖 Additional Information

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 📝 License

[MIT License](LICENSE)