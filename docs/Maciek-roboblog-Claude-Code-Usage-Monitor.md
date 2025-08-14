# 🚀 Claude Code Usage Monitor: Real-time AI Token Tracking & Analytics

**Effortlessly monitor your Claude AI token usage with advanced analytics, intelligent predictions, and a beautiful terminal interface. ([View on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor))**

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)
<br>
![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## ✨ Key Features

*   **🔮 ML-Powered Predictions:** Leverage machine learning for accurate session limit detection and burn rate forecasting.
*   **🔄 Real-time Monitoring:** Track your token consumption with configurable refresh rates and dynamic display updates.
*   **📊 Rich Terminal UI:** Enjoy a visually appealing interface with color-coded progress bars, sortable tables, and adaptive layouts.
*   **🤖 Smart Auto-Detection:** Benefit from automatic plan switching and personalized limit discovery based on your usage.
*   **📈 Cost & Time Analytics:** Detailed cost breakdowns, model-specific pricing, and time-based session projections.
*   **✅ Easy Installation:** Simple setup with `uv`, `pip`, `pipx`, or `conda/mamba`.

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

The easiest and fastest way to install, offering isolated environments and hassle-free management.

```bash
# Install directly from PyPI with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```
#### First-time uv users
```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor

# Add to PATH if command not found (if needed)
# echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
# source ~/.bashrc  # or restart your terminal

claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   **pipx (Isolated Environments):** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within your conda environment)

## 📖 Usage

### 🎯 Get Help

```bash
claude-monitor --help
```

### ⚙️ Key Command-Line Parameters

| Parameter             | Type    | Default   | Description                                                                          |
| --------------------- | ------- | --------- | ------------------------------------------------------------------------------------ |
| `--plan`              | string  | `custom`  | Plan type: `pro`, `max5`, `max20`, or `custom`                                       |
| `--custom-limit-tokens` | int     | `None`    | Token limit for custom plan (must be > 0)                                             |
| `--view`              | string  | `realtime` | View type: `realtime`, `daily`, or `monthly`                                         |
| `--timezone`          | string  | `auto`    | Timezone (auto-detected). Examples: `UTC`, `America/New_York`, `Europe/London`         |
| `--time-format`       | string  | `auto`    | Time format: `12h`, `24h`, or `auto`                                                 |
| `--theme`             | string  | `auto`    | Display theme: `light`, `dark`, `classic`, or `auto`                                  |
| `--refresh-rate`      | int     | `10`      | Data refresh rate in seconds (1-60)                                                   |
| `--refresh-per-second`| float   | `0.75`    | Display refresh rate in Hz (0.1-20.0)                                                 |
| `--reset-hour`        | int     | `None`    | Daily reset hour (0-23)                                                                |
| `--log-level`         | string  | `INFO`    | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                          |
| `--log-file`          | path    | `None`    | Log file path                                                                        |
| `--debug`             | flag    | `False`   | Enable debug logging                                                                   |
| `--version, -v`       | flag    | `False`   | Show version information                                                               |
| `--clear`             | flag    | `False`   | Clear saved configuration                                                            |

### 📝 Saved Configuration

The monitor automatically saves your preferences for future use.  Your settings are stored in `~/.claude-monitor/last_used.json`. You can clear them with `--clear`.

### 🔌 Basic Usage

```bash
# Default (Custom plan with auto-detection)
claude-monitor

# Alternative commands
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# Exit the monitor: Ctrl+C
```

## 📚 Additional Information

*   **[Development Roadmap](DEVELOPMENT.md)**
*   **[Contributing Guide](CONTRIBUTING.md)**
*   **[Troubleshooting](TROUBLESHOOTING.md)**

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

*   **Sponsors:** (See README for Details)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>