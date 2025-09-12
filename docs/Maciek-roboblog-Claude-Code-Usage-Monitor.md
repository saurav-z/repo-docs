# 🚀 Claude Code Usage Monitor: Stay on Top of Your Claude AI Token Usage

**Tired of unexpected charges and wasted tokens?** 🚀 This powerful, real-time terminal monitoring tool lets you track your Claude AI token consumption, analyze your burn rate, predict session limits, and optimize your costs.  [View the original repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 🔑 Key Features

*   ✅ **Real-time Monitoring:** Configurable refresh rates (0.1-20 Hz) with intelligent display updates.
*   📊 **Advanced Rich UI:** Beautiful, color-coded progress bars, sortable tables, and WCAG-compliant contrast.
*   🔮 **ML-Powered Predictions:** P90 percentile calculations for smart session limit detection and cost analysis.
*   🤖 **Intelligent Auto-Detection:** Automatic plan switching and custom limit discovery.
*   📈 **Detailed Cost Analytics:** Model-specific pricing with cache token calculations.
*   🔔 **Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   🛠️ **Easy Installation:** Supports `uv` (recommended), `pip`, `pipx`, and `conda/mamba`.

---

## 📦 Installation

### 🚀  Recommended: Modern Installation with `uv`

`uv` is the fastest, easiest, and most reliable way to install and manage dependencies.  It automatically creates isolated environments, eliminating version conflicts.

1.  **Install `uv`:**

    ```bash
    # On Linux/macOS:
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows:
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Install `claude-monitor`:**

    ```bash
    # Install from PyPI with uv (easiest)
    uv tool install claude-monitor

    # Run from anywhere
    claude-monitor  # or cmonitor, ccmonitor for short
    ```

### 📦 Installation with `pip`

1.  **Install `claude-monitor`:**

    ```bash
    pip install claude-monitor
    ```

2.  **Add to PATH (if needed):** If `claude-monitor` is not found after installation, add the following to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`):

    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    source ~/.bashrc  # or restart your terminal
    ```

3.  **Run `claude-monitor`:**

    ```bash
    claude-monitor  # or cmonitor, ccmonitor for short
    ```

### 📦 Other Installation Options

*   **`pipx` (Isolated Environments):** `pipx install claude-monitor`
*   **`conda/mamba`:** `pip install claude-monitor` within your conda environment.

---

## 📖 Usage

### ⚙️ Configuration Options

Customize your monitoring experience with these command-line parameters:

| Parameter              | Type    | Default    | Description                                                                  |
|------------------------|---------|------------|------------------------------------------------------------------------------|
| `--plan`              | string  | `custom`   | Plan type: `pro`, `max5`, `max20`, or `custom` (default)                     |
| `--custom-limit-tokens` | int     | `None`     | Token limit for custom plan (must be > 0)                                    |
| `--view`              | string  | `realtime` | View type: `realtime`, `daily`, or `monthly`                                  |
| `--timezone`          | string  | `auto`     | Timezone (e.g., `UTC`, `America/New_York`, `Europe/London`)                    |
| `--time-format`       | string  | `auto`     | Time format: `12h`, `24h`, or `auto`                                          |
| `--theme`             | string  | `auto`     | Display theme: `light`, `dark`, `classic`, or `auto`                          |
| `--refresh-rate`       | int     | `10`       | Data refresh rate in seconds (1-60)                                          |
| `--refresh-per-second` | float   | `0.75`     | Display refresh rate in Hz (0.1-20.0)                                         |
| `--reset-hour`        | int     | `None`     | Daily reset hour (0-23)                                                        |
| `--log-level`         | string  | `INFO`     | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                   |
| `--log-file`          | path    | `None`     | Log file path                                                                 |
| `--debug`             | flag    | `False`    | Enable debug logging                                                           |
| `--version`, `-v`     | flag    | `False`    | Show version information                                                       |
| `--clear`             | flag    | `False`    | Clear saved configuration (stored in `~/.claude-monitor/last_used.json`)    |

### 🚀 Quick Start Examples

*   **Run with default custom plan:**  `claude-monitor`
*   **Monitor with Pro plan:** `claude-monitor --plan pro`
*   **Set a custom reset time:** `claude-monitor --reset-hour 9 --timezone America/New_York`
*   **View daily token usage:** `claude-monitor --view daily`

---

## 💡 Plan Selection

| Plan          | Token Limit     | Best For                      |
|---------------|-----------------|-------------------------------|
| **custom**  | P90 auto-detect | Intelligent limit detection (default)  |
| **pro**       | ~19,000         | Claude Pro subscription        |
| **max5**      | ~88,000         | Claude Max5 subscription       |
| **max20**     | ~220,000        | Claude Max20 subscription      |

The **Custom plan** is the default and uses machine learning to analyze your usage history and set personalized limits.

---

## ✨ What's New in v3.0.0

*   **Complete Architecture Rewrite:** Modular design, Pydantic validation, comprehensive testing.
*   **Enhanced Functionality:**  P90 analysis, updated plan limits, cost analytics.
*   **New CLI Options:** `--refresh-per-second`, `--time-format`, `--custom-limit-tokens`, `--log-file`, `--log-level`, `--clear`, command aliases.
*   **Breaking Changes:** Package name changed from claude-usage-monitor, default plan is now `custom`, minimum Python version is 3.9+.

---

## 🙋‍♀️ Contact

Have questions or want to collaborate? Contact:  [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

---

## 📚 Additional Resources

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

---

## 📝 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgments

Special thanks to Ed for supporting this project!  ⭐ Thanks to all contributors!

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>