# 🚀 Claude Code Usage Monitor: Real-time AI Token Tracking & Analytics

**Tired of guessing how much your Claude AI usage costs?** Get real-time insights with the Claude Code Usage Monitor, a powerful terminal tool that tracks your token consumption, predicts session limits, and provides advanced analytics. 

[View the original repository](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## 🌟 Key Features

*   **📊 Real-time Monitoring:** Track token usage, burn rate, and cost in real-time with configurable refresh rates.
*   **🔮 ML-Based Predictions:** Intelligent session limit detection using P90 percentile calculations and session forecasting.
*   **📈 Advanced Analytics:** Gain insights with cost analysis and model-specific pricing.
*   **🤖 Smart Auto-Detection:** Automatically detects terminal theme, timezone, and plan usage.
*   **🎨 Rich Terminal UI:** Enjoy a beautiful, color-coded display with WCAG-compliant contrast.
*   **✅ Easy Installation:** Install quickly with `uv tool install claude-monitor` (recommended).

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is the fastest and easiest way to install and use the monitor.

```bash
# Install uv (if you don't have it already)
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install and Run
uv tool install claude-monitor
claude-monitor # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

*   **⚠️ PATH Setup**: If you see a warning about the script not being on your PATH, follow the instructions in the original README to add the installation directory to your environment's PATH variable.
*   **⚠️ For Linux Distributions**: If using `pip` on modern Linux distributions, consider using `uv` or a virtual environment to avoid "externally-managed-environment" errors. See the original README for detailed solutions.

### 🛠️ Other Package Managers

The original README provides instructions for installation with `pipx` and `conda/mamba`.

## 📖 Usage

### 🚀 Basic Usage

```bash
# Default (Custom plan with auto-detection)
claude-monitor

# Alternative commands
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# Exit the monitor
# Press Ctrl+C to gracefully exit
```

### 📚 Configuration Options

*   **`--plan`**: Select your Claude plan (`pro`, `max5`, `max20`, or `custom` - default).
*   **`--custom-limit-tokens`**: Set a custom token limit for the `custom` plan.
*   **`--view`**: Choose a view (`realtime`, `daily`, or `monthly`).
*   **`--timezone`**: Set your timezone (auto-detected by default).
*   **`--time-format`**: Set time format (`12h`, `24h`, or `auto`).
*   **`--theme`**: Choose a display theme (`light`, `dark`, `classic`, or `auto`).
*   **`--refresh-rate`**: Set the data refresh rate (in seconds).
*   **`--refresh-per-second`**: Set the display refresh rate (in Hz).
*   **`--reset-hour`**: Set the daily reset hour.
*   **`--log-level`**: Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
*   **`--log-file`**: Specify a log file path.
*   **`--debug`**: Enable debug logging.
*   **`--clear`**: Clear saved configuration.

### 💡 Command Aliases & Saved Preferences

*   The tool remembers your preferences, so you don't have to specify them every time.
*   Use command aliases for quicker access (e.g., `cmonitor`).
*   Use the `--clear` flag to reset saved preferences.

### ⚙️ Plan Options

| Plan        | Token Limit | Description                                         |
|-------------|-------------|-----------------------------------------------------|
| **custom**  | P90-based   | Intelligent limit detection (default)               |
| **pro**     | ~19,000     | Claude Pro subscription                             |
| **max5**    | ~88,000     | Claude Max5 subscription                            |
| **max20**   | ~220,000    | Claude Max20 subscription                           |

## ✨ Features & How It Works

This monitor provides a modular architecture with Single Responsibility Principle (SRP).

*   **Real-time Monitoring:** See your token usage, burn rate, and cost at a glance.
*   **ML-Based Predictions:** The "custom" plan uses machine learning to analyze your historical usage and predict limits.
*   **Auto-Detection:** Automatically detects terminal theme, timezone, time format and plan recognition for optimal use.
*   **Usage Views:** "Realtime", "Daily", and "Monthly" views offer different perspectives.

## 🛠️ Development Installation

For those looking to contribute, the original README provides detailed development installation instructions, including how to set up a virtual environment and run tests.

## 📞 Contact

Reach out with any questions, suggestions, or collaboration ideas!

*   **📧 Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

Want to contribute? Check out the [Contributing Guide](CONTRIBUTING.md) in the original repo!

## 🙏 Acknowledgments

A special thanks to the supporters who are helping keep this project going!

**Ed** - *Buy Me Coffee Supporter*

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>