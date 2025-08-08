# 🚀 Claude Code Usage Monitor: Track, Predict, and Optimize Your Claude AI Token Usage

Tired of unexpectedly hitting your Claude AI token limits?  **Claude Code Usage Monitor** is a powerful terminal tool designed to provide real-time insights, intelligent predictions, and cost analysis to help you maximize your Claude AI usage. [Visit the GitHub Repository](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## 🔑 Key Features

*   **Real-time Monitoring:**  Track token consumption, burn rate, and cost in a dynamic, visually appealing terminal interface.
*   **ML-Powered Predictions:**  Intelligent session limit detection with P90 percentile calculations and session forecasting.
*   **Advanced Analytics:**  Gain insights into your usage patterns with customizable views (realtime, daily, monthly) and detailed cost analysis.
*   **Smart Auto-Detection:** Automatically adjusts to your usage, with auto plan switching and background detection.
*   **Flexible Plan Support:** Supports all Claude AI plans (Pro, Max5, Max20, Custom).
*   **Rich Terminal UI:**  Beautiful color-coded progress bars, tables, and layouts with WCAG-compliant contrast for optimal readability.
*   **Easy Installation:**  Multiple installation methods including uv, pip, pipx, and conda.

---

## 🚀 Installation

Choose your preferred installation method:

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is a fast and reliable package manager that simplifies installation and environment management.

```bash
# Install uv (if you haven't already)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

# Install Claude Monitor
uv tool install claude-monitor

# Run the monitor from anywhere
claude-monitor
```

### 📦 Installation with `pip`

```bash
# Install claude-monitor
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run the monitor from anywhere
claude-monitor
```

### 🛠️ Other Package Managers

*   **`pipx` (Isolated Environments):** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` within your conda environment

---

## 📖 Usage

### ⚙️ Command-Line Options

```bash
# Show help information
claude-monitor --help
```

#### Key Parameters:

*   `--plan`: Specify your Claude AI plan (pro, max5, max20, custom).  Defaults to `custom`.
*   `--custom-limit-tokens`: Set a custom token limit for the `custom` plan.
*   `--view`: Select your view (realtime, daily, monthly).  Defaults to `realtime`.
*   `--timezone`: Set your timezone (e.g., UTC, America/New_York, Europe/London). Auto-detected by default.
*   `--time-format`:  Choose time format (12h, 24h). Auto-detected by default.
*   `--theme`: Set the terminal theme (light, dark, classic, auto). Auto-detected by default.
*   `--refresh-rate`: Data refresh rate in seconds (1-60).
*   `--refresh-per-second`: Display refresh rate in Hz (0.1-20.0).
*   `--reset-hour`: Set daily reset hour (0-23).
*   `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO.
*   `--log-file`: Specify a log file path.
*   `--debug`: Enable debug logging.
*   `--clear`: Clear saved configuration.

### 💡 Examples

```bash
# Monitor with default settings (Custom plan)
claude-monitor

# Monitor your Pro plan
claude-monitor --plan pro

# Monitor your Max5 plan
claude-monitor --plan max5

# Monitor your Max20 plan
claude-monitor --plan max20

# Daily view to review usage
claude-monitor --view daily

# Set a custom reset hour
claude-monitor --reset-hour 9
```

### 💾 Saved Configuration

The monitor intelligently saves your preferences to persist settings across sessions (e.g. view type, theme, time zone).

---

## 💻  What's New in v3.0.0

*   **Complete Architecture Rewrite**: Designed with modularity and the Single Responsibility Principle (SRP) for better maintainability.
*   **ML-Based Predictions**:  Leverages machine learning for more accurate limit detection and session forecasting.
*   **Enhanced Functionality**: Updated plan limits, and richer UI with WCAG-compliant themes.
*   **New CLI Options**: Improved control with refresh rate and custom token settings.

---

## 🛠️ Development & Contributing

For developers:

1.  **Clone the repository:** `git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git`
2.  **Install Development Dependencies:** Follow the [Development Installation](#-development-installation) instructions.
3.  **Run Tests:**  Run `python -m pytest` in the `src/` directory.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

---

## 🤝 Acknowledgements

Special thanks to Ed for their generous support!
---

## 📝 License

Released under the [MIT License](LICENSE).

## Contact

For questions or suggestions, reach out to maciek@roboblog.eu.

---

<div align="center">

**⭐  Like this project? Star it on GitHub! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>