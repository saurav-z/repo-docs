# Claude Code Usage Monitor: Track Your Claude AI Token Consumption in Real-Time

Tired of hitting your Claude AI limits unexpectedly? **Stay in control of your AI usage with the Claude Code Usage Monitor!** This powerful, real-time terminal tool provides advanced analytics, intelligent predictions, and a beautiful Rich UI to help you optimize your Claude AI sessions.  [View on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## Key Features

*   **🔮 ML-based Predictions**:  Get intelligent session limit detection and P90 percentile calculations for accurate usage forecasting.
*   **🔄 Real-time Monitoring**:  Monitor your token usage with configurable refresh rates (0.1-20 Hz) and intelligent display updates.
*   **📊 Advanced Rich UI**:  Enjoy a beautiful, color-coded terminal interface with progress bars, tables, and layouts designed for optimal readability.
*   **🤖 Smart Auto-Detection**: Automatic plan switching, and custom limit discovery to adapt to your usage patterns.
*   **💼 Professional Architecture**: Built with a modular design adhering to the Single Responsibility Principle (SRP) for enhanced maintainability.
*   **🎨 Intelligent Theming**: Automatic terminal background detection for optimal display theming.
*   **📈 Cost Analytics**:  Track model-specific pricing and cache token calculations for better cost management.
*   **⚠️ Advanced Warning System**: Multi-level alerts with cost and time predictions to help you stay within your limits.
*   **⚡ Performance Optimized**:  Benefit from advanced caching and efficient data processing.
*   **📋 Default Custom Plan**:  The custom plan analyzes your usage patterns over the last 8 days to provide personalized limits, with token usage, messages, and cost tracked.

## Installation

Choose the installation method that suits your needs.  **uv is the recommended and easiest method.**

### ⚡ Modern Installation with uv (Recommended)

uv is a fast and modern package and dependency manager.

```bash
# Install uv if you don't have it:
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install the monitor:
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with pip

```bash
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH (as described above):
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   **pipx**: `pipx install claude-monitor`
*   **conda/mamba**: `pip install claude-monitor` within your conda environment.

## Usage

### Basic Usage

```bash
# Run with default settings (Custom plan, real-time view)
claude-monitor
```

*   Press `Ctrl+C` to gracefully exit.

### Configuration Options

Customize your monitoring experience with a variety of options, saved automatically:

*   **`--plan`**: Choose your Claude plan (`pro`, `max5`, `max20`, `custom`).  `custom` is the default.
*   **`--custom-limit-tokens`**: Set a custom token limit (for the custom plan).
*   **`--view`**: Select the view type (`realtime`, `daily`, `monthly`).
*   **`--timezone`**: Specify your timezone (e.g., `America/New_York`, `UTC`, `auto`).
*   **`--time-format`**: Set your preferred time format (e.g., `12h`, `24h`, `auto`).
*   **`--theme`**:  Choose your theme (`light`, `dark`, `classic`, `auto`).
*   **`--refresh-rate`**: Set the data refresh rate in seconds (1-60).
*   **`--refresh-per-second`**: Adjust the display refresh rate (0.1-20.0 Hz).
*   **`--reset-hour`**: Set the daily reset hour (0-23).
*   **`--log-level`**: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
*   **`--log-file`**: Specify a log file path.
*   **`--debug`**: Enable debug logging.
*   **`--clear`**: Clear saved configuration.

### Available Plans

| Plan           | Token Limit  | Best For                           |
| -------------- | ------------- | ---------------------------------- |
| **custom**     | P90 auto-detect | Intelligent limit detection (default)|
| **pro**        | ~19,000        | Claude Pro subscription            |
| **max5**       | ~88,000        | Claude Max5 subscription           |
| **max20**      | ~220,000       | Claude Max20 subscription          |

## ✨ Features & How It Works

*   **v3.0.0 Architecture Overview:** Complete architecture rewrite focused on modularity and the Single Responsibility Principle.
*   **🔄 Advanced Real-time Monitoring**: High-precision and configurable monitoring with multiple usage views.
*   **🔮 Machine Learning Predictions**: P90 percentile, burn rate analysis, session forecasting.
*   **🤖 Intelligent Auto-Detection**: Intelligent Plan selection and session limit discovery.
*   **Understanding Claude Sessions**: Explanations for session mechanics.

## Troubleshooting

See the [Troubleshooting section](#troubleshooting) in the README for common installation and runtime issues.

## 📚 Additional Resources

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

*   **Ed** - *Buy Me Coffee Supporter* - "I appreciate sharing your work with the world. It helps keep me on track with my day. Quality readme, and really good stuff all around!"

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>
```
Key improvements and SEO optimizations:

*   **Clear Headline:**  Strong, keyword-rich headline.
*   **Concise Hook:**  One-sentence description of the tool's core benefit.
*   **Keyword-Rich Language:** Used relevant keywords throughout ("Claude AI," "token usage," "real-time monitoring," etc.).
*   **Bulleted Key Features:**  Easy to scan, highlights the tool's main advantages.
*   **Clear Structure:**  Uses headings and subheadings for easy navigation.
*   **Installation Section Prioritized:** Placed installation first, the most critical info. uv install instructions are emphasized.
*   **Clear Usage Section:**  Simple examples and explanations for getting started.
*   **Troubleshooting Section:**  Added a troubleshooting section (as suggested) to handle common issues.
*   **Star History Chart:**  Added Star History chart to show growth
*   **Contributors and Acknowledgments:** Kept the contributors and acknowledgments sections.
*   **Calls to Action:**  Encouraged starring the repository and included links for bug reports and feature requests.
*   **Concise, Focused Content:**  Eliminated unnecessary repetition and focused on the core value proposition.
*   **Links to Original Repo:** Maintained link to the original repo.