# 🚀 Claude Code Usage Monitor: Real-time Tracking & AI-Powered Predictions

**Keep your Claude AI usage under control with this powerful terminal monitor.** [View on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   ✅ **Real-time Monitoring:** Track token usage, burn rate, and cost in real-time with configurable refresh rates.
*   🔮 **AI-Powered Predictions:** Get intelligent session limit detection and cost projections based on machine learning.
*   📊 **Rich Terminal UI:** Enjoy a beautiful, color-coded interface with WCAG-compliant contrast for easy readability.
*   🤖 **Smart Auto-Detection:** Automatic plan switching and custom limit discovery tailored to your usage.
*   📈 **Advanced Analytics:** Daily and monthly usage views, model-specific pricing, and cache token calculations.
*   🚀 **Easy Installation:** Simple installation using `uv` and pip, with detailed setup instructions.
*   ⚙️ **Configurable Options:** Customize your experience with timezone, theme, refresh rate, and logging settings.

## Installation

### ⚡ Modern Installation with `uv` (Recommended)

The fastest and easiest way to install, `uv` creates isolated environments and handles dependencies.

1.  **Install `uv`:**  Follow the instructions at [https://astral.sh/uv](https://astral.sh/uv)

2.  **Install Claude Monitor:**

    ```bash
    uv tool install claude-monitor
    ```
    or
    ```bash
    git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
    cd Claude-Code-Usage-Monitor
    uv tool install .
    ```

3.  **Run the Monitor:**

    ```bash
    claude-monitor  # or cmonitor, ccmonitor for short
    ```

### 📦 Installation with `pip`

1.  **Install:**

    ```bash
    pip install claude-monitor
    ```

2.  **Run the Monitor:**

    ```bash
    claude-monitor  # or cmonitor, ccmonitor for short
    ```

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within a conda environment)

## 📖 Usage

### Basic Usage

1.  **Run:** `claude-monitor` (or a shorter alias) to start monitoring with the default settings (Custom plan).  Press `Ctrl+C` to exit.

### Key Configuration Options

*   `--plan`:  Choose your plan (pro, max5, max20, custom – the default).
*   `--custom-limit-tokens`: Set a token limit for the custom plan.
*   `--view`: Select real-time, daily, or monthly usage views.
*   `--timezone`: Set your timezone (e.g., "America/New_York", "UTC").
*   `--time-format`: Choose 12h or 24h time format.
*   `--theme`:  Set the terminal theme (light, dark, classic, auto).
*   `--refresh-rate` and `--refresh-per-second`: Control the data/display refresh rates.
*   `--reset-hour`: Set a custom daily reset time.
*   `--log-level` and `--log-file`: Configure logging options.

### Example Commands

*   `claude-monitor --plan pro --theme dark`: Start with the Pro plan and dark theme.
*   `claude-monitor --view daily --timezone "Europe/London"`: View daily usage in London time.
*   `claude-monitor --clear`: Clear saved configuration.

## 🚀 What's New in v3.0.0

*   **Complete Architecture Rewrite:** Improved modularity and maintainability.
*   **Machine Learning Predictions:**  P90 percentile calculations for more accurate predictions.
*   **Updated Plan Limits:** Supports Claude Pro (19k), Max5 (88k), and Max20 (220k) token limits.
*   **New CLI Options:** More control over refresh rates, time format, logging, and saved settings.
*   **Breaking Changes**: Improved plan limits, default plan changed to custom and more.

## 💡 How It Works: Key Components

**Architecture Overview:**
The new version features a modular architecture following Single Responsibility Principle (SRP):

*   **User Interface Layer:**  CLI, settings/configuration, error handling, and a Rich terminal UI.
*   **Monitoring Orchestrator:** Central control hub, data management, session monitoring, UI control, and Analytics
*   **Foundation Layer:**  Core models, analysis engine, terminal themes, and Claude API data

**Data Flow:**
Claude config files → data layer → analysis engine → UI components → Terminal Display

### Current Features

*   **Real-time Monitoring:** Configurable update intervals and high-precision display refresh.
*   **Rich UI Components:** Progress bars, sortable data tables, and a responsive layout manager.
*   **Multiple Usage Views:** Realtime, daily, and monthly views.
*   **Machine Learning Predictions:** P90 calculator, burn rate analysis, cost projections, and session forecasting.
*   **Intelligent Auto-Detection:** Terminal theme, system timezone/time format, plan recognition, and limit discovery.

### Understanding Claude Sessions and Token Limits

*   **How Claude Sessions Work:** 5-hour rolling session windows.
*   **Token Limits by Plan:** Pro (~19,000 tokens), Max5 (~88,000 tokens), Max20 (~220,000 tokens), Custom (P90-based).
*   **Advanced Limit Detection:** P90 analysis, 95% confidence, and cache token calculations.

## 🚀 Usage Examples

*   **Morning Developer:**  `claude-monitor --reset-hour 9` (with custom timezone)
*   **Heavy User:** `claude-monitor --plan custom`
*   **Quick Check:** `claude-monitor`
*   **Daily Analysis:** `claude-monitor --view daily`

## 🔧 Development Installation

For contributors and developers:

1.  **Clone:** `git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git`
2.  **Install (Development Mode):** `pip install -e .`
3.  **Run:** `python -m claude_monitor` (from the project root)
4.  **Run Tests:**  `python -m pytest`

## Troubleshooting

*   See the [Troubleshooting section](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/blob/main/README.md#troubleshooting) for installation and runtime issue solutions, including the "externally-managed-environment" error and common issues.

## 📞 Contact

maciek@roboblog.eu - Reach out with questions, suggestions, or collaboration requests!

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

MIT License

## 🤝 Contributors

See the [Contributors](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/blob/main/README.md#contributors) section.

## 🙏 Acknowledgments

Special thanks to the supporters listed in the [Acknowledgments](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/blob/main/README.md#acknowledgments) section.