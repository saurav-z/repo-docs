# Claude Code Usage Monitor: Real-Time Token Tracking for Claude AI

**Effortlessly monitor your Claude AI token usage with this powerful terminal tool, gaining insights into your usage, cost, and AI session limits. Check out the original repo at [https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)!**

---

## Key Features

*   **Real-time Monitoring:** Track token consumption, burn rate, and cost with configurable refresh rates.
*   **Advanced Rich UI:** Beautiful, color-coded progress bars and tables with WCAG-compliant contrast.
*   **ML-Based Predictions:** Intelligent session limit detection and forecasting based on machine learning.
*   **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **Cost Analytics:** Model-specific pricing and cache token calculations.
*   **Custom Plan Support:** Specifically designed for 5-hour Claude Code sessions, providing personalized limits based on your usage.

## Installation

### ⚡ Modern Installation with uv (Recommended)

uv streamlines installation and eliminates version conflicts.

```bash
# Install with uv
uv tool install claude-monitor

# Run
claude-monitor
```

See the original [README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor) for detailed installation instructions.

### 📦 Installation with pip

```bash
pip install claude-monitor
claude-monitor # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` within a conda environment

## Usage

### Basic Usage

```bash
claude-monitor  # or cmonitor, ccmonitor
```

Press `Ctrl+C` to exit the monitor gracefully.

### Configuration Options

*   `--plan`: Select plan (e.g., `pro`, `max5`, `max20`, or `custom`).  `custom` is the default, utilizing machine learning to determine the best limits for your usage.
*   `--custom-limit-tokens`: Specify token limit for custom plan.
*   `--view`:  Choose view type (`realtime`, `daily`, `monthly`).
*   `--timezone`:  Set your timezone (auto-detects by default).
*   `--time-format`:  Choose time format (`12h`, `24h`, or `auto`).
*   `--theme`: Select a display theme (`light`, `dark`, `classic`, or `auto`).
*   `--refresh-rate`: Refresh rate in seconds.
*   `--refresh-per-second`: Display refresh rate in Hz.
*   `--reset-hour`: Set the daily reset hour.
*   `--log-level`: Set the logging level.
*   `--log-file`: Specify a log file path.
*   `--debug`: Enable debug logging.
*   `--clear`: Clear saved configuration.

### Available Plans

*   **custom:** P90 auto-detect (default) - intelligent limit detection
*   **pro:** ~19,000 tokens - Claude Pro subscription
*   **max5:** ~88,000 tokens - Claude Max5 subscription
*   **max20:** ~220,000 tokens - Claude Max20 subscription

## v3.0.0: Major Updates

*   **Complete Architecture Rewrite:** Modular design with SRP compliance and a comprehensive test suite.
*   **Enhanced Functionality:** Machine learning-based limit detection, updated plan limits, and advanced cost analytics.
*   **New CLI Options:**  More granular control over refresh rates, time format, and logging.

See the original [README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor) for a detailed architecture overview, data flow, and technical requirements.

## Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

Special thanks to our supporters, including **Ed**.

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>