# Claude Code Usage Monitor: Real-time Token Tracking & AI Session Management

**Stay in control of your Claude AI usage!** The Claude Code Usage Monitor is a powerful, real-time terminal tool for tracking your token consumption, predicting session limits, and optimizing your Claude AI experience. [See it on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   ✅ **Real-time Monitoring:** Track token usage, burn rate, and session duration.
*   🔮 **ML-Powered Predictions:** Intelligent session limit detection and burn rate analysis.
*   📊 **Rich Terminal UI:** Color-coded progress bars, tables, and WCAG-compliant themes.
*   🤖 **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   📈 **Cost Analytics:** Model-specific pricing with cache token calculations.
*   ⏰ **Customizable Alerts:** Multi-level warnings with cost and time predictions.
*   🔄 **Flexible Views:** Real-time, Daily, and Monthly usage summaries.

## Installation

### Recommended: Installation with uv

**uv** is a modern, fast package manager that simplifies installation and avoids common Python environment issues.

```bash
# Install directly from PyPI with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### Alternative: Installation with pip

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within your conda environment)

See the full README for detailed instructions, including troubleshooting and alternative installation methods.

## Usage

### Basic Usage

Run the monitor with the default "Custom" plan:

```bash
claude-monitor
```

Press `Ctrl+C` to exit.

### Configuration Options

Customize your monitoring experience:

*   **Plan Selection:** `claude-monitor --plan pro`, `--plan max5`, `--plan max20`, or `--plan custom` (default).
*   **Custom Token Limits:** `--plan custom --custom-limit-tokens 100000`
*   **Views:** `--view realtime` (default), `--view daily`, `--view monthly`
*   **Refresh Rate:** `--refresh-rate 5` (update every 5 seconds) `--refresh-per-second 1.0` (Display refresh rate)
*   **Timezone:** `--timezone America/New_York` (or any valid timezone)
*   **Logging and Debugging:** `--log-level DEBUG`, `--log-file /path/to/log.txt`
*   **Theme**:  `--theme dark` or `--theme light`
*   **Clear Config:** `--clear` to clear saved settings.

Full command-line parameter options available with: `claude-monitor --help`

## Available Plans

| Plan        | Token Limit        | Best For                        |
| ----------- | ------------------ | ------------------------------- |
| **custom**   | P90 auto-detect   | Intelligent limit detection (default) |
| **pro**      | ~19,000           | Claude Pro subscription         |
| **max5**     | ~88,000           | Claude Max5 subscription        |
| **max20**    | ~220,000          | Claude Max20 subscription       |

## Advanced Features & How It Works

This project uses a modular architecture, featuring:

*   **Real-time Monitoring:** Tracks usage at configurable intervals.
*   **Machine Learning Predictions:**  P90 percentile analysis for intelligent limit detection and burn rate analysis.
*   **Customizable UI:** Offers a rich terminal user interface.
*   **Intelligent Auto-Detection:** Automatically adapts to your usage patterns.
*   **Advanced Session Control:** Auto-detection and management.

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>