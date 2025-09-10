# Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

**Tired of exceeding your Claude AI token limits?** Stay in control with the Claude Code Usage Monitor, a real-time terminal tool providing advanced analytics, intelligent session limit predictions, and a beautiful Rich UI, now with a complete architecture rewrite! Track your token consumption, analyze your burn rate, and optimize your workflow. [Check out the original repo here!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **🔮 ML-Based Predictions**: Get P90 percentile calculations and intelligent session limit detection for smarter usage.
*   **🔄 Real-Time Monitoring**: Configure refresh rates (0.1-20 Hz) with intelligent display updates for a smooth, dynamic experience.
*   **📊 Advanced Rich UI**: Enjoy a beautiful, color-coded UI with progress bars, tables, and layouts, all WCAG-compliant for readability.
*   **🤖 Smart Auto-Detection**: Automatically switch plans and discover custom limits for efficient usage.
*   **📋 Enhanced Plan Support**: Updated limits for Claude Pro, Max5, Max20, and Custom plans.
*   **⚠️ Advanced Warning System**: Receive multi-level alerts with cost and time predictions to avoid surprises.
*   **📈 Cost Analytics**: Understand your spending with model-specific pricing and cache token calculations.
*   **⚡ Performance Optimized**: Benefit from advanced caching and efficient data processing for a fast experience.

## Installation

### 1. Modern Installation with `uv` (Recommended)

`uv` offers isolated environments, eliminates Python version issues, and simplifies updates.

```bash
# Install from PyPI using uv:
uv tool install claude-monitor

# Run from anywhere:
claude-monitor  # or cmonitor, ccmonitor for short
```

### 2. Installation with `pip`

```bash
pip install claude-monitor

# If command not found, add ~/.local/bin to PATH.
# Add this to ~/.bashrc or ~/.zshrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere:
claude-monitor  # or cmonitor, ccmonitor for short
```

**Important:** If you encounter "externally-managed-environment" errors on modern Linux distributions, using `uv` or a virtual environment (e.g., `python3 -m venv myenv && source myenv/bin/activate`) is strongly recommended.

### 3. Other Package Managers

*   **pipx**: `pipx install claude-monitor`
*   **conda/mamba**: `pip install claude-monitor` (within a conda environment)

## Usage

### Basic Usage

```bash
# Run with default settings (custom plan)
claude-monitor
```

### Command-Line Parameters

*   **`--plan`**: Specify your plan (e.g., `pro`, `max5`, `max20`, `custom`).
*   **`--custom-limit-tokens`**: Set a custom token limit.
*   **`--view`**: Select the view (e.g., `realtime`, `daily`, `monthly`).
*   **`--timezone`**: Set your timezone (e.g., `America/New_York`).
*   **`--time-format`**: Choose time format (e.g., `12h`, `24h`).
*   **`--theme`**: Set the display theme (e.g., `light`, `dark`).
*   **`--refresh-rate`**: Set data refresh rate in seconds.
*   **`--refresh-per-second`**: Display refresh rate in Hz.
*   **`--reset-hour`**: Set the daily reset hour.
*   **`--log-level`**: Configure logging level (e.g., `DEBUG`, `INFO`).
*   **`--log-file`**: Specify a log file path.
*   **`--debug`**: Enable debug logging.
*   **`--version, -v`**: Display version information.
*   **`--clear`**: Clear saved configuration.

### Plan Options

| Plan      | Token Limit | Best For                              |
| --------- | ----------- | ------------------------------------- |
| **custom** | P90-based   | Intelligent limit detection (default) |
| **pro**    | ~19,000     | Claude Pro subscription                |
| **max5**   | ~88,000     | Claude Max5 subscription               |
| **max20**  | ~220,000    | Claude Max20 subscription              |

### Saving Preferences
The monitor automatically saves your preferences for future use. These preferences can be overridden with command-line arguments or cleared with `--clear`.

## Features & How It Works

### v3.0.0 Architecture Overview

*   **Modular Design**: Adheres to the Single Responsibility Principle (SRP) for maintainability.
*   **Pydantic-based Configuration**: Ensures type safety and validation.
*   **Rich Terminal UI**: Adaptive theming for optimal readability.
*   **ML-Powered Analysis**: Includes P90 percentile calculations, burn rate analytics, and session forecasting.

### Key Features

*   **Real-time Monitoring**: Customizable update intervals with intelligent change detection.
*   **Rich UI Components**: Including progress bars, data tables, and a responsive layout.
*   **Multiple Usage Views**: Realtime, daily, and monthly views for comprehensive analysis.
*   **Machine Learning Predictions**: P90 percentile analysis, burn rate analytics, cost projections, and session forecasting.
*   **Intelligent Auto-Detection**: Automatic plan switching, background theme detection, timezone and time format preferences, and limit discovery.

## Development Installation

```bash
# Clone the repository
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor

# Install in development mode
pip install -e .

# Run from source
python -m claude_monitor
```

## Troubleshooting

*   **Installation Issues**: Refer to the "Troubleshooting" section for help with common installation problems, including the "externally-managed-environment" error.
*   **Runtime Issues**: Common runtime issues are included in the "Troubleshooting" section.

## Contact

*   **Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## License

[MIT License](LICENSE)

## Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## Acknowledgments

### Sponsors

**Ed** - *Buy Me Coffee Supporter*

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)
---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>