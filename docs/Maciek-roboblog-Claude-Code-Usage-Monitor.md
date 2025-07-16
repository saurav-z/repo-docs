# Claude Code Usage Monitor: Stay Ahead of Your Claude AI Token Limits 🚀

Are you a Claude AI user looking to optimize your token usage and prevent unexpected session limits? **Claude Code Usage Monitor** is the perfect solution!  This powerful, real-time terminal monitoring tool provides advanced analytics, machine-learning-based predictions, and a beautiful Rich UI, helping you track your Claude AI token consumption with ease.  Check out the original repo [here](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **🔮 ML-Based Predictions:**  Intelligent session limit detection and P90 percentile calculations.
*   **🔄 Real-Time Monitoring:** Configurable refresh rates (0.1-20 Hz) with dynamic display updates.
*   **📊 Advanced Rich UI:** Stunning color-coded progress bars, tables, and layouts with WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **💼 Professional Architecture:**  Modular design with Single Responsibility Principle (SRP) compliance.
*   **🎨 Intelligent Theming:**  Automatic terminal background detection for optimal readability.
*   **⏰ Advanced Scheduling:**  Auto-detected system timezone and time format preferences.
*   **📈 Cost Analytics:** Model-specific pricing and cache token calculations.
*   **📝 Comprehensive Logging:** Optional file logging with configurable levels.
*   **🧪 Extensive Testing:** 100+ test cases ensuring robust performance.

## Installation

### Recommended: Install with `uv` (Fast & Easy)

`uv` is the preferred way to install for simplicity and to avoid potential Python environment conflicts.

```bash
# Install directly from PyPI with uv
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

### Other Installation Methods

*   **📦 Installation with `pip`:**

    ```bash
    pip install claude-monitor
    claude-monitor
    ```
    *(Remember to add the install location to your PATH if needed – see the original README)*

*   **🛠️ Install with `pipx`:**

    ```bash
    pipx install claude-monitor
    ```

*   **💻 Install with `conda/mamba`:**

    ```bash
    # Install with pip in conda environment
    pip install claude-monitor
    ```
---
## Usage

### Basic Commands and Configuration

*   **Get Help:** `claude-monitor --help`
*   **Run the Monitor:** `claude-monitor` (or one of the shorthand aliases).

### Key Command-Line Parameters

| Parameter             | Type    | Default      | Description                                                 |
|-----------------------|---------|--------------|-------------------------------------------------------------|
| `--plan`              | string  | `custom`     | Plan type: `pro`, `max5`, `max20`, or `custom`              |
| `--custom-limit-tokens` | integer | `None`       | Token limit for custom plan (must be > 0)                   |
| `--timezone`          | string  | `auto`       | Timezone (e.g., `UTC`, `America/New_York`).                 |
| `--time-format`       | string  | `auto`       | Time format: `12h`, `24h`, or `auto`.                      |
| `--theme`             | string  | `auto`       | Display theme: `light`, `dark`, `classic`, or `auto`.      |
| `--refresh-rate`      | integer | `10`         | Data refresh rate in seconds (1-60)                         |
| `--refresh-per-second` | float   | `0.75`       | Display refresh rate in Hz (0.1-20.0)                       |
| `--reset-hour`        | integer | `None`       | Daily reset hour (0-23)                                     |
| `--log-level`         | string  | `INFO`       | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`|
| `--log-file`          | path    | `None`       | Log file path                                               |
| `--debug`             | flag    | `False`      | Enable debug logging                                        |
| `--version`, `-v`     | flag    | `False`      | Show version information                                    |
| `--clear`             | flag    | `False`      | Clear saved configuration                                    |

### Available Plans

| Plan       | Token Limit   | Best For                           |
|------------|---------------|------------------------------------|
| `custom`   | P90 auto-detect | Intelligent limit detection (default) |
| `pro`      | ~19,000       | Claude Pro subscription            |
| `max5`     | ~88,000       | Claude Max5 subscription           |
| `max20`    | ~220,000      | Claude Max20 subscription          |

## What's New in v3.0.0

*   **Complete Architecture Rewrite:**  Improved modularity, type safety, and testing.
*   **Enhanced Functionality:**  Machine-learning-based P90 limit detection, updated plan limits.
*   **Expanded CLI Options:** Fine-grained control over refresh rates, time formats, and more.
*   **Breaking Changes:** Package name changed to `claude-monitor`, the default plan is now custom.

## Development Installation

Detailed development setup instructions are in the original README.

## Troubleshooting

*   **Installation Errors:** See the original README for solutions to common issues.
*   **Runtime Issues:** Ensure you have sent at least two messages for the initial setup. Specify a custom config path as necessary.

## Support

*   **Contact:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)
*   **Documentation:** [Development Roadmap](DEVELOPMENT.md) , [Contributing Guide](CONTRIBUTING.md), [Troubleshooting](TROUBLESHOOTING.md)
*   **License:** [MIT License](LICENSE)

## Contributors

A big thank you to the project contributors:
[@adawalli](https://github.com/adawalli)
[@taylorwilsdon](https://github.com/taylorwilsdon)
[@moneroexamples](https://github.com/moneroexamples)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐  Find this tool useful? Star the repo!  ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>