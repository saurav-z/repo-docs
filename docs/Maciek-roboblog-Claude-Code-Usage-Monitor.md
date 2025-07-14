# Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

**Tired of running out of tokens in the middle of your Claude session?**  Stay in control with the **Claude Code Usage Monitor**, a powerful and intuitive terminal tool to track your Claude AI token usage, predict session limits, and optimize your workflow.  [Check out the repo for the latest updates!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **🚀 Real-time Monitoring:** Track token consumption, burn rate, and cost in real-time with configurable refresh rates.
*   **🔮 AI-Powered Predictions:**  Machine learning models provide intelligent session limit detection and accurate usage forecasts.
*   **📊 Rich Terminal UI:**  Enjoy a beautiful, color-coded terminal interface with WCAG-compliant contrast for optimal readability.
*   **🤖 Smart Auto-Detection:** Automatically switch between usage plans and discover custom limits based on your usage history.
*   **📋 Comprehensive Plan Support:** Supports various Claude plans (Pro, Max5, Max20, and Custom) with updated token limits.
*   **⚠️ Advanced Warning System:** Receive multi-level alerts with cost and time predictions to prevent unexpected interruptions.
*   **🎨 Intelligent Theming:**  Automatic terminal background detection ensures a visually appealing and accessible user experience.
*   **📈 Cost Analytics:** Gain insights into model-specific pricing and utilize cache token calculations for improved cost awareness.
*   **⚡ Performance Optimized:** Advanced caching and efficient data processing ensure a smooth and responsive experience.
*   **📝 Type-Safe Configuration:** Configuration is type-safe with Pydantic validation.

### v3.0.0 - Major Update Highlights:

*   **Complete Architecture Rewrite:** Improved modular design and adherence to Single Responsibility Principle.
*   **P90 Analysis:** ML-based limit detection with intelligent session limit detection.
*   **Enhanced UI:** More color coded progress bars and tables with WCAG compliant contrast.
*   **Updated Plan Limits:** Supports Claude Pro (19k tokens), Max5 (88k tokens), Max20 (220k tokens), and Custom plans.
*   **New CLI Options:** Configure refresh rates, timezone, display theme, and more.

---

## Installation

### ⚡ Modern Installation with uv (Recommended)

**Why uv is the best choice:**
- ✅ Creates isolated environments automatically (no system conflicts)
- ✅ No Python version issues
- ✅ No "externally-managed-environment" errors
- ✅ Easy updates and uninstallation
- ✅ Works on all platforms

The fastest and easiest way to install and use the monitor:

[![PyPI](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)

#### Install from PyPI

```bash
# Install directly from PyPI with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

#### Install from Source

```bash
# Clone and install from source
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
uv tool install .

# Run from anywhere
claude-monitor
```

#### First-time uv users
If you don't have uv installed yet, get it with one command:

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

### 📦 Installation with pip

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

>
> **⚠️ PATH Setup**: If you see WARNING: The script claude-monitor is installed in '/home/username/.local/bin' which is not on PATH, follow the export PATH command above.
>
> **⚠️ Important**: On modern Linux distributions (Ubuntu 23.04+, Debian 12+, Fedora 38+), you may encounter an "externally-managed-environment" error. Instead of using --break-system-packages, we strongly recommend:
> 1. **Use uv instead** (see above) - it's safer and easier
> 2. **Use a virtual environment** - python3 -m venv myenv && source myenv/bin/activate
> 3. **Use pipx** - pipx install claude-monitor
>
> See the Troubleshooting section for detailed solutions.

### 🛠️ Other Package Managers

#### pipx (Isolated Environments)

```bash
# Install with pipx
pipx install claude-monitor

# Run from anywhere
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```

#### conda/mamba

```bash
# Install with pip in conda environment
pip install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

---

## Usage

### Basic Usage

```bash
# Run with default settings (Custom plan with auto-detection)
claude-monitor
```

### Command-Line Options

Use these options to customize the monitor to your needs:

```bash
claude-monitor --help # Show help information
```

**Key Parameters:**

| Parameter              | Type      | Default | Description                                                                |
| ---------------------- | --------- | ------- | -------------------------------------------------------------------------- |
| `--plan`               | string    | `custom` | Plan type: `pro`, `max5`, `max20`, or `custom`                             |
| `--custom-limit-tokens` | int       | `None`  | Token limit for custom plan (must be > 0)                                  |
| `--timezone`           | string    | `auto`  | Timezone (auto-detected). Examples: `UTC`, `America/New_York`, `Europe/London` |
| `--time-format`        | string    | `auto`  | Time format: `12h`, `24h`, or `auto`                                      |
| `--theme`              | string    | `auto`  | Display theme: `light`, `dark`, `classic`, or `auto`                       |
| `--refresh-rate`       | int       | `10`    | Data refresh rate in seconds (1-60)                                        |
| `--refresh-per-second` | float     | `0.75`  | Display refresh rate in Hz (0.1-20.0)                                      |
| `--reset-hour`         | int       | `None`  | Daily reset hour (0-23)                                                     |
| `--log-level`          | string    | `INFO`  | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`             |
| `--log-file`           | path      | `None`  | Log file path                                                              |
| `--debug`              | flag      | `False` | Enable debug logging                                                       |
| `--version`, `-v`      | flag      | `False` | Show version information                                                   |
| `--clear`              | flag      | `False` | Clear saved configuration                                                  |

### Plan Options

Choose the plan that matches your Claude subscription:

| Plan     | Token Limit       | Best For                  |
|----------|-------------------|---------------------------|
| `custom` | P90 auto-detect   | Intelligent limit detection (default) |
| `pro`    | ~19,000          | Claude Pro subscription   |
| `max5`   | ~88,000          | Claude Max5 subscription  |
| `max20`  | ~220,000         | Claude Max20 subscription |

### Example Usage

```bash
# Start with the Custom Plan (Recommended)
claude-monitor

# Specify your plan
claude-monitor --plan pro
claude-monitor --plan max5

# Set a reset hour for your work schedule
claude-monitor --reset-hour 9 --timezone America/Los_Angeles

# Enable debug logging
claude-monitor --debug
```

---

## Development

See the [Development Installation](#-development-installation) section for information on setting up a development environment and contributing to the project.

---

## Troubleshooting

See the [Troubleshooting](#troubleshooting) section for common installation and runtime issues and solutions.

---

## Contact

For any questions or collaboration inquiries, please contact:

**📧 Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

---

## Additional Information

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

A special thanks to our supporters!

A special thanks to:

**Ed** - *Buy Me Coffee Supporter*
> "I appreciate sharing your work with the world. It helps keep me on track with my day. Quality readme, and really good stuff all around!"

### Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>