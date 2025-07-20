# 🚀 Claude Code Usage Monitor: Real-time Token Tracking and AI-Powered Predictions

**Tired of guessing your Claude AI token usage?** [Track your Claude Code sessions with precision and predict your limits with the Claude Code Usage Monitor!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

This powerful, real-time terminal tool provides advanced analytics and intelligent session management for your Claude AI usage.  Get detailed insights into token consumption, burn rate, and costs, with AI-driven predictions to help you optimize your workflow.  Enjoy a beautiful, user-friendly Rich UI built with scientific color schemes and WCAG-compliant contrast for ultimate readability.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **🔮 AI-Powered Predictions:** Machine Learning-based token limit predictions, 90th percentile analysis, and session expiry forecasting.
*   **🔄 Real-time Monitoring:**  Configure refresh rates (0.1-20 Hz) and benefit from an intelligent UI with adaptive terminal themes.
*   **📊 Rich Terminal UI:** Color-coded progress bars, sortable tables, and responsive layouts designed with WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Automatic plan switching with custom limit discovery, adapting to your actual usage.
*   **📋 Enhanced Plan Support:** Updated plan limits for Pro (19k), Max5 (88k), Max20 (220k), and Custom (P90-based).
*   **⚠️ Advanced Warning System:**  Multi-level alerts and cost / time predictions to keep you informed.
*   **🎨 Intelligent Theming:**  Scientific color schemes with automatic terminal background detection.
*   **📈 Cost Analytics:**  Model-specific pricing with cache token calculations.
*   **📝 Comprehensive Logging:**  Optional file logging with configurable levels.
*   **🧪 Extensive Testing:**  100+ test cases with full coverage.
*   **⚡ Performance Optimized:**  Advanced caching and efficient data processing.

## Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` offers the fastest and easiest way to install and use the monitor with automatically isolated environments.

1.  **Install `uv`:**

    ```bash
    # On Linux/macOS:
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows:
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    # After installation, restart your terminal
    ```

2.  **Install `claude-monitor`:**

    ```bash
    # Install directly from PyPI with uv
    uv tool install claude-monitor

    # Run from anywhere
    claude-monitor  # or cmonitor, ccmonitor for short
    ```

3.  **Install from Source**

    ```bash
    # Clone and install from source
    git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
    cd Claude-Code-Usage-Monitor
    uv tool install .

    # Run from anywhere
    claude-monitor
    ```

### 📦 Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

>   **⚠️ PATH Setup**: If you see `WARNING: The script claude-monitor is installed in '/home/username/.local/bin' which is not on PATH`, follow the `export PATH` command.
>
>   **⚠️ Important**:  For modern Linux distributions, consider `uv`, `pipx`, or a virtual environment to avoid "externally-managed-environment" errors.  See the [Troubleshooting](#troubleshooting) section for more details.

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

## Usage

### Basic Usage

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

### Configuration Options

*   **Plan Selection:**
    *   `--plan custom` (Default - P90 auto-detect)
    *   `--plan pro` (19,000 tokens)
    *   `--plan max5` (88,000 tokens)
    *   `--plan max20` (220,000 tokens)
    *   `--plan custom --custom-limit-tokens <value>` (Custom limit)

*   **Custom Reset Times:** `--reset-hour <hour>` (e.g., `--reset-hour 3`)

*   **Performance and Display Configuration:**
    *   `--refresh-rate <seconds>` (Data refresh rate - 1-60 seconds)
    *   `--refresh-per-second <Hz>` (Display refresh rate - 0.1-20 Hz)
    *   `--time-format <12h | 24h | auto>`
    *   `--theme <light | dark | classic | auto>`
    *   `--clear` (Clear saved configuration)

*   **Timezone Configuration:** `--timezone <timezone>` (e.g., `--timezone America/New_York`, `--timezone UTC`)

*   **Logging and Debugging:**
    *   `--debug` (Enable debug logging)
    *   `--log-file <path>` (Log to file)
    *   `--log-level <DEBUG | INFO | WARNING | ERROR | CRITICAL>`

*   **Get Help:** `claude-monitor --help` for a full list of parameters.

### Available Plans

| Plan       | Token Limit     | Best For                         |
| ---------- | --------------- | -------------------------------- |
| **custom** | P90 auto-detect | Intelligent limit detection (default) |
| **pro**    | ~19,000         | Claude Pro subscription         |
| **max5**   | ~88,000         | Claude Max5 subscription        |
| **max20**  | ~220,000        | Claude Max20 subscription       |

#### Advanced Plan Features

*   **P90 Analysis**: Custom plan uses 90th percentile calculations from your usage history
*   **Cost Tracking**: Model-specific pricing with cache token calculations
*   **Limit Detection**: Intelligent threshold detection with 95% confidence

---

**For detailed documentation and examples, please refer to the [original repository](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).**