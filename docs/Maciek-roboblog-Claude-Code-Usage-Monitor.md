# ⏱️ Claude Code Usage Monitor: Real-time Token Tracking for Claude AI

**Effortlessly monitor your Claude AI token usage, predict session limits, and optimize your workflow with this powerful, real-time terminal-based tool.  [Check it out on GitHub!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)**

---

## Key Features

*   **🚀 Real-time Monitoring:** Track token consumption, burn rate, and session limits in real-time with configurable refresh rates.
*   **📊 Advanced Rich UI:** Enjoy a beautiful, color-coded terminal interface with progress bars, tables, and WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Automatically switch plans and discover custom limits based on your usage patterns.
*   **🔮 ML-Based Predictions:** Leverage machine learning for intelligent session limit detection, cost projections, and burn rate analysis.
*   **📈 Comprehensive Analytics:** View daily and monthly usage summaries for long-term trend analysis and budget planning.
*   **✅ Easy Installation:**  Quickly install with `uv tool install claude-monitor` or `pip install claude-monitor`.
*   **💼 Customizable & Configurable:** Configure plans, timezones, logging, and more to suit your needs.

---

## Table of Contents

*   [🚀 What's New in v3.0.0](#-what's-new-in-v300)
*   [🚀 Installation](#-installation)
    *   [⚡ Modern Installation with uv (Recommended)](#-modern-installation-with-uv-recommended)
    *   [📦 Installation with pip](#-installation-with-pip)
    *   [🛠️ Other Package Managers](#️-other-package-managers)
*   [📖 Usage](#-usage)
    *   [Get Help](#get-help)
    *   [Basic Usage](#basic-usage)
    *   [Configuration Options](#configuration-options)
    *   [Available Plans](#available-plans)
    *   [🚀 Usage Examples](#-usage-examples)
*   [✨ Features & How It Works](#-features--how-it-works)
    *   [Current Features](#current-features)
    *   [Understanding Claude Sessions](#understanding-claude-sessions)
    *   [Token Limits by Plan](#token-limits-by-plan)
*   [🔧 Development Installation](#-development-installation)
*   [Troubleshooting](#troubleshooting)
    *   [Installation Issues](#installation-issues)
    *   [Runtime Issues](#runtime-issues)
*   [📞 Contact](#-contact)
*   [📚 Additional Documentation](#-additional-documentation)
*   [📝 License](#-license)
*   [🤝 Contributors](#-contributors)
*   [🙏 Acknowledgments](#-acknowledgments)
*   [Star History](#star-history)

---

## 🚀 What's New in v3.0.0

This major update brings significant improvements:

*   **Complete Architecture Rewrite:**  Improved modular design and testing.
*   **Enhanced Functionality:** ML-based limit detection, updated plan limits (Pro, Max5, Max20), and cost analytics.
*   **New CLI Options:**  Greater control over refresh rates, time formats, logging, and saved configurations.
*   **Breaking Changes:** Package name changed and default plan is now 'custom'.

---

## 🚀 Installation

Choose your preferred installation method:

### ⚡ Modern Installation with uv (Recommended)

**Why uv is the best choice:**
*   ✅ Creates isolated environments automatically (no system conflicts)
*   ✅ No Python version issues
*   ✅ No "externally-managed-environment" errors
*   ✅ Easy updates and uninstallation
*   ✅ Works on all platforms

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

> **⚠️ PATH Setup**: If you see WARNING: The script claude-monitor is installed in '/home/username/.local/bin' which is not on PATH, follow the export PATH command above.

> **⚠️ Important**: On modern Linux distributions (Ubuntu 23.04+, Debian 12+, Fedora 38+), you may encounter an "externally-managed-environment" error. Instead of using --break-system-packages, we strongly recommend:
> 1. **Use uv instead** (see above) - it's safer and easier
> 2. **Use a virtual environment** - python3 -m venv myenv && source myenv/bin/activate
> 3. **Use pipx** - pipx install claude-monitor
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

## 📖 Usage

### Get Help

```bash
# Show help information
claude-monitor --help
```

#### Available Command-Line Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| --plan | string | custom | Plan type: pro, max5, max20, or custom |
| --custom-limit-tokens | int | None | Token limit for custom plan (must be > 0) |
| --view | string | realtime | View type: realtime, daily, or monthly |
| --timezone | string | auto | Timezone (auto-detected). Examples: UTC, America/New_York, Europe/London |
| --time-format | string | auto | Time format: 12h, 24h, or auto |
| --theme | string | auto | Display theme: light, dark, classic, or auto |
| --refresh-rate | int | 10 | Data refresh rate in seconds (1-60) |
| --refresh-per-second | float | 0.75 | Display refresh rate in Hz (0.1-20.0) |
| --reset-hour | int | None | Daily reset hour (0-23) |
| --log-level | string | INFO | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| --log-file | path | None | Log file path |
| --debug | flag | False | Enable debug logging |
| --version, -v | flag | False | Show version information |
| --clear | flag | False | Clear saved configuration |

#### Plan Options

| Plan | Token Limit | Cost Limit       | Description |
|---|---|---|---|
| pro | ~19,000 | $18.00           | Claude Pro subscription |
| max5 | ~88,000 | $35.00           | Claude Max5 subscription |
| max20 | ~220,000 | $140.00          | Claude Max20 subscription |
| custom | P90-based | (default) $50.00 | Auto-detection with ML analysis |

#### Command Aliases

The tool can be invoked using any of these commands:
- claude-monitor (primary)
- claude-code-monitor (full name)
- cmonitor (short)
- ccmonitor (short alternative)
- ccm (shortest)

#### Save Flags Feature

The monitor automatically saves your preferences to avoid re-specifying them on each run:

**What Gets Saved:**
- View type (--view)
- Theme preferences (--theme)
- Timezone settings (--timezone)
- Time format (--time-format)
- Refresh rates (--refresh-rate, --refresh-per-second)
- Reset hour (--reset-hour)
- Custom token limits (--custom-limit-tokens)

**Configuration Location:** ~/.claude-monitor/last_used.json

**Usage Examples:**
```bash
# First run - specify preferences
claude-monitor --plan pro --theme dark --timezone "America/New_York"

# Subsequent runs - preferences automatically restored
claude-monitor --plan pro

# Override saved settings for this session
claude-monitor --plan pro --theme light

# Clear all saved preferences
claude-monitor --clear
```

**Key Features:**
- ✅ Automatic parameter persistence between sessions
- ✅ CLI arguments always override saved settings
- ✅ Atomic file operations prevent corruption
- ✅ Graceful fallback if config files are damaged
- ✅ Plan parameter never saved (must specify each time)

### Basic Usage

#### With uv tool installation (Recommended)

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

#### Development mode

If running from source, use python -m claude_monitor from the src/ directory.

### Configuration Options

#### Specify Your Plan

```bash
# Custom plan with P90 auto-detection (Default)
claude-monitor --plan custom

# Pro plan (~44,000 tokens)
claude-monitor --plan pro

# Max5 plan (~88,000 tokens)
claude-monitor --plan max5

# Max20 plan (~220,000 tokens)
claude-monitor --plan max20

# Custom plan with explicit token limit
claude-monitor --plan custom --custom-limit-tokens 100000
```

#### Custom Reset Times

```bash
# Reset at 3 AM
claude-monitor --reset-hour 3

# Reset at 10 PM
claude-monitor --reset-hour 22
```

#### Usage View Configuration

```bash
# Real-time monitoring with live updates (Default)
claude-monitor --view realtime

# Daily token usage aggregated in table format
claude-monitor --view daily

# Monthly token usage aggregated in table format
claude-monitor --view monthly
```

#### Performance and Display Configuration

```bash
# Adjust refresh rate (1-60 seconds, default: 10)
claude-monitor --refresh-rate 5

# Adjust display refresh rate (0.1-20 Hz, default: 0.75)
claude-monitor --refresh-per-second 1.0

# Set time format (auto-detected by default)
claude-monitor --time-format 24h  # or 12h

# Force specific theme
claude-monitor --theme dark  # light, dark, classic, auto

# Clear saved configuration
claude-monitor --clear
```

#### Timezone Configuration

The default timezone is **auto-detected from your system**. Override with any valid timezone:

```bash
# Use US Eastern Time
claude-monitor --timezone America/New_York

# Use Tokyo time
claude-monitor --timezone Asia/Tokyo

# Use UTC
claude-monitor --timezone UTC

# Use London time
claude-monitor --timezone Europe/London
```

#### Logging and Debugging

```bash
# Enable debug logging
claude-monitor --debug

# Log to file
claude-monitor --log-file ~/.claude-monitor/logs/monitor.log

# Set log level
claude-monitor --log-level WARNING  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Available Plans

| Plan | Token Limit     | Best For |
|---|---|---|
| **custom** | P90 auto-detect | Intelligent limit detection (default) |
| **pro** | ~19,000         | Claude Pro subscription |
| **max5** | ~88,000         | Claude Max5 subscription |
| **max20** | ~220,000        | Claude Max20 subscription |

#### Advanced Plan Features

-   **P90 Analysis**: Custom plan uses 90th percentile calculations from your usage history
-   **Cost Tracking**: Model-specific pricing with cache token calculations
-   **Limit Detection**: Intelligent threshold detection with 95% confidence

### 🚀 Usage Examples

See the original README for usage examples.

---

## ✨ Features & How It Works

### Current Features

*   **🔄 Real-time Monitoring:** Configurable update intervals and display refresh rates.
*   **📊 Rich UI Components:** Progress bars, data tables, and a responsive layout.
*   **📈 Multiple Usage Views:** Realtime, Daily, and Monthly views for detailed analysis.
*   **🔮 Machine Learning Predictions:** P90 calculations, burn rate analytics, and session forecasting.
*   **🤖 Intelligent Auto-Detection:** Automatic theme detection, timezone/format preferences, plan recognition, and limit discovery.

### Understanding Claude Sessions

*   Claude Code operates on a 5-hour rolling session window system.
*   You can have multiple active sessions simultaneously.
*   The monitor calculates your burn rate and predicts when session tokens will deplete.

### Token Limits by Plan

See the original README for plan details.

---

## 🔧 Development Installation

See the original README for details on contributing and development installation.

---

## Troubleshooting

### Installation Issues

See the original README for troubleshooting common installation issues, including the "externally-managed-environment" error.

### Runtime Issues

See the original README for troubleshooting common runtime issues.

---

## 📞 Contact

See the original README for contact information.

---

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

---

## 📝 License

See the original README for the license.

---

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

---

## 🙏 Acknowledgments

See the original README for acknowledgements.

---

## Star History

See the original README for the Star History.