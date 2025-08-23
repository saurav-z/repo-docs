# 📊 Claude Code Usage Monitor: Stay Ahead of Your Token Limits!

Tired of hitting those Claude AI token limits? This real-time terminal monitor provides beautiful, insightful analytics and intelligent predictions to help you optimize your Claude Code usage.  [Check out the original repo here!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

**Key Features:**

*   🔮 **ML-Based Predictions:** Get intelligent session limit detection using 90th percentile calculations.
*   🔄 **Real-Time Monitoring:** Configurable refresh rates (0.1-20 Hz) for up-to-the-second updates.
*   📊 **Advanced Rich UI:** Beautiful color-coded progress bars, tables, and WCAG-compliant layouts.
*   🤖 **Smart Auto-Detection:** Automatically adjusts to your plan and custom limit discovery.
*   📈 **Cost Analytics:** Model-specific pricing and cache token calculations.
*   ⚠️ **Advanced Warning System:** Multi-level alerts with cost and time predictions.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

**Table of Contents:**

*   [✨ Key Features](#-key-features)
*   [🚀 Installation](#-installation)
    *   [⚡ Modern Installation with uv (Recommended)](#-modern-installation-with-uv-recommended)
    *   [📦 Installation with pip](#-installation-with-pip)
    *   [🛠️ Other Package Managers](#️-other-package-managers)
*   [📖 Usage](#-usage)
    *   [Get Help](#get-help)
    *   [Basic Usage](#basic-usage)
    *   [Configuration Options](#configuration-options)
    *   [Available Plans](#available-plans)
*   [🚀 What's New in v3.0.0](#-whats-new-in-v300)
*   [✨ Features & How It Works](#-features--how-it-works)
*   [🚀 Usage Examples](#-usage-examples)
*   [🔧 Development Installation](#-development-installation)
*   [Troubleshooting](#troubleshooting)
*   [📞 Contact](#-contact)
*   [📚 Additional Documentation](#-additional-documentation)
*   [📝 License](#-license)
*   [🤝 Contributors](#-contributors)
*   [🙏 Acknowledgments](#-acknowledgments)
*   [⭐ Star History](#star-history)

---

## ✨ Key Features

### 🚀 **v3.0.0: Complete Architecture Rewrite**

*   🔮 **ML-based predictions**: P90 percentile calculations and intelligent session limit detection
*   🔄 **Real-time monitoring**: Configurable refresh rates with intelligent display updates.
*   📊 **Advanced Rich UI**: Color-coded progress bars, sortable tables, and WCAG-compliant contrast.
*   🤖 **Smart auto-detection**: Automatic plan switching with custom limit discovery.
*   📋 **Enhanced plan support**: Updated limits (Pro, Max5, Max20, Custom).
*   ⚠️ **Advanced warning system**: Multi-level alerts for cost and time predictions.
*   🎨 **Intelligent theming**: Automatic terminal background detection for optimal readability.
*   ⏰ **Advanced scheduling**: Auto-detected system timezone and time format preferences.
*   📈 **Cost analytics**: Model-specific pricing with cache token calculations.
*   📝 **Comprehensive logging**: Optional file logging with configurable levels.
*   🧪 **Extensive testing**: 100+ test cases with full coverage.
*   🎯 **Error reporting**: Optional Sentry integration for production monitoring.
*   ⚡ **Performance optimized**: Advanced caching and efficient data processing.

### 📋 Default Custom Plan

The **Custom plan** is the default and designed for 5-hour Claude Code sessions, monitoring token usage, message count, and cost. It analyzes your usage patterns over the last 192 hours to calculate personalized limits, providing accurate predictions and tailored warnings.

---

## 🚀 Installation

### ⚡ Modern Installation with uv (Recommended)

**Why uv is the best choice:**

*   ✅ Creates isolated environments automatically (no system conflicts)
*   ✅ No Python version issues
*   ✅ No "externally-managed-environment" errors
*   ✅ Easy updates and uninstallation
*   ✅ Works on all platforms

**Installation via uv:**

```bash
# Install directly from PyPI with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

**Install from Source with uv:**

```bash
# Clone and install from source
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
uv tool install .

# Run from anywhere
claude-monitor
```

**First-time uv users:**

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
> 1.  **Use uv instead** (see above) - it's safer and easier
> 2.  **Use a virtual environment** - `python3 -m venv myenv && source myenv/bin/activate`
> 3.  **Use pipx** - `pipx install claude-monitor`
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

## 📖 Usage

### Get Help

```bash
# Show help information
claude-monitor --help
```

**Command-Line Parameters:**

| Parameter              | Type   | Default   | Description                                                     |
| ---------------------- | ------ | --------- | --------------------------------------------------------------- |
| `--plan`               | string | `custom`  | Plan type: `pro`, `max5`, `max20`, or `custom`                  |
| `--custom-limit-tokens` | int    | `None`    | Token limit for custom plan (must be > 0)                     |
| `--view`               | string | `realtime` | View type: `realtime`, `daily`, or `monthly`                   |
| `--timezone`           | string | `auto`    | Timezone (auto-detected). Examples: `UTC`, `America/New_York`    |
| `--time-format`        | string | `auto`    | Time format: `12h`, `24h`, or `auto`                           |
| `--theme`              | string | `auto`    | Display theme: `light`, `dark`, `classic`, or `auto`           |
| `--refresh-rate`       | int    | `10`      | Data refresh rate in seconds (1-60)                             |
| `--refresh-per-second` | float  | `0.75`    | Display refresh rate in Hz (0.1-20.0)                            |
| `--reset-hour`         | int    | `None`    | Daily reset hour (0-23)                                        |
| `--log-level`          | string | `INFO`    | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file`           | path   | `None`    | Log file path                                                  |
| `--debug`              | flag   | `False`   | Enable debug logging                                           |
| `--version, -v`        | flag   | `False`   | Show version information                                       |
| `--clear`              | flag   | `False`   | Clear saved configuration                                      |

**Plan Options:**

| Plan      | Token Limit | Cost Limit  | Description                        |
| --------- | ----------- | ----------- | ---------------------------------- |
| `pro`     | ~19,000     | $18.00      | Claude Pro subscription            |
| `max5`    | ~88,000     | $35.00      | Claude Max5 subscription           |
| `max20`   | ~220,000    | $140.00     | Claude Max20 subscription          |
| `custom`  | P90-based   | (default) $50.00 | Auto-detection with ML analysis  |

**Command Aliases:**

*   `claude-monitor` (primary)
*   `claude-code-monitor` (full name)
*   `cmonitor` (short)
*   `ccmonitor` (short alternative)
*   `ccm` (shortest)

**Save Flags Feature:** The monitor saves your preferences to avoid re-specifying them on each run:

*   **Saved Settings:** View type, theme, timezone, time format, refresh rates, reset hour, custom token limits.
*   **Configuration Location:** `~/.claude-monitor/last_used.json`

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

*   ✅ Automatic parameter persistence between sessions
*   ✅ CLI arguments always override saved settings
*   ✅ Atomic file operations prevent corruption
*   ✅ Graceful fallback if config files are damaged
*   ✅ Plan parameter never saved (must specify each time)

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

If running from source, use `python -m claude_monitor` from the `src/` directory.

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

| Plan        | Token Limit     | Best For                     |
| ----------- | --------------- | ---------------------------- |
| **custom**  | P90 auto-detect | Intelligent limit detection (default) |
| **pro**     | ~19,000         | Claude Pro subscription      |
| **max5**    | ~88,000         | Claude Max5 subscription     |
| **max20**   | ~220,000        | Claude Max20 subscription    |

**Advanced Plan Features:**

*   **P90 Analysis**: Custom plan uses 90th percentile calculations from your usage history.
*   **Cost Tracking**: Model-specific pricing with cache token calculations.
*   **Limit Detection**: Intelligent threshold detection with 95% confidence.

---

## 🚀 What's New in v3.0.0

### Major Changes

#### **Complete Architecture Rewrite**
*   Modular design with Single Responsibility Principle (SRP) compliance
*   Pydantic-based configuration with type safety and validation
*   Advanced error handling with optional Sentry integration
*   Comprehensive test suite with 100+ test cases

#### **Enhanced Functionality**
*   **P90 Analysis**: Machine learning-based limit detection using 90th percentile calculations
*   **Updated Plan Limits**: Pro (44k), Max5 (88k), Max20 (220k) tokens
*   **Cost Analytics**: Model-specific pricing with cache token calculations
*   **Rich UI**: WCAG-compliant themes with automatic terminal background detection

#### **New CLI Options**
*   `--refresh-per-second`: Configurable display refresh rate (0.1-20 Hz)
*   `--time-format`: Automatic 12h/24h format detection
*   `--custom-limit-tokens`: Explicit token limits for custom plans
*   `--log-file` and `--log-level`: Advanced logging capabilities
*   `--clear`: Reset saved configuration
*   Command aliases: `claude-code-monitor`, `cmonitor`, `ccmonitor`, `ccm` for convenience

#### **Breaking Changes**
*   Package name changed from `claude-usage-monitor` to `claude-monitor`
*   Default plan changed from `pro` to `custom` (with auto-detection)
*   Minimum Python version increased to 3.9+
*   Command structure updated (see examples above)

---

## ✨ Features & How It Works

### v3.0.0 Architecture Overview

The new version features a complete rewrite with modular architecture following Single Responsibility Principle (SRP).

### 🖥️ User Interface Layer

| Component            | Description           |
| -------------------- | --------------------- |
| **CLI Module**       | Pydantic-based        |
| **Settings/Config**  | Type-safe             |
| **Error Handling**   | Sentry-ready          |
| **Rich Terminal UI** | Adaptive Theme        |

---

### 🎛️ Monitoring Orchestrator

| Component                | Key Responsibilities                                             |
| ------------------------ | ---------------------------------------------------------------- |
| **Central Control Hub**  | Session Mgmt · Real-time Data Flow · Component Coordination      |
| **Data Manager**         | Cache Mgmt · File I/O · State Persist                           |
| **Session Monitor**      | Real-time · 5 hr Windows · Token Track                           |
| **UI Controller**        | Rich Display · Progress Bars · Theme System                     |
| **Analytics**            | P90 Calculator · Burn Rate · Predictions                        |

---

### 🏗️ Foundation Layer

| Component           | Core Features                                           |
| ------------------- | ------------------------------------------------------- |
| **Core Models**     | Session Data · Config Schema · Type Safety             |
| **Analysis Engine** | ML Algorithms · Statistical · Forecasting              |
| **Terminal Themes** | Auto-detection · WCAG Colors · Contrast Opt            |
| **Claude API Data** | Token Tracking · Cost Calculator · Session Blocks      |

---

**🔄 Data Flow:** Claude Config Files → Data Layer → Analysis Engine → UI Components → Terminal Display

### Current Features

#### 🔄 Advanced Real-time Monitoring

*   Configurable update intervals (1-60 seconds)
*   High-precision display refresh (0.1-20 Hz)
*   Intelligent change detection to minimize CPU usage
*   Multi-threaded orchestration with callback system

#### 📊 Rich UI Components

*   **Progress Bars**: WCAG-compliant color schemes.
*   **Data Tables**: Sortable columns with model-specific statistics.
*   **Layout Manager**: Responsive design that adapts to terminal size.
*   **Theme System**: Auto-detects terminal background.

#### 📈 Multiple Usage Views

*   **Realtime View** (Default): Live monitoring.
*   **Daily View**: Aggregated daily statistics.
*   **Monthly View**: Monthly aggregated data.

#### 🔮 Machine Learning Predictions

*   **P90 Calculator**: 90th percentile analysis for limit detection.
*   **Burn Rate Analytics**: Multi-session consumption pattern analysis.
*   **Cost Projections**: Model-specific pricing.
*   **Session Forecasting**: Predicts session expiration based on usage patterns.

#### 🤖 Intelligent Auto-Detection

*   **Background Detection**: Automatically determines terminal theme.
*   **System Integration**: Auto-detects timezone and time format preferences.
*   **Plan Recognition**: Analyzes usage patterns to suggest optimal plans.
*   **Limit Discovery**: Scans historical data to find actual token limits.

### Understanding Claude Sessions

#### How Claude Code Sessions Work

Claude Code uses a **5-hour rolling session window system**:

1.  **Session Start**: Begins with your first message to Claude.
2.  **Session Duration**: Lasts exactly 5 hours from that first message.
3.  **Token Limits**: Apply within each 5-hour session window.
4.  **Multiple Sessions**: Can have several active sessions simultaneously.
5.  **Rolling Windows**: New sessions can start while others are still active.

#### Session Reset Schedule

**Example Session Timeline:**

*   10:30 AM - First message (Session A starts at 10 AM)
*   03:00 PM - Session A expires (5 hours later)
*   12:15 PM - First message (Session B starts 12PM)
*   05:15 PM - Session B expires (5 hours later 5PM)

#### Burn Rate Calculation

The monitor calculates burn rate using:

1.  **Data Collection**: Gathers token usage from all sessions in the last hour.
2.  **Pattern Analysis**: Identifies consumption trends.
3.  **Velocity Tracking**: Calculates tokens consumed per minute.
4.  **Prediction Engine**: Estimates when current session tokens will deplete.
5.  **Real-time Updates**: Adjusts predictions as usage patterns change.

### Token Limits by Plan

#### v3.0.0 Updated Plan Limits

| Plan           | Limit (Tokens) | Cost Limit  | Messages | Algorithm         |
| -------------- | -------------- | ----------- | -------- | ----------------- |
| **Claude Pro**   | 19,000         | $18.00      | 250      | Fixed limit       |
| **Claude Max5**  | 88,000         | $35.00      | 1,000    | Fixed limit       |
| **Claude Max20** | 220,000        | $140.00     | 2,000    | Fixed limit       |
| **Custom**       | P90-based      | (default) $50.00| 250+      | Machine learning  |

#### Advanced Limit Detection

*   **P90 Analysis**: Uses 90th percentile of your historical usage.
*   **Confidence Threshold**: 95% accuracy in limit detection.
*   **Cache Support**: Includes cache creation and read token costs.
*   **Model-Specific**: Adapts to Claude 3.5, Claude 4, and future models.

### Technical Requirements

#### Dependencies (v3.0.0)

```toml
# Core dependencies (automatically installed)
pytz>=2023.3                # Timezone handling
rich>=13.7.0                # Rich terminal UI
pydantic>=2.0.0             # Type validation
pydantic-settings>=2.0.0    # Configuration management
numpy>=1.21.0               # Statistical calculations
sentry-sdk>=1.40.0          # Error reporting (optional)
pyyaml>=6.0                 # Configuration files
tzdata                      # Windows timezone data
```

#### Python Requirements

*   **Minimum**: Python 3.9+
*   **Recommended**: Python 3.11+
*   **Tested on**: Python 3.9, 3.10, 3.11, 3.12, 3.13

### Smart Detection Features

#### Automatic Plan Switching

When using the default Pro plan:

1.  **Detection**: Monitor notices token usage exceeding 7,000.
2.  **Analysis**: Scans previous sessions for actual limits.
3.  **Switch**: Automatically changes to `custom` mode.
4.  **Notification**: Displays clear message about the change.
5.  **Continuation**: Keeps monitoring with the new, higher limit.

#### Limit Discovery Process

The auto-detection system:

1.  **Scans History**: Examines all available session blocks.
2.  **Finds Peaks**: Identifies highest token usage achieved.
3.  **Validates Data**: Ensures data quality and recency.
4.  **Sets Limits**: Uses discovered maximum as the new limit.
5.  **Learns Patterns**: Adapts to your actual usage capabilities.

---

## 🚀 Usage Examples

### Common Scenarios

#### 🌅 Morning Developer

```bash
# Set custom reset time to 9 AM
./claude_monitor.py --reset-hour 9

# With your timezone
./claude_monitor.py --reset-hour 9 --timezone US/Eastern
```

**Benefits**:

*   Reset times align with your work schedule.
*   Better planning for daily token allocation.
*   Predictable session windows.

#### 🌙 Night Owl Coder

```bash
# Reset at midnight
./claude_monitor.py --reset-hour 0

# Late evening reset (11 PM)
./claude_monitor.py --reset-hour 23
```

**Strategy**:

*   Plan heavy coding sessions around reset times.
*   Use late resets to span midnight work sessions.
*   Monitor burn rate during peak hours.

#### 🔄 Heavy User with Variable Limits

```bash
# Auto-detect your highest previous usage
claude-monitor --plan custom
```

**Approach**:

*   Let auto-detection find your real limits.
*   Monitor for a week to understand patterns.
*   Note when limits change or reset.

#### 🌍 International User

```bash
# US East Coast
claude-monitor --timezone America/New_York

# Europe
claude-monitor --timezone Europe/London

# Asia Pacific
claude-monitor --timezone Asia/Singapore

# UTC for international team coordination
claude-monitor --timezone UTC --reset-hour 12
```

#### ⚡ Quick Check

```bash
# Just run it with defaults
claude-monitor

# Press Ctrl+C after checking status
```

#### 📊 Usage Analysis Views

```bash
# View daily usage breakdown with detailed statistics
claude-monitor --view daily

# Analyze monthly token consumption trends
claude-monitor --view monthly --plan max20

# Export daily usage data to log file for analysis
claude-monitor --view daily --log-file ~/daily-usage.log

# Review usage in different timezone
claude-monitor --view daily --timezone America/New_York
```

**Use Cases**:

*   **Realtime**: Live monitoring.
*   **Daily**: Analyze daily consumption patterns.
*   **Monthly**: Long-term trend analysis.

### Plan Selection Strategies

#### How to Choose Your Plan

**Start with Default (Recommended for New Users)**

```bash
# Pro plan detection with auto-switching
claude-monitor
```

*   Monitor will detect if you exceed Pro limits.
*   Automatically switches to `custom` if needed.
*   Shows notification when switching occurs.

**Known Subscription Users**

```bash
# If you know you have Max5
claude-monitor --plan max5

# If you know you have Max20
claude-monitor --plan max20
```

**Unknown Limits**

```bash
# Auto-detect from previous usage
claude-monitor --plan custom
```

### Best Practices

#### Setup Best Practices

1.  **Start Early in Sessions**

```bash
# Begin monitoring when starting Claude work (uv installation)
claude-monitor

# Or development mode
./claude_monitor.py
```

*   Gives accurate session tracking from the start.
*   Better burn rate calculations.
*   Early warning for limit approaches.

2.  **Use Modern Installation (Recommended)**

```bash
# Easy installation and updates with uv
uv tool install claude-monitor
claude-monitor --plan max5
```

*   Clean system installation.
*   Easy updates and maintenance.
*   Available from anywhere.

3.  **Custom Shell Alias (Legacy Setup)**

```bash
# Add to ~/.bashrc or ~/.zshrc (only for development setup)
alias claude-monitor='cd ~/Claude-Code-Usage-Monitor && source venv/bin/activate && ./claude_monitor.py'
```

#### Usage Best Practices

1.  **Monitor Burn Rate Velocity**

    *   Watch for sudden spikes in token consumption.
    *   Adjust coding intensity based on remaining time.
    *   Plan big refactors around session resets.
2.  **Strategic Session Planning**

```bash
# Plan heavy usage around reset times
claude-monitor --reset-hour 9
```

    *   Schedule large tasks after resets.
    *   Use lighter tasks when approaching limits.
    *   Leverage multiple overlapping sessions.
3.  **Timezone Awareness**

```bash
# Always use your actual timezone
claude-monitor --timezone Europe/Warsaw
```

    *   Accurate reset time predictions.
    *   Better planning for work schedules.
    *   Correct session expiration estimates.

#### Optimization Tips

1.  **Terminal Setup**

    *   Use terminals with at least 80 character width.
    *   Enable color support for better visual feedback.
    *   Consider dedicated terminal window for monitoring.
    *   Use terminals with truecolor support for best theme experience.
2.  **Workflow Integration**

```bash
# Start monitoring with your development session (uv installation)
tmux new-session -d -s claude-monitor 'claude-monitor'

# Or development mode
tmux new-session -d -s claude-monitor './claude_monitor.py'

# Check status anytime
tmux attach -t claude-monitor
```

3.  **Multi-Session Strategy**

    *   Remember sessions last exactly 5 hours.
    *   You can have multiple overlapping sessions.
    *   Plan work across session boundaries.

#### Real-World Workflows

**Large Project Development**

```bash
# Setup for sustained development
claude-monitor --plan max20 --reset-hour 8 --timezone America/New_York
```

**Daily Routine**:

1.  **8:00 AM**: Fresh tokens, start major features.
2.  **10:00 AM**: Check burn rate, adjust intensity.
3.  **12:00 PM**: Monitor for afternoon session planning.
4.  **2:00 PM**: New session window, tackle complex problems.
5.  **4:00 PM**: Light tasks, prepare for evening session.

**Learning & Experimentation**

```bash
# Flexible setup for learning
claude-monitor --plan pro
```

**Sprint Development**

```bash
# High-intensity development setup
claude-monitor --plan max20 --reset-hour 6
```

---

## 🔧 Development Installation

For contributors and developers:

### Quick Start (Development/Testing)

```bash
# Clone the repository
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor

# Install in development mode
pip install -e .

# Run from source
python -m claude_monitor
```

### v3.0.0 Testing Features

The new version includes a comprehensive test suite:

*   **100+ test cases** with full coverage
*   **Unit tests** for all components
*   **Integration tests** for end-to-end workflows
*   **Performance tests** with benchmarking
*   **Mock objects** for isolated testing

```bash
# Run tests
cd src/
python -m pytest

# Run with coverage
python -m pytest --cov=claude_monitor --cov-report=html

# Run specific test modules
python -m pytest tests/test_analysis.py -v
```

### Prerequisites

1.  **Python 3.9+** installed.
2.  **Git** for cloning the repository.

### Virtual Environment Setup

#### Why Use Virtual Environment?

Using a virtual environment is **strongly recommended** because:

*   🛡️ **Isolation**: Keeps your system Python clean and prevents dependency conflicts.
*   📦 **Portability**: Easy to replicate the exact environment.
*   🔄 **Version Control**: Lock specific versions of dependencies.
*   🧹 **Clean Uninstall**: Simply delete the virtual environment folder.
*   👥 **Team Collaboration**: Everyone uses the same Python and package versions.

#### Installing virtualenv (if needed)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-venv

# Fedora/RHEL/CentOS
sudo dnf install python3-venv

# macOS (usually comes with Python)
# If not available, install Python via Homebrew:
brew install python3

# Windows (usually comes with Python)
# If not available, reinstall Python from python.org
# Make sure to check "Add Python to PATH" during installation
```

Alternatively, use the virtualenv package:

```bash
# Install virtualenv via pip
pip install virtualenv

# Then create virtual environment with:
virtualenv venv
# instead of: python3 -m venv venv
```

#### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor

# 2. Create virtual environment
python3 -m venv venv
# Or if using virtualenv package:
# virtualenv venv

# 3. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 4. Install Python dependencies
pip install -r requirements.txt # or pip install pytz rich>=13.0.0
# 5. Make script executable (Linux/Mac only)
chmod +x claude_monitor.py

# 6. Run the monitor
./claude_monitor.py
```

#### Daily Usage

```bash
# Navigate to project directory
cd Claude-Code-Usage-Monitor

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Run monitor
./claude_monitor.py  # Linux/Mac
# python claude_monitor.py  # Windows

# When done, deactivate
deactivate
```

#### Pro Tip: Shell Alias

```bash
# Add to ~/.bashrc or ~/.zshrc
alias claude-monitor='cd ~/Claude-Code-Usage-Monitor && source venv/bin/activate && ./claude_monitor.py'

# Then just run:
claude-monitor
```

---

## Troubleshooting

### Installation Issues

#### "externally-managed-environment" Error

On modern Linux distributions:
```
error: externally-managed-environment
× This environment is externally managed
```

**Solutions (in order of preference):**

1.  **Use uv (Recommended)**

```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then install with uv
uv tool install claude-monitor
```

2.  **Use pipx (Isolated Environment)**

```bash
# Install pipx
sudo apt install pipx  # Ubuntu/Debian
# or
python3 -m pip install --user pipx

# Install claude-monitor
pipx install claude-monitor
```

3.  **Use virtual environment**

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install claude-monitor
```

4.  **Force installation (Not Recommended)**

```bash
pip install --user claude-monitor --break-system-packages
```

⚠️ **Warning**: This bypasses system protection.

#### Command Not Found After pip Install

If claude-monitor command is not found: