# 🚀 Claude Code Usage Monitor: Stay Ahead of Your Claude AI Token Limits

**Tired of hitting those Claude AI token limits unexpectedly?**  Get real-time insights and smart predictions with the **Claude Code Usage Monitor**, a powerful terminal tool to track and optimize your Claude AI usage.  [Check it out on GitHub!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

The Claude Code Usage Monitor is a beautiful, real-time terminal monitoring tool that gives you the edge by helping you understand and manage your Claude AI token consumption.  It features advanced analytics, machine learning-based predictions, and a rich, user-friendly interface to keep you in control.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features:

*   **Real-Time Monitoring**: Monitor token usage, burn rate, and cost in real-time with configurable refresh rates.
*   **ML-Powered Predictions**: Get intelligent session limit predictions and warnings.
*   **Advanced Rich UI**: Beautiful, color-coded progress bars, tables, and layouts with WCAG-compliant contrast, making it easy to see your token usage at a glance.
*   **Smart Auto-Detection**: Automatic plan switching and custom limit discovery.
*   **Cost Analytics**: Model-specific pricing and cache token calculations to understand your spending.
*   **Customizable**: Configure refresh rates, timezones, themes, and logging levels.
*   **Easy Installation**: Available through `uv`, `pip`, `pipx`, and `conda`.

## Installation

Choose your preferred installation method:

### 🚀 Modern Installation with `uv` (Recommended)

`uv` is a blazing-fast Python package and virtual environment manager.  It is the easiest way to install and use the monitor, and also has a small footprint.

```bash
# Install with uv
uv tool install claude-monitor

# Run
claude-monitor
```

For first-time `uv` users:

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run
claude-monitor
```

### 🛠️ Other Package Managers

#### `pipx` (Isolated Environments)

```bash
pipx install claude-monitor

# Run
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```

#### `conda/mamba`

```bash
pip install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📖 Usage

Get started quickly:

```bash
# Show help information
claude-monitor --help

# Run with custom settings
claude-monitor --plan pro --theme dark --timezone "America/New_York"
```

**Available Command-Line Parameters and Plan Options:**
See original README.

### 🚀 What's New in v3.0.0

[See original README for all the details on the rewrite and enhanced features.]

### 📝 License

[MIT License](LICENSE) - feel free to use and modify as needed.

### 🙏 Acknowledgments

See original README for more info.

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>