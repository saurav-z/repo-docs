# ⏱️ Claude Code Usage Monitor: Stay on Top of Your Anthropic Tokens

**Effortlessly track your Claude AI token usage with real-time monitoring, intelligent predictions, and a beautiful terminal UI. Check it out on [GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).**

This powerful Python-based tool provides real-time insights into your Anthropic Claude AI token consumption, helping you stay within your limits and optimize your usage. Get advanced analytics, machine learning-based session limit predictions, and a rich terminal UI for a seamless monitoring experience.

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **Real-Time Monitoring:** Track token usage, burn rate, and cost analysis in real-time with configurable refresh rates.
*   **Advanced Analytics:** ML-based predictions, session limit detection, and multi-level alerts to avoid overspending.
*   **Rich Terminal UI:** Beautiful, color-coded progress bars, tables, and layouts with WCAG-compliant contrast.
*   **Smart Auto-Detection:** Automatic plan switching and intelligent limit discovery based on your usage.
*   **Custom Plan Support:**  Offers a 'custom' plan option leveraging machine learning to calculate personalized limits based on your last 8 days of usage.
*   **Configuration & Logging:**  Type-safe configuration, Pydantic validation, and comprehensive logging capabilities.

## Installation

Choose your preferred installation method:

### 1. Recommended: Modern Installation with `uv`

`uv` provides isolated environments automatically, avoids Python version issues, and simplifies updates.

#### Install

```bash
# Install from PyPI with uv (easiest)
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


### 2. Pip Installation

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

> **Important:**  For some Linux distributions, using a virtual environment or pipx is recommended to avoid "externally-managed-environment" errors.

### 3.  Pipx Installation (Isolated Environments)

```bash
pipx install claude-monitor
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```

### 4.  Conda/Mamba Installation

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```


## Usage

### Basic Usage

```bash
# Run with default settings (Custom plan)
claude-monitor

# For full name, short aliases
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# Exit gracefully
# Press Ctrl+C to exit the tool
```

### Configuration Options

Customize your monitoring experience with these command-line parameters:

*   `--plan`: Select your Anthropic plan ( `pro`, `max5`, `max20`, or `custom` - the default).
*   `--custom-limit-tokens`: Specify a custom token limit for the `custom` plan.
*   `--view`: Choose between `realtime`, `daily`, or `monthly` views.
*   `--timezone`: Set your timezone (e.g., `America/New_York`, `UTC`). Auto-detected by default.
*   `--time-format`: Specify `12h`, `24h`, or `auto` for time display.
*   `--theme`: Select `light`, `dark`, `classic`, or `auto` theme.
*   `--refresh-rate`: Set data refresh rate in seconds (1-60).
*   `--refresh-per-second`: Adjust display refresh rate in Hz (0.1-20.0).
*   `--reset-hour`: Set the daily reset hour (0-23).
*   `--log-level`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
*   `--log-file`: Specify a log file path.
*   `--debug`: Enable debug logging.
*   `--version, -v`: Show version information.
*   `--clear`: Clear saved configuration.

### Plan Options

| Plan        | Token Limit     | Best For                        |
| ----------- | --------------- | ------------------------------- |
| **custom** | P90 auto-detect | Intelligent limit detection     |
| **pro**     | ~19,000         | Claude Pro subscription        |
| **max5**    | ~88,000         | Claude Max5 subscription       |
| **max20**   | ~220,000        | Claude Max20 subscription      |


## What's New in v3.0.0

### Major Changes

*   **Complete Architecture Rewrite:** Improved modularity, testing, and maintainability.
*   **Enhanced Functionality:** ML-based limit detection and updated plan limits.
*   **New CLI Options:**  Flexible control over refresh rates, time format, and logging.
*   **Breaking Changes:** Package name change ( `claude-monitor` ) and updated minimum Python version.

## Need Help?

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## License

This project is licensed under the [MIT License](LICENSE).

## Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## Acknowledgments

Special thanks to our supporters!
[Ed] - *Buy Me Coffee Supporter*

---

<div align="center">
**⭐ Star this repo if you find it useful! ⭐**
[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)
</div>