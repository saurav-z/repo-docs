# Claude Code Usage Monitor: Track, Analyze, and Optimize Your Claude AI Token Usage

**Are you tired of guessing how much you're spending on Claude AI?** Claude Code Usage Monitor is your real-time terminal companion, providing in-depth token tracking, intelligent predictions, and cost analysis. Stay in control of your AI usage and optimize your workflow!

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

[**View the source on GitHub**](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

## Key Features

*   **Real-Time Monitoring:** Track token usage, burn rate, and cost with configurable refresh rates.
*   **ML-Powered Predictions:** Get intelligent session limit detection and P90 percentile calculations.
*   **Advanced UI:** Experience a beautiful, color-coded terminal UI with WCAG-compliant contrast.
*   **Smart Auto-Detection:** Automatic plan switching with custom limit discovery to maximize efficiency.
*   **Cost Analytics:** Model-specific pricing and cache token calculations.

## Table of Contents

*   [Installation](#installation)
    *   [Recommended: Installation with `uv`](#recommended-installation-with-uv)
    *   [Installation with `pip`](#installation-with-pip)
    *   [Other Package Managers](#other-package-managers)
*   [Usage](#usage)
    *   [Basic Usage](#basic-usage)
    *   [Configuration Options](#configuration-options)
    *   [Available Plans](#available-plans)
*   [What's New in v3.0.0](#whats-new-in-v300)
*   [Development Installation](#development-installation)
*   [Troubleshooting](#troubleshooting)
*   [Contact](#contact)
*   [License](#license)
*   [Contributors](#contributors)
*   [Acknowledgments](#acknowledgments)

## Installation

### Recommended: Installation with `uv`

`uv` is the fastest and easiest way to install and use the monitor.

```bash
# Install uv (if you don't have it)
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install claude-monitor from PyPI
uv tool install claude-monitor

# Run
claude-monitor
```

### Installation with `pip`

```bash
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run
claude-monitor
```

### Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within your conda environment)

## Usage

### Basic Usage

```bash
# Run with default settings (Custom plan)
claude-monitor
```

### Configuration Options

*   `--plan`:  `pro`, `max5`, `max20`, or `custom` (default).
*   `--custom-limit-tokens`:  Token limit for the custom plan.
*   `--view`:  `realtime` (default), `daily`, or `monthly`.
*   `--timezone`:  Timezone (e.g., `UTC`, `America/New_York`).  Auto-detected by default.
*   `--time-format`:  `12h`, `24h`, or `auto`.
*   `--theme`:  `light`, `dark`, `classic`, or `auto`.
*   `--refresh-rate`: Refresh rate in seconds (1-60).
*   `--refresh-per-second`: Display refresh rate in Hz (0.1-20.0).
*   `--reset-hour`:  Daily reset hour (0-23).
*   `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   `--log-file`: Log file path.
*   `--debug`: Enable debug logging.
*   `--clear`: Clear saved configuration.

**Example Usage:**

```bash
# Monitor with pro plan and dark theme
claude-monitor --plan pro --theme dark

#  Set custom reset time
claude-monitor --reset-hour 3

# Clear settings
claude-monitor --clear
```

### Available Plans

| Plan        | Token Limit | Cost Limit       | Best For                         |
| ----------- | ----------- | ---------------- | -------------------------------- |
| **custom**  | P90-based   | (default) $50.00 | Intelligent limit detection      |
| **pro**     | ~19,000     | $18.00           | Claude Pro subscription          |
| **max5**    | ~88,000     | $35.00           | Claude Max5 subscription         |
| **max20**   | ~220,000    | $140.00          | Claude Max20 subscription        |

## What's New in v3.0.0

*   **Complete Architecture Rewrite**: Modular design, SRP compliance, Pydantic validation, advanced error handling.
*   **Enhanced Functionality**: ML-based limit detection using 90th percentile calculations, updated plan limits, cost analytics, and a rich terminal UI.
*   **New CLI Options**: Fine-grained control over refresh rates, time format, logging, and saved settings.
*   **Breaking Changes**: Package name changed, default plan updated, Python 3.9+ required.

## Development Installation

For contributing or development:

```bash
# Clone the repository
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor

# Install in development mode
pip install -e .

# Run
python -m claude_monitor
```

## Troubleshooting

See detailed troubleshooting steps in the [Troubleshooting](#troubleshooting) section above.

## Contact

For questions, suggestions, or collaboration:

*   **Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## License

[MIT License](LICENSE)

## Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## Acknowledgments

Thanks to our supporters!

*   **Ed** - *Buy Me Coffee Supporter*

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>