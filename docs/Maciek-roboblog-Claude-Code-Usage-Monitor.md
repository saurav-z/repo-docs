# Claude Code Usage Monitor: Stay in Control of Your AI Token Usage ⏱️

**Tired of unexpected AI costs and session limits?** Claude Code Usage Monitor provides real-time terminal monitoring, advanced analytics, and predictive insights for your Claude AI token consumption.  [View the original repo on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **Real-time Monitoring:** Track token usage, burn rate, and cost analysis in a beautiful terminal UI.
*   **ML-Based Predictions:** Get intelligent session limit detection and usage forecasts.
*   **Smart Auto-Detection:**  Automatic plan switching and custom limit discovery for optimal usage.
*   **Custom Plan Defaults**: Now uses custom session limits to adapt to your usage patterns
*   **Rich UI:** Visualize your data with color-coded progress bars, tables, and WCAG-compliant themes.
*   **Cost Analytics:** Model-specific pricing and cache token calculations.
*   **Flexible Installation:** Supports `uv`, `pip`, `pipx`, and `conda/mamba`.

## Installation

Choose your preferred installation method.  **`uv` is highly recommended!**

### ⚡ Modern Installation with `uv` (Recommended)

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install claude-monitor
uv tool install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor

#  If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run
claude-monitor
```

### 🛠️ Other Package Managers
Refer to the original README for pipx and conda/mamba instructions.

## Usage

### Command Line Options

```bash
# Show help information
claude-monitor --help
```

| Parameter            | Type   | Default  | Description                                                     |
| -------------------- | ------ | -------- | --------------------------------------------------------------- |
| `--plan`             | string | `custom`  | Plan type: `pro`, `max5`, `max20`, or `custom`                  |
| `--custom-limit-tokens` | int    | `None`   | Token limit for custom plan (must be > 0)                     |
| `--view`             | string | `realtime`| View type: `realtime`, `daily`, or `monthly`                     |
| `--timezone`         | string | `auto`   | Timezone (auto-detected). Examples: `UTC`, `America/New_York`, `Europe/London` |
| `--time-format`      | string | `auto`   | Time format: `12h`, `24h`, or `auto`                            |
| `--theme`            | string | `auto`   | Display theme: `light`, `dark`, `classic`, or `auto`              |
| `--refresh-rate`     | int    | `10`      | Data refresh rate in seconds (1-60)                             |
| `--refresh-per-second`| float  | `0.75`   | Display refresh rate in Hz (0.1-20.0)                           |
| `--reset-hour`       | int    | `None`   | Daily reset hour (0-23)                                         |
| `--log-level`        | string | `INFO`   | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`     |
| `--log-file`         | path   | `None`   | Log file path                                                  |
| `--debug`            | flag   | `False`  | Enable debug logging                                           |
| `--version`, `-v`    | flag   | `False`  | Show version information                                        |
| `--clear`            | flag   | `False`  | Clear saved configuration                                       |

### Example Usage

```bash
# Default (Custom plan with auto-detection)
claude-monitor

#Alternative commands
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# Pro Plan
claude-monitor --plan pro

# Set Timezone
claude-monitor --timezone "America/Los_Angeles"

# View Daily Usage
claude-monitor --view daily
```

## Available Plans

| Plan       | Token Limit    | Best For                         |
| ----------- | ------------- | ------------------------------- |
| **custom** | P90 auto-detect | Intelligent limit detection (default) |
| **pro**    | ~19,000        | Claude Pro subscription         |
| **max5**   | ~88,000        | Claude Max5 subscription        |
| **max20**  | ~220,000       | Claude Max20 subscription       |

## Development

See the original README for development setup, testing, and contribution guidelines.

## Troubleshooting

See the original README for installation and runtime troubleshooting.

## Contact

*   **Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## Additional Information

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

A special thanks to our supporters who help keep this project going:
*   **Ed** - *Buy Me Coffee Supporter*

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>