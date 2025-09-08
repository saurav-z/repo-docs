# 🚀 Claude Code Usage Monitor: Real-Time AI Token Tracking & Analysis

**Effortlessly monitor your Anthropic Claude AI token usage with this powerful terminal tool, complete with advanced analytics and smart predictions.** 

[View the original repository on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **Real-time Monitoring**: Track token consumption, burn rate, and cost in real-time with configurable refresh rates.
*   **ML-Powered Predictions**: Intelligent session limit detection using machine learning (P90 percentile analysis).
*   **Advanced UI**: Rich, color-coded terminal interface with WCAG-compliant contrast for readability.
*   **Smart Auto-Detection**: Automatic plan switching and custom limit discovery for optimal usage.
*   **Comprehensive Plan Support**: Supports Claude Pro, Max5, Max20, and Custom plans.
*   **Cost Analysis**: Model-specific pricing and cache token calculations.
*   **Advanced Warnings**: Multi-level alerts and time predictions to prevent session overruns.
*   **Customizable**: Configure refresh rates, time zones, themes, and logging.

## Installation

### 1. Modern Installation with `uv` (Recommended)

`uv` provides an easy way to install and manage project dependencies, and provides a clean install without conflicts.

```bash
# Install uv (Linux/macOS):
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install uv (Windows):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install from PyPI with uv:
uv tool install claude-monitor

# Run from anywhere:
claude-monitor  # or cmonitor, ccmonitor for short
```

### 2. Installation with `pip`

```bash
# Install from PyPI
pip install claude-monitor

# If 'claude-monitor' command is not found, add to PATH:
#   (Add this to your .bashrc or .zshrc and restart your terminal)
#   export PATH="$HOME/.local/bin:$PATH"

# Run:
claude-monitor  # or cmonitor, ccmonitor for short
```

### 3. Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within a conda environment)

## Usage

### Basic Usage

*   Simply run `claude-monitor` to start monitoring with default settings (Custom plan).

*   Use `claude-code-monitor`, `cmonitor`, `ccmonitor`, or `ccm` for shorter commands.

*   Press `Ctrl+C` to gracefully exit.

### Command Line Options

Use the following parameters to customize your monitoring:

| Parameter                | Type     | Default     | Description                                                                                                               |
| :----------------------- | :------- | :---------- | :------------------------------------------------------------------------------------------------------------------------ |
| `--plan`                 | string   | `custom`    | Choose a plan: `pro`, `max5`, `max20`, or `custom`                                                                         |
| `--custom-limit-tokens`  | integer  | `None`      | Specify a token limit for the custom plan.                                                                                |
| `--view`                 | string   | `realtime`  | Choose a view: `realtime`, `daily`, or `monthly`.                                                                         |
| `--timezone`             | string   | `auto`      | Set a timezone (e.g., `UTC`, `America/New_York`).                                                                           |
| `--time-format`          | string   | `auto`      | Choose time format: `12h`, `24h`, or `auto`.                                                                                |
| `--theme`                | string   | `auto`      | Set a display theme: `light`, `dark`, `classic`, or `auto`.                                                                 |
| `--refresh-rate`         | integer  | `10`        | Data refresh rate in seconds (1-60).                                                                                        |
| `--refresh-per-second`   | float    | `0.75`      | Display refresh rate in Hz (0.1-20.0).                                                                                      |
| `--reset-hour`           | integer  | `None`      | Set the daily reset hour (0-23).                                                                                          |
| `--log-level`            | string   | `INFO`      | Set log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.                                                           |
| `--log-file`             | path     | `None`      | Specify a log file path.                                                                                                  |
| `--debug`                | flag     | `False`     | Enable debug logging.                                                                                                     |
| `--version`, `-v`        | flag     | `False`     | Show version information.                                                                                                 |
| `--clear`                | flag     | `False`     | Clear saved configuration.                                                                                                |

### Key Features in Detail

*   **Plan Selection:**  Choose between `pro`, `max5`, `max20` and the default `custom` plan for intelligent limit detection based on your usage patterns, calculated using the 90th percentile.
*   **Custom Reset Times:** Configure daily reset times using `--reset-hour` to align with your work schedule.
*   **Usage Views:** Select different views (realtime, daily, monthly) for comprehensive analysis.
*   **Performance & Display:** Adjust the refresh rate and display refresh rate for optimal performance. Customize the time format and theme to your preferences.
*   **Timezone Configuration:** Adjust timezones using the `--timezone` parameter to reflect your local time zone.
*   **Logging & Debugging:** Utilize the `--debug`, `--log-file`, and `--log-level` options for comprehensive debugging.

## Troubleshooting

*   **"externally-managed-environment" error**: Use `uv`, `pipx`, or a virtual environment as described above.

*   **Command Not Found**:  Ensure the installation directory is in your `PATH` or use the full path to the script.

*   **No active session found**: Send at least two messages in Claude Code. Consider specifying a custom configuration path: `CLAUDE_CONFIG_DIR=~/.config/claude ./claude_monitor.py`

## Further Information

*   **[Development Roadmap](DEVELOPMENT.md)** - Future ML features, PyPI, Docker plans.
*   **[Contributing Guide](CONTRIBUTING.md)** - Instructions for contributing to the project.
*   **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions.

## Contact

For questions, suggestions, or to collaborate, contact [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

*   Special thanks to our sponsors.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">
  **⭐ Star this repo if you find it helpful! ⭐**
  <br>
  [Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)
</div>
```
Key improvements and optimization:

*   **SEO Optimization**:  The title and introduction now include relevant keywords ("Claude", "AI", "token usage", "monitoring", "real-time", "analysis") to improve search engine visibility.
*   **Concise & Engaging Hook**: The first sentence serves as a strong hook, immediately conveying the tool's purpose and value.
*   **Clear Headings**:  Uses headings and subheadings to improve readability and organization.
*   **Bulleted Key Features**:  Highlights key features using bullet points for easy scanning.  Features are reworded to emphasize benefits.
*   **Actionable Installation Instructions**: Streamlined installation, with the "uv" install as the primary suggestion due to its advantages.  Included clearer notes on what to do if you have path issues.
*   **Clear Usage Examples**:  Provides practical usage scenarios with specific commands and benefits.
*   **Complete Command Line Options**: Clearly outlines the available command-line parameters.
*   **Troubleshooting Section**:  A dedicated troubleshooting section addresses common installation and runtime issues with explicit solutions.
*   **Improved Structure**:  Organized content logically, using spacing and formatting to enhance readability.  The sections on "What's New" and the architecture are removed to reduce length.
*   **Concise Summarization**:  Removed redundant information to keep the README focused and concise.
*   **Call to Action**:  Encourages users to star the repository.
*   **Maintainability**: The improved structure makes it easier to update and maintain the README.
*   **Removed Unnecessary Details**: Eliminated some of the less critical details to improve the focus on the main features.
*   **Contact Section**:  Directs users to contact you with a clear email.