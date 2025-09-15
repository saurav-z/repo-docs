# Claude Code Usage Monitor: Stay in Control of Your Claude AI Token Usage!

Tired of unexpected token overages? **Claude Code Usage Monitor** is a real-time terminal monitoring tool that provides advanced analytics, machine learning-based predictions, and a Rich UI to track your Claude AI token consumption. Get intelligent insights and proactive warnings to optimize your usage. 

[View the Original Repo](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## Key Features

*   **Real-time Monitoring:** Track token usage, burn rate, and cost in real-time with configurable refresh rates (0.1-20 Hz).
*   **ML-Powered Predictions:** Get intelligent session limit detection and P90 percentile calculations based on your usage.
*   **Advanced Rich UI:** Enjoy a beautiful terminal interface with color-coded progress bars, tables, and WCAG-compliant contrast for readability.
*   **Smart Auto-Detection:**  Automatically detect your current Claude plan and suggest optimal limits with custom limit discovery.
*   **Proactive Warnings:**  Receive multi-level alerts with cost and time predictions to avoid overspending.
*   **Comprehensive Plan Support:** Supports Pro, Max5, Max20, and Custom plans with updated token limits.
*   **Custom Plan Default:** The "Custom" plan analyzes your past 8 days of usage to calculate personalized limits for accurate predictions.

## Installation

### Modern Installation (Recommended): `uv`

The fastest and easiest method with automatic environment isolation.

```bash
# Install from PyPI with uv
uv tool install claude-monitor
claude-monitor
```

See original README for `uv` install instructions if needed, or use:
```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
### Other Installation Methods

*   **pip:**
    ```bash
    pip install claude-monitor
    claude-monitor
    ```
*   **pipx:**
    ```bash
    pipx install claude-monitor
    claude-monitor
    ```
*   **conda/mamba:**
    ```bash
    pip install claude-monitor
    claude-monitor
    ```

See original README for troubleshooting installation errors.

## Usage

### Basic Usage

Run the monitor with the default custom plan:

```bash
claude-monitor
```
Press `Ctrl+C` to exit gracefully.

### Configuration Options

Customize your monitoring experience using the command-line parameters.

*   **`--plan`**:  Choose your plan (pro, max5, max20, custom).
*   **`--custom-limit-tokens`**:  Set a custom token limit.
*   **`--view`**: Select the view (realtime, daily, monthly).
*   **`--timezone`**: Set your timezone (e.g., "America/New_York").
*   **`--time-format`**:  Choose time format (12h, 24h).
*   **`--theme`**: Select a theme (light, dark, classic, auto).
*   **`--refresh-rate`**: Data refresh rate in seconds (1-60).
*   **`--refresh-per-second`**: Display refresh rate in Hz (0.1-20.0).
*   **`--reset-hour`**: Set daily reset hour (0-23).
*   **`--log-level`**: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
*   **`--log-file`**: Set a log file path.
*   **`--debug`**: Enable debug logging.
*   **`--clear`**: Clear saved configuration.

### Example Usage
```bash
# Run with Pro Plan:
claude-monitor --plan pro

# Set a custom token limit:
claude-monitor --plan custom --custom-limit-tokens 50000

# Enable debug logging:
claude-monitor --debug

# View your daily usage:
claude-monitor --view daily
```
### Available Plans

| Plan        | Token Limit      |
|-------------|------------------|
| **custom**  | P90 auto-detect  |
| **pro**     | ~19,000          |
| **max5**    | ~88,000          |
| **max20**   | ~220,000         |

## What's New in v3.0.0

The latest version includes a complete architectural rewrite with major enhancements for improved performance, accuracy, and usability.

*   **Complete Architecture Rewrite:** Modular design, Pydantic validation, advanced error handling, and extensive testing.
*   **Enhanced Functionality:** P90 analysis, updated plan limits, cost analytics, and a richer terminal UI.
*   **New CLI Options:** Flexible control over refresh rates, time formats, and custom limits.
*   **Breaking Changes:** Package name changed, default plan now "custom."

## Development

Follow the instructions in the original README for development and testing setup.
## Troubleshooting

See the original README's troubleshooting section for detailed solutions to common installation and runtime issues.

## Contact

Reach out with any questions, suggestions, or for collaboration!

*   **Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## Additional Resources

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

A special thanks to Ed (Buy Me Coffee Supporter).

---

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)