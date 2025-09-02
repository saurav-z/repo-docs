# 🤖 Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

**Effortlessly monitor and optimize your Anthropic Claude AI token usage with the `Claude Code Usage Monitor`.** [Check out the original repo here](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

This powerful, real-time terminal tool provides in-depth token consumption analysis, burn rate tracking, cost estimations, and AI-driven predictions for your Anthropic Claude sessions. Get a clear understanding of your usage and optimize your workflow!

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## Key Features

*   **Real-time Monitoring:** Dynamic token usage, burn rate, and cost tracking.
*   **AI-Powered Predictions:** Machine learning-based session limit estimations.
*   **Advanced UI:** Rich, customizable terminal UI with color-coded progress bars and tables.
*   **Smart Auto-Detection:** Intelligent plan switching & custom limit discovery.
*   **Plan Support:** Includes Pro, Max5, Max20, and Custom plans.
*   **Cost Analytics:** Model-specific pricing and cache token calculations.
*   **Configuration & Logging:** Extensive options for logging, themes, and timezones.

### Why Use the Claude Code Usage Monitor?

*   **Stay Within Budget:** Track your spending and avoid unexpected charges.
*   **Optimize Usage:** Understand your burn rate and adjust your prompts for efficiency.
*   **Plan Your Sessions:** Get accurate predictions for session limits and expiration.
*   **Gain Insights:** Access detailed analytics and reports for informed decision-making.

## 🚀 Installation

### Recommended: Modern Installation with `uv`

`uv` is a fast and reliable package and dependency manager.

1.  **Install `uv` (Linux/macOS):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    **(Windows):**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Install the Monitor:**
    ```bash
    uv tool install claude-monitor  # Fastest and easiest installation
    ```

3.  **Run from Anywhere:**
    ```bash
    claude-monitor  # or cmonitor, ccmonitor for short
    ```

### Alternative: Installation with `pip`

```bash
pip install claude-monitor
# If command not found:
# Add this to your .bashrc or .zshrc:
# export PATH="$HOME/.local/bin:$PATH"
# source ~/.bashrc  or restart your terminal
claude-monitor  # or cmonitor, ccmonitor for short
```

### Other Installation Methods

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within your conda environment)

## 📖 Usage

### Basic Commands
```bash
claude-monitor  # or cmonitor, ccmonitor, ccm for short (run with default settings)
claude-monitor --help # get help
```

### Customization

*   **Plan Selection:**

    ```bash
    claude-monitor --plan pro    # For Claude Pro users
    claude-monitor --plan max5   # For Claude Max5 users
    claude-monitor --plan max20  # For Claude Max20 users
    claude-monitor --plan custom # For the Custom plan (with auto-detection)
    ```

*   **View Configuration:**
    ```bash
    claude-monitor --view realtime # (Default) Live monitoring
    claude-monitor --view daily    # Daily token usage
    claude-monitor --view monthly  # Monthly token usage
    ```

*   **Timezone and Formatting:**
    ```bash
    claude-monitor --timezone "America/New_York" # Set your timezone
    claude-monitor --time-format 24h            # Use 24-hour format
    ```

*   **Logging and Debugging:**
    ```bash
    claude-monitor --debug         # Enable debug logging
    claude-monitor --log-file /path/to/log.txt # Log to a file
    claude-monitor --log-level DEBUG # Set the log level
    ```

*   **Clear saved configurations:**

    ```bash
    claude-monitor --clear
    ```

### Additional Parameters

| Parameter               | Type    | Default     | Description                                                                         |
| ----------------------- | ------- | ----------- | ----------------------------------------------------------------------------------- |
| `--plan`                | string  | `custom`    | Plan type: `pro`, `max5`, `max20`, or `custom`                                     |
| `--custom-limit-tokens` | integer | `None`      | Token limit for custom plan (must be > 0)                                             |
| `--view`                | string  | `realtime`  | View type: `realtime`, `daily`, or `monthly`                                          |
| `--timezone`            | string  | `auto`      | Timezone (auto-detected). Examples: `UTC`, `America/New_York`, `Europe/London`       |
| `--time-format`         | string  | `auto`      | Time format: `12h`, `24h`, or `auto`                                              |
| `--theme`               | string  | `auto`      | Display theme: `light`, `dark`, `classic`, or `auto`                                |
| `--refresh-rate`        | integer | `10`        | Data refresh rate in seconds (1-60)                                                    |
| `--refresh-per-second`  | float   | `0.75`      | Display refresh rate in Hz (0.1-20.0)                                                  |
| `--reset-hour`          | integer | `None`      | Daily reset hour (0-23)                                                                |
| `--log-level`           | string  | `INFO`      | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                        |
| `--log-file`            | path    | `None`      | Log file path                                                                       |
| `--debug`               | flag    | `False`     | Enable debug logging                                                                 |
| `--version, -v`         | flag    | `False`     | Show version information                                                              |
| `--clear`               | flag    | `False`     | Clear saved configuration                                                              |

### Saved Configuration

The monitor saves your most recent configuration to your `~/.claude-monitor/last_used.json` file, so that you don't have to re-enter all the settings every time. The following settings will be saved:
*   View type (--view)
*   Theme preferences (--theme)
*   Timezone settings (--timezone)
*   Time format (--time-format)
*   Refresh rates (--refresh-rate, --refresh-per-second)
*   Reset hour (--reset-hour)
*   Custom token limits (--custom-limit-tokens)

## 🚀 What's New in v3.0.0

This major update brings a complete architecture rewrite for improved performance, enhanced features, and a more user-friendly experience:

*   **Complete Architecture Rewrite:** Modular design, Pydantic-based configuration, and extensive testing.
*   **Enhanced Functionality:** ML-based limit detection, updated plan limits, cost analytics, and a rich UI.
*   **New CLI Options:** Improved control over refresh rates, logging, and configuration.
*   **Breaking Changes:** Default plan changed to `custom`, and the minimum Python version is 3.9+.

## 🔧 Development

For development and contributions, see the [Development Installation](#-development-installation) section.

## 🙏 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

For questions, suggestions, or collaboration, contact: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>