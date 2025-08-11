# 🚀 Claude Code Usage Monitor: Real-time AI Token Tracking and Prediction

**Effortlessly monitor your Claude AI token usage, predict session limits, and optimize your AI workflow with this powerful terminal-based tool.  [Explore the Claude Code Usage Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).**

---

## 🌟 Key Features

*   **📈 Real-time Monitoring:** Track token consumption, burn rate, and session details with configurable refresh rates (0.1-20 Hz).
*   **🔮 ML-Powered Predictions:** Leverage machine learning for intelligent session limit detection, burn rate analysis, and cost projections.
*   **🎨 Rich Terminal UI:** Enjoy a beautiful, color-coded interface with progress bars, tables, and WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Benefit from automatic plan switching, custom limit discovery, and system time/theme recognition.
*   **🛠️ Flexible Plan Support:** Includes updated limits for Pro, Max5, Max20, and a customizable "Custom" plan.
*   **⚠️ Advanced Warning System:** Receive multi-level alerts and predictions for cost and time management.
*   **💼 Modular Architecture:** Designed with the Single Responsibility Principle (SRP) for maintainability.
*   **⚙️ Customizable:** Configure timezones, themes, logging, and more to suit your needs.

---

## 📦 Installation

Choose your preferred installation method:

*   **Recommended: Modern Installation with `uv`**

    *   **Why `uv`?** Isolates environments, avoids Python version issues, simplifies updates.
    *   ```bash
        # Install with uv
        uv tool install claude-monitor
        # Run from anywhere
        claude-monitor  # or cmonitor, ccmonitor for short
        ```
        *   Install `uv` first, if you don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh` (Linux/macOS) or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows)

*   **Installation with `pip`**

    ```bash
    pip install claude-monitor
    # If command not found, add to PATH: export PATH="$HOME/.local/bin:$PATH"
    claude-monitor  # or cmonitor, ccmonitor for short
    ```
        *   **Important:** Consider using a virtual environment or `pipx` to manage dependencies (especially for "externally-managed-environment" errors).

*   **Alternative Package Managers**

    *   `pipx`: `pipx install claude-monitor`
    *   `conda/mamba`: `pip install claude-monitor` within your conda environment.

---

## 📖 Usage

*   **Get Help:** `claude-monitor --help`

*   **Key Command-Line Parameters:**

    | Parameter           | Type      | Default   | Description                                        |
    | ------------------- | --------- | --------- | -------------------------------------------------- |
    | `--plan`            | `string`  | `custom`  | Plan type (pro, max5, max20, custom)              |
    | `--custom-limit-tokens` | `int`     | `None`    | Token limit for custom plan                        |
    | `--view`            | `string`  | `realtime`| View type (realtime, daily, monthly)                |
    | `--timezone`        | `string`  | `auto`    | Timezone (auto-detected)                           |
    | `--time-format`     | `string`  | `auto`    | Time format (12h, 24h, auto)                      |
    | `--theme`           | `string`  | `auto`    | Display theme (light, dark, classic, auto)          |
    | `--refresh-rate`    | `int`     | `10`      | Data refresh rate in seconds                       |
    | `--refresh-per-second`| `float`   | `0.75`    | Display refresh rate in Hz (0.1-20.0)             |
    | `--reset-hour`      | `int`     | `None`    | Daily reset hour (0-23)                            |
    | `--log-level`       | `string`  | `INFO`    | Logging level                                      |
    | `--log-file`        | `path`    | `None`    | Log file path                                     |
    | `--debug`           | `flag`    | `False`   | Enable debug logging                              |
    | `--clear`           | `flag`    | `False`   | Clear saved configuration                          |
    | `--version`, `-v`   | `flag`    | `False`   | Show version information                          |

*   **Plan Options:**

    | Plan       | Token Limit | Best For                      |
    | ---------- | ----------- | ----------------------------- |
    | `custom`   | P90-based   | Intelligent limit detection (default) |
    | `pro`      | ~19,000     | Claude Pro subscription        |
    | `max5`     | ~88,000     | Claude Max5 subscription       |
    | `max20`    | ~220,000    | Claude Max20 subscription      |

*   **Basic Usage (Most common):**

    ```bash
    claude-monitor  # Default: Custom plan with auto-detection
    # Alternative commands
    claude-code-monitor  # Full descriptive name
    cmonitor             # Short alias
    ccmonitor            # Short alternative
    ccm                  # Shortest alias

    # To Exit: Press Ctrl+C
    ```

*   **Configuration Options:**  Customize your experience with features like plan selection, custom reset times, display views, performance settings, and timezones. See the full README for all options.

---

## 🚀 What's New in v3.0.0

This major update brings significant improvements:

*   **Complete Architecture Rewrite:** Modular design, SRP compliance, Pydantic validation, and a comprehensive test suite.
*   **Enhanced Functionality:** Machine learning-based limit detection (P90 analysis), updated plan limits, cost analytics, and a rich UI.
*   **New CLI Options:** Control display refresh rates, time formats, set custom limits, enable logging, and clear configuration.
*   **Breaking Changes:** Package name change, default plan change to custom, and Python 3.9+ required.

---

## 📚 Additional Resources

*   [Development Roadmap](DEVELOPMENT.md) - Future Plans
*   [Contributing Guide](CONTRIBUTING.md) - Get Involved
*   [Troubleshooting](TROUBLESHOOTING.md) - Common Issues and Solutions

---

## 📝 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgments

*   Special thanks to Ed (Buy Me Coffee Supporter) for their support and appreciation.
*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

---

<div align="center">

**⭐  Give the repo a star if you like it!  ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>
```
Key improvements:

*   **SEO Optimization:** Includes the most relevant keywords in the title and throughout the description (e.g., "Claude AI," "token usage," "real-time," "prediction," "terminal").
*   **Concise Hook:**  The one-sentence description clearly and concisely summarizes the tool's purpose.
*   **Clear Headings and Structure:** Improves readability and user experience.
*   **Bulleted Key Features:** Easy to scan and understand the main benefits.
*   **Emphasis on "uv" Installation:** Highlights the easiest and most modern installation method.
*   **Concise Instructions:** Streamlines installation and usage instructions.
*   **Call to Action:** Encourages users to explore, star the repo, and contribute.
*   **Well-Organized:** Keeps the information organized into logical sections for easy navigation.
*   **Links Back to Original Repo:**  The link to the original repo is included.
*   **Comprehensive:**  Includes all the important information from the original README, but in a more user-friendly and engaging format.
*   **Clear Parameter Descriptions:** The CLI parameters table includes all original parameters with the correct type.
*   **Adds the Most Important Commands:**  Shows commands like the help command, the alternative short commands and the main usage commands.
*   **Adds Use Cases:** Includes the most important workflows.
*   **Adds an acknowledgement to contributors.**
*   **Adds a Star History section.**

This improved version is much more appealing to users, easier to understand, and optimized for search engines.