# 📊 Claude Code Usage Monitor: Real-Time Token Tracking & AI-Powered Insights

**Stay in control of your Claude AI usage!** The Claude Code Usage Monitor provides real-time, beautiful terminal monitoring with advanced analytics, machine learning-based predictions, and a user-friendly Rich UI. 
[Check out the project on GitHub!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## Key Features

*   **✨ Real-time Monitoring:** Track token consumption, burn rate, and costs with configurable refresh rates.
*   **🔮 AI-Powered Predictions:** Get intelligent session limit detection and cost projections using Machine Learning.
*   **📊 Advanced Rich UI:** Enjoy a beautiful, color-coded interface with progress bars, tables, and WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **🚀 Comprehensive Plan Support:** Monitor usage for Pro, Max5, Max20, and a custom P90-based plan.

---

## Installation

Choose the installation method that suits your needs:

*   **⭐ Recommended: uv (Modern Installation)**

    `uv tool install claude-monitor`
    `claude-monitor` (or use short aliases like `cmonitor`, `ccmonitor`, or `ccm`)

    *uv* offers isolated environments automatically, removing potential conflicts, simplifies updates, and supports all platforms. See [uv installation instructions](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor#first-time-uv-users) if you haven't installed it yet.

*   **📦 pip**

    `pip install claude-monitor`

    Then, run the monitor:
    `claude-monitor` (or use short aliases)

    *Make sure* your PATH includes `~/.local/bin` if the `claude-monitor` command isn't found (see the original README for instructions).

*   **🛠️ Other Package Managers:**  Support for pipx, conda/mamba is also available.

---

## Usage

*   **Get Help:**
    `claude-monitor --help`  (Shows all available command-line parameters.)

*   **Basic Usage:**

    Just run:
    `claude-monitor`

*   **Configuration Options:**

    | Option          | Description                                    | Default   |
    | --------------- | ---------------------------------------------- | --------- |
    | `--plan`        | Plan type: pro, max5, max20, or custom         | custom    |
    | `--view`        | View type: realtime, daily, monthly             | realtime  |
    | `--timezone`    | Timezone (e.g., UTC, America/New_York)         | auto      |
    | `--time-format` | Time format: 12h, 24h, or auto                 | auto      |
    | `--theme`       | Display theme: light, dark, classic, or auto | auto      |
    | `--refresh-rate`| Data refresh rate in seconds (1-60)         | 10        |

    You can save your preferred settings. Example:
    `claude-monitor --plan pro --theme dark --timezone "America/New_York"`

    Subsequent runs:
    `claude-monitor --plan pro` (restores saved preferences)
    `claude-monitor --clear` (clears saved preferences)

---

## Default Custom Plan

The **Custom plan** (the default) is tailored for Claude Code 5-hour sessions and monitors:

*   **Token usage**
*   **Messages usage**
*   **Cost usage**

It adapts to your usage patterns, analyzing the last 8 days of sessions to calculate personalized limits, ensuring accurate predictions and warnings.

---

## v3.0.0 - What's New

*   **Complete Architecture Rewrite:** Modular design, Pydantic-based configuration, comprehensive testing.
*   **P90 Analysis:** Machine learning-based limit detection using 90th percentile calculations.
*   **Updated Plan Limits:** Pro (19,000 tokens), Max5 (88,000 tokens), Max20 (220,000 tokens)
*   **Rich UI Enhancements:** Customizable display refresh rates.

---

## Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) for details.

---

## Contact

**📧 Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

---
```
Key improvements in this version:

*   **SEO Optimization:**  Added a one-sentence hook. Improved the structure of headers and descriptions. Included key search terms like "Claude AI", "token tracking", and "real-time monitoring".
*   **Concise and Clear:** Simplified the language and focused on the most important information for users.
*   **Emphasis on Key Features:** Highlighted the most valuable features using bullet points.
*   **Installation Guidance:**  Prioritized the *uv* installation method as the easiest and most recommended and clearly explained the alternatives.
*   **Simplified Usage Section:**  Provided simple examples and a clear table of options.
*   **Call to Action:** Included a "Check out the project on GitHub!" link at the top.
*   **Removed unnecessary information:** Removed redundant information and some of the less important original information.