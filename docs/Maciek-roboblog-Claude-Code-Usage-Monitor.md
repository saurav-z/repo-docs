# 🤖 Claude Code Usage Monitor: Stay Ahead of Your Token Limits!

**Track and optimize your Anthropic Claude AI token usage with this powerful terminal monitor!**  [View the original repository](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

---

## 🚀 Key Features

*   📊 **Real-time Monitoring:**  Track token consumption, burn rate, cost analysis, and more with configurable refresh rates for up-to-the-second insights.
*   🔮 **ML-Powered Predictions:** Intelligent session limit detection leveraging machine learning and P90 percentile calculations to estimate remaining tokens accurately.
*   🎨 **Rich Terminal UI:** Beautiful, color-coded progress bars, data tables, and layouts enhanced with WCAG-compliant contrast for optimal readability.
*   🤖 **Smart Auto-Detection:** Automatic plan switching with custom limit discovery, allowing seamless adaptation to your Claude subscription tier.
*   ✅ **Multiple Usage Views:** Switch between Realtime, Daily, and Monthly views to analyze consumption patterns effectively and make informed decisions.
*   📦 **Comprehensive Analytics:** Includes Model-specific pricing, cache token calculations, cost predictions, and daily/monthly usage aggregation.
*   ⚙️ **Configuration & Customization:** Supports custom reset times, timezones, themes, and more, with CLI arguments that always override saved settings.
*   💻 **Easy Installation:** Supports installation via `uv`, `pip`, `pipx`, and `conda/mamba`.

---

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` provides a streamlined installation experience with isolated environments and no Python version conflicts.

```bash
# Install with uv (if you don't have it, install with the command below)
uv tool install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

**First-time `uv` Users:**

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

*   **PATH Setup:**  If you encounter a warning about the script location, add  `export PATH="$HOME/.local/bin:$PATH"` to your shell's configuration file (e.g., `~/.bashrc` or `~/.zshrc`) and reload the shell.
*   **"externally-managed-environment" error:** For modern Linux,  use `uv`, a virtual environment, or `pipx`.  See Troubleshooting.

### 🛠️ Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` within a conda environment.

---

## 📖 Usage

### ⚙️ Command-Line Parameters

| Parameter          | Type       | Default | Description                                                                   |
| ------------------ | ---------- | ------- | ----------------------------------------------------------------------------- |
| `--plan`           | string     | `custom`  | Plan type: `pro`, `max5`, `max20`, or `custom`                               |
| `--custom-limit-tokens` | integer    | -       | Token limit for the `custom` plan.                                    |
| `--view`           | string     | `realtime`  | View type: `realtime`, `daily`, or `monthly`                                     |
| `--timezone`       | string     | `auto`    | Timezone (e.g., `UTC`, `America/New_York`)                                  |
| `--time-format`    | string     | `auto`    | Time format: `12h`, `24h`, or `auto`                                          |
| `--theme`          | string     | `auto`    | Display theme: `light`, `dark`, `classic`, or `auto`                        |
| `--refresh-rate`   | integer    | `10`      | Data refresh rate in seconds (1-60)                                            |
| `--refresh-per-second` | float      | `0.75`    | Display refresh rate in Hz (0.1-20.0)                                             |
| `--reset-hour`     | integer    | -       | Daily reset hour (0-23)                                                       |
| `--log-level`      | string     | `INFO`    | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                    |
| `--log-file`       | string     | -       | Log file path                                                                 |
| `--debug`          | flag       | `False`   | Enable debug logging                                                          |
| `--version`, `-v`  | flag       | `False`   | Show version information                                                      |
| `--clear`          | flag       | `False`   | Clear saved configuration                                                     |

### 💡 Example Usage

```bash
# Run with the default settings (custom plan, real-time view)
claude-monitor

# Specify your plan
claude-monitor --plan pro  # Pro plan
claude-monitor --plan max5 # Max5 plan

# Adjust refresh rate
claude-monitor --refresh-rate 5

# Set a custom reset time
claude-monitor --reset-hour 0 # resets at midnight

# View your daily usage
claude-monitor --view daily
```

### ✅ Saved Preferences

*   The monitor automatically saves settings like view, theme, timezone, and refresh rates to `~/.claude-monitor/last_used.json`.
*   CLI arguments override saved settings.
*   Plan settings are *not* saved and must be specified each time.
*   Use `--clear` to reset your saved preferences.

---

## 🛡️ Available Plans

| Plan           | Token Limit     | Best For                             |
| -------------- | --------------- | ------------------------------------ |
| `custom`       | P90 auto-detect | Intelligent limit detection (default) |
| `pro`          | ~19,000         | Claude Pro subscription              |
| `max5`         | ~88,000         | Claude Max5 subscription             |
| `max20`        | ~220,000        | Claude Max20 subscription            |

---

## 🚀 What's New in v3.0.0

*   **Complete Architecture Rewrite:** Modular design, Pydantic-based configuration, extensive testing.
*   **P90 Analysis:**  Machine learning for advanced limit detection.
*   **Updated Plan Limits:**  Pro (44k), Max5 (88k), Max20 (220k) tokens.
*   **Rich UI Improvements:** Configurable display refresh rate and automatic 12h/24h format detection.
*   **Command Aliases:** Shorter and more convenient command options like `cmonitor` and `ccm`.

---

## 🔧 Development

For developers, follow the [Development Installation](#-development-installation) instructions.

---

## 🤝 Contributors

A big thank you to our contributors:

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

---

## 🙏 Acknowledgments

Thank you to Ed (Buy Me Coffee Supporter) for their generous support and feedback!

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**⭐ Star this repo to show your support! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>