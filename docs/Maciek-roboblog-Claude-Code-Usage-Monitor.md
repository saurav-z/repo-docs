# ⏱️ Claude Code Usage Monitor: Stay in Control of Your AI Token Usage

**Tired of exceeding your Claude AI token limits?**  The Claude Code Usage Monitor provides a real-time terminal dashboard to track your token consumption, predict session limits, and optimize your AI usage. [Check it out on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

---

## 🚀 Key Features

*   **📊 Real-time Monitoring:**  Monitor token usage with customizable refresh rates.
*   **🔮 ML-Powered Predictions:** Intelligent session limit detection and burn rate analysis.
*   **🎨 Rich Terminal UI:**  Beautiful, color-coded displays with WCAG-compliant contrast.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **📈 Cost Analytics:** Model-specific pricing and cache token calculations.
*   **🔄 Customizable Plans:** Support for Pro, Max5, Max20, and Custom plans with dynamic limits.
*   **⚙️ Advanced Configuration:** Timezone, theming, and logging options for tailored insights.

---

## 🚀 Installation

Choose your preferred installation method for the Claude Code Usage Monitor.

### ⚡ Modern Installation with `uv` (Recommended)

`uv` offers the easiest setup and most reliable experience.

```bash
# Install uv (if you haven't already)
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Claude Monitor
uv tool install claude-monitor

# Run the monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
pip install claude-monitor

# If command not found, add to PATH:
# echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
# source ~/.bashrc  # or restart terminal

claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

Installation options include `pipx` and `conda/mamba`.  See the original README for details.

---

## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Basic Usage

```bash
# Run with default settings (Custom plan with auto-detection)
claude-monitor

# Exit the monitor
# Press Ctrl+C to exit
```

### Key Command-Line Parameters

| Parameter              | Type      | Default   | Description                                                                         |
|------------------------|-----------|-----------|-------------------------------------------------------------------------------------|
| `--plan`               | `string`  | `custom`  | Plan type: `pro`, `max5`, `max20`, or `custom`                                    |
| `--custom-limit-tokens` | `int`     | `None`    | Token limit for the custom plan.                                                    |
| `--view`               | `string`  | `realtime`| View type: `realtime`, `daily`, or `monthly`                                      |
| `--timezone`           | `string`  | `auto`    | Timezone (e.g., `UTC`, `America/New_York`)                                          |
| `--time-format`        | `string`  | `auto`    | Time format: `12h`, `24h`, or `auto`                                                 |
| `--theme`              | `string`  | `auto`    | Display theme: `light`, `dark`, `classic`, or `auto`                                |
| `--refresh-rate`       | `int`     | `10`      | Data refresh rate in seconds (1-60)                                                   |
| `--refresh-per-second` | `float`   | `0.75`    | Display refresh rate in Hz (0.1-20.0)                                                 |
| `--reset-hour`         | `int`     | `None`    | Daily reset hour (0-23)                                                               |
| `--log-level`          | `string`  | `INFO`    | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                        |
| `--log-file`           | `path`    | `None`    | Log file path                                                                       |
| `--debug`              | `flag`    | `False`   | Enable debug logging                                                                 |
| `--version`, `-v`      | `flag`    | `False`   | Show version information                                                             |
| `--clear`              | `flag`    | `False`   | Clear saved configuration                                                          |

### Available Plans

| Plan        | Token Limit          | Best For                              |
|-------------|----------------------|---------------------------------------|
| **custom**  | P90 auto-detect      | Intelligent limit detection (default) |
| **pro**     | ~19,000              | Claude Pro subscription             |
| **max5**    | ~88,000              | Claude Max5 subscription            |
| **max20**   | ~220,000             | Claude Max20 subscription           |

---

## ✨ Features & How It Works

### Core Architecture

*   **User Interface Layer:**  CLI Module, settings, error handling, Rich Terminal UI.
*   **Monitoring Orchestrator:**  Central control hub, data management, session monitoring, UI control, analytics.
*   **Foundation Layer:**  Core models, analysis engine, terminal themes, Claude API data.
*   **Data Flow:**  Configuration → Data Layer → Analysis Engine → UI → Terminal

### Key Components

*   **Real-time Monitoring:** Configurable update intervals and high-precision display.
*   **Rich UI:** Progress bars, data tables, and theme support for optimal readability.
*   **Multiple Usage Views:** Real-time, daily, and monthly views for trend analysis.
*   **Machine Learning Predictions:** P90 calculations, burn rate analysis, and session forecasting.
*   **Intelligent Auto-Detection:** Automatic plan switching and limit discovery.

---

## 🔧 Development Installation

*   See the original README for development setup instructions.

---

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

---

## 📞 Contact

*   **Email:** [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

---

## 📝 License

*   [MIT License](LICENSE) -  Use and modify as needed.

---

<div align="center">
  
**⭐ Star this repo to show your support! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>