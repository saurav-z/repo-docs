# 🚀 Claude Code Usage Monitor

**Tired of guessing your Claude AI token usage?** Track, analyze, and predict your token consumption with the **Claude Code Usage Monitor**, a real-time terminal monitoring tool for efficient AI interaction. ([View on GitHub](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor))

---

## Key Features

*   ✅ **Real-time Monitoring:** Track token usage, burn rate, and session limits.
*   🔮 **ML-Powered Predictions:** Get intelligent session limit predictions and P90 percentile calculations.
*   📊 **Advanced Rich UI:** Visualize data with color-coded progress bars, tables, and responsive layouts.
*   🤖 **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   📈 **Cost Analytics:** Model-specific pricing and cache token calculations.
*   🔌 **Easy Installation:** Support for `uv`, `pip`, `pipx`, and `conda/mamba`.

## 🚀 Installation

Choose your preferred method:

### ⚡ Modern Installation with uv (Recommended)

```bash
# Install
uv tool install claude-monitor
# Run
claude-monitor
```

### 📦 Installation with pip

```bash
# Install
pip install claude-monitor
# Run
claude-monitor
```

### 🛠️ Other Package Managers

*   **pipx:**  `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within a conda environment)

## 📖 Usage

### Basic Usage

```bash
# Run with default settings (Custom plan)
claude-monitor
# or shorter aliases
cmonitor, ccmonitor, ccm
```

### Command-Line Parameters

| Parameter          | Description                                  | Default          |
| ------------------ | -------------------------------------------- | ---------------- |
| `--plan`           | Choose your plan (pro, max5, max20, custom) | `custom`        |
| `--view`           | Realtime, daily, or monthly view              | `realtime`        |
| `--timezone`       | Set your timezone                          | `auto`           |
| `--time-format`    | Choose 12h or 24h format                   | `auto`           |
| `--theme`          | Select theme (light, dark, classic, auto)   | `auto`           |
| `--refresh-rate`   | Data refresh rate (seconds)                 | `10`             |
| `--reset-hour`     | Set daily reset hour                       | `None`           |
| `--log-level`      | Logging level (DEBUG, INFO, WARNING, ERROR)  | `INFO`           |

### Plan Options

*   **custom:** P90 auto-detect (default)
*   **pro:** ~19,000 tokens
*   **max5:** ~88,000 tokens
*   **max20:** ~220,000 tokens

## ✨ Features Explained

*   **Real-time Monitoring:** Configurable refresh rates and display.
*   **Rich UI:**  Color-coded progress, tables, and layouts.
*   **ML-based Predictions:** Intelligent limit detection and burn rate analysis.
*   **Automatic Plan Switching:** Switch between plans based on usage patterns.
*   **Cost Analytics:** Model-specific pricing to optimize your AI spending.

## 🔧 Development Installation

For contributors:
```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
pip install -e .
python -m claude_monitor  # Run from source
```

## Troubleshooting

See the [Troubleshooting section in the original README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor#troubleshooting) for common issues and solutions.

## 📚 Resources

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [License](LICENSE)

---

**Enjoy efficient Claude AI usage!**