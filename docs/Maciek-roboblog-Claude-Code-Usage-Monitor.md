# Claude Code Usage Monitor: Real-Time Token Tracking for Anthropic's Claude AI

**Effortlessly monitor and manage your Anthropic Claude AI token usage with real-time insights, machine learning-powered predictions, and a beautiful terminal UI.**  [Visit the Repository](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor) to get started!

---

## Key Features

*   ✅ **Real-time Monitoring:** Track token consumption, burn rate, and cost in real-time with configurable refresh rates (0.1-20 Hz).
*   🔮 **ML-Powered Predictions:** Benefit from machine learning-based token limit detection and session forecasting.
*   📊 **Advanced Rich UI:** Enjoy a visually appealing and informative terminal interface with color-coded progress bars, data tables, and WCAG-compliant contrast.
*   🤖 **Smart Auto-Detection:** Automatically switch plans and detect custom session limits.
*   📈 **Cost Analytics:** Analyze model-specific pricing and cache token calculations.
*   💡 **Plan Support:** Full support for Claude Pro, Max5, Max20, and the powerful, default *Custom* plan.
*   📝 **Configuration & Logging:** Customize settings, save preferences, and enable comprehensive logging.

## Installation

Choose your preferred method:

### 🚀 **Recommended: Install with `uv`** (Fastest & Simplest)

```bash
# Install uv (if you don't have it already)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

# Install claude-monitor
uv tool install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor, ccm
```

### 📦 Install with `pip`

```bash
pip install claude-monitor

# If 'claude-monitor' not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Run
claude-monitor  # or cmonitor, ccmonitor, ccm
```

### 📦 Other Package Managers

*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor` (within your conda environment)

## 📖 Usage

### Basic Usage

```bash
claude-monitor  # or cmonitor, ccmonitor, ccm - uses the default 'custom' plan
```

### Command Line Options

```bash
claude-monitor --help  # view all options
```

**Key Parameters:**

| Parameter            | Description                                      | Default Value |
| -------------------- | ------------------------------------------------ | ------------- |
| `--plan`             | Select plan: `pro`, `max5`, `max20`, or `custom` | `custom`      |
| `--custom-limit-tokens` | Set custom token limit for `custom` plan       | `None`       |
| `--view`             | View type: `realtime`, `daily`, `monthly`        | `realtime`    |
| `--timezone`         | Timezone (e.g., `UTC`, `America/New_York`)      | `auto`        |
| `--time-format`      | Time format: `12h`, `24h`, or `auto`             | `auto`        |
| `--theme`            | Terminal theme: `light`, `dark`, `classic`, `auto` | `auto`        |
| `--refresh-rate`     | Data refresh rate (seconds)                     | `10`          |
| `--refresh-per-second` | Display refresh rate (Hz)                      | `0.75`       |
| `--reset-hour`       | Daily reset hour (0-23)                           | `None`       |
| `--log-level`        | Logging level: `DEBUG`, `INFO`, `WARNING`, ...    | `INFO`        |
| `--log-file`         | Log file path                                  | `None`        |
| `--debug`            | Enable debug logging                           | `False`       |
| `--clear`            | Clear saved configuration                      | `False`       |

### Example Usage

```bash
# Monitor with the 'pro' plan
claude-monitor --plan pro

# Set a reset time
claude-monitor --reset-hour 3 --timezone America/New_York

# Show daily token usage
claude-monitor --view daily
```

## ✨ Features & How It Works

*   **v3.0.0 Architecture Rewrite**: Built with a modular design adhering to the Single Responsibility Principle (SRP). Features a Pydantic-based configuration, an advanced error-handling system, and comprehensive testing.
*   **Understanding Claude Sessions**: The monitor tracks the 5-hour session windows in which tokens are used.
*   **Token Limits by Plan**: Clear limits for Pro, Max5, Max20, and Custom plans.  The *Custom* plan intelligently adapts to your usage.

## 🤝 Contributing

Contributions are welcome!  See the [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

Special thanks to the supporters!

*   **Ed** - *Buy Me Coffee Supporter*

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)
</div>