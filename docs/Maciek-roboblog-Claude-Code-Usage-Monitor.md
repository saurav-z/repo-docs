# Claude Code Usage Monitor: Track Your Claude AI Token Usage in Real-time with Intelligent Analytics

**Tired of exceeding your Claude AI token limits?** The **Claude Code Usage Monitor** provides a beautiful, real-time terminal interface to monitor your token consumption, offering advanced analytics, machine learning-powered predictions, and detailed cost analysis.  [Check it out on GitHub!](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)

---

## Key Features

*   📊 **Real-time Monitoring**: Customizable refresh rates (0.1-20 Hz) for up-to-the-second data.
*   🤖 **Intelligent Auto-Detection**: Automatic plan switching with custom limit discovery.
*   🔮 **ML-based Predictions**: Includes P90 percentile calculations and intelligent session limit detection.
*   📈 **Cost Analytics**: Model-specific pricing with cache token calculations.
*   🎨 **Rich UI**: Beautiful, color-coded progress bars, tables, and layouts with WCAG-compliant contrast.
*   ⚠️ **Advanced Warning System**: Multi-level alerts with cost and time predictions.
*   📋 **Enhanced Plan Support**: Support for Pro, Max5, Max20, and a custom plan with auto-detection.
*   ⚡️ **Performance Optimized**: Advanced caching and efficient data processing.

---

## 🚀 Installation

### ⚡ Modern Installation with uv (Recommended)

**Why uv?**

*   ✅ Automatically creates isolated environments (avoids system conflicts).
*   ✅ No Python version issues.
*   ✅ No "externally-managed-environment" errors.
*   ✅ Easy to update and uninstall.
*   ✅ Works on all platforms.

#### Install from PyPI

```bash
# Install directly from PyPI with uv (easiest)
uv tool install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

#### Install from Source

```bash
# Clone and install from source
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
uv tool install .

# Run from anywhere
claude-monitor
```

#### First-time uv users
If you don't have uv installed yet, get it with one command:

```bash
# On Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

### 📦 Installation with pip

```bash
# Install from PyPI
pip install claude-monitor

# If claude-monitor command is not found, add ~/.local/bin to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or restart your terminal

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

> **⚠️ PATH Setup**: If you see `WARNING: The script claude-monitor is installed in '/home/username/.local/bin'` which is not on PATH, follow the `export PATH` command.
>
> **⚠️ Important**: On modern Linux distributions, consider using `uv` or a virtual environment to avoid "externally-managed-environment" errors.

### 🛠️ Other Package Managers

#### pipx (Isolated Environments)

```bash
# Install with pipx
pipx install claude-monitor

# Run from anywhere
claude-monitor  # or claude-code-monitor, cmonitor, ccmonitor, ccm for short
```

#### conda/mamba

```bash
# Install with pip in conda environment
pip install claude-monitor

# Run from anywhere
claude-monitor  # or cmonitor, ccmonitor for short
```

---

## 📖 Usage

### Get Help

```bash
# Show help information
claude-monitor --help
```

### Basic Usage

```bash
# Default (Custom plan with auto-detection)
claude-monitor

# Alternative commands
claude-code-monitor  # Full descriptive name
cmonitor             # Short alias
ccmonitor            # Short alternative
ccm                  # Shortest alias

# Exit the monitor
# Press Ctrl+C to gracefully exit
```

### Command-Line Parameters

| Parameter               | Type       | Default | Description                                                   |
| :---------------------- | :--------- | :------ | :------------------------------------------------------------ |
| `--plan`                | `string`   | `custom` | Plan type: `pro`, `max5`, `max20`, or `custom`               |
| `--custom-limit-tokens` | `int`      | `None`   | Token limit for custom plan (must be > 0)                    |
| `--view`                | `string`   | `realtime` | View type: `realtime`, `daily`, or `monthly`                   |
| `--timezone`            | `string`   | `auto`   | Timezone (auto-detected). Examples: `UTC`, `America/New_York` |
| `--time-format`         | `string`   | `auto`   | Time format: `12h`, `24h`, or `auto`                          |
| `--theme`               | `string`   | `auto`   | Display theme: `light`, `dark`, `classic`, or `auto`         |
| `--refresh-rate`        | `int`      | `10`     | Data refresh rate in seconds (1-60)                          |
| `--refresh-per-second`  | `float`    | `0.75`   | Display refresh rate in Hz (0.1-20.0)                         |
| `--reset-hour`          | `int`      | `None`   | Daily reset hour (0-23)                                      |
| `--log-level`           | `string`   | `INFO`   | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file`            | `path`     | `None`   | Log file path                                                |
| `--debug`               | `flag`     | `False`  | Enable debug logging                                          |
| `--version`, `-v`       | `flag`     | `False`  | Show version information                                       |
| `--clear`               | `flag`     | `False`  | Clear saved configuration                                     |

### Configuration Options

*   **Plan Selection:** Choose your plan (pro, max5, max20, or custom). Custom plan is the default and offers auto-detection.
*   **Custom Reset Times:** Configure your daily token reset time.
*   **Usage Views:** Real-time, daily, and monthly views to track usage trends.
*   **Performance and Display:** Adjust refresh rates, time format, and theme.
*   **Timezone Configuration:** Set your timezone for accurate monitoring.
*   **Logging and Debugging:** Enable logging and debugging options for troubleshooting.

### Available Plans

| Plan       | Token Limit    | Best For                                  |
| :--------- | :------------- | :---------------------------------------- |
| **custom** | P90 auto-detect | Intelligent limit detection (default) |
| **pro**    | ~19,000        | Claude Pro subscription                 |
| **max5**   | ~88,000        | Claude Max5 subscription                |
| **max20**  | ~220,000       | Claude Max20 subscription               |

---

## ✨ Features & How It Works

*   **Real-time Monitoring:** Configurable update intervals and high-precision display refresh, with intelligent change detection.
*   **Rich UI Components:** Includes progress bars, sortable data tables, and an adaptive theme system.
*   **Usage Views:** Realtime, Daily, and Monthly, for tracking your usage over different time periods.
*   **Machine Learning Predictions:** Includes P90 calculation for intelligent limit detection, burn rate analytics, and cost projections.
*   **Intelligent Auto-Detection:** Features automatic plan switching, background and system detection, and limit discovery.

---

## 📝 License

MIT License - feel free to use and modify.

---

## 🤝 Contributors

-   [@adawalli](https://github.com/adawalli)
-   [@taylorwilsdon](https://github.com/taylorwilsdon)
-   [@moneroexamples](https://github.com/moneroexamples)

---
<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>