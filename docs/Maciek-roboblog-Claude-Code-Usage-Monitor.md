# ⏱️ Claude Code Usage Monitor: Real-time Token Tracking and Analytics

**Tired of exceeding your Claude AI token limits?** Track your usage in real-time with intelligent predictions. Get the [Claude Code Usage Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor), the ultimate terminal tool for monitoring and optimizing your Claude AI usage!

---

## 📑 Table of Contents

*   [✨ Key Features](#-key-features)
*   [🚀 Installation](#-installation)
*   [📖 Usage](#-usage)
*   [🔧 Development Installation](#-development-installation)
*   [Troubleshooting](#troubleshooting)
*   [📞 Contact](#-contact)
*   [📚 Additional Documentation](#-additional-documentation)
*   [📝 License](#-license)
*   [🤝 Contributors](#-contributors)
*   [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ Key Features

*   **Real-time Monitoring:** Track token usage, burn rate, and costs with configurable refresh rates.
*   **Advanced UI:** Beautiful, color-coded terminal display with WCAG-compliant contrast.
*   **ML-Powered Predictions:** Intelligent session limit detection based on your usage history.
*   **Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **Comprehensive Plan Support:** Support for Pro, Max5, Max20 and Custom plans.
*   **Cost Analytics:** Model-specific pricing and cache token calculations.
*   **Flexible Configuration:** Customize timezone, time format, theme, and refresh rate.
*   **Detailed Logging:** Optional file logging for in-depth analysis.

---

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` is the fastest and most reliable way to install and manage dependencies.

```bash
# Install using uv
uv tool install claude-monitor

# Run the monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

### 📦 Installation with `pip`

```bash
# Install using pip
pip install claude-monitor

# Run the monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

For more options including pipx and conda/mamba install instructions, see the original [README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

---

## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Key CLI Parameters:

*   `--plan`:  Select plan (pro, max5, max20, custom).  Defaults to `custom`.
*   `--view`: Choose view type (realtime, daily, monthly). Defaults to `realtime`.
*   `--timezone`:  Set your timezone (auto-detected by default).
*   `--time-format`: Set time format (auto, 12h, 24h).
*   `--theme`: Select a theme (light, dark, classic, auto).
*   `--refresh-rate`: Set data refresh rate in seconds (1-60).
*   `--reset-hour`: Set daily reset hour (0-23).

### Example Usage:

```bash
# Start with default settings
claude-monitor

# Start with custom plan
claude-monitor --plan custom

# Start a Pro plan with a custom limit (if you know it)
claude-monitor --plan pro --custom-limit-tokens 20000

# Specify timezone and theme
claude-monitor --timezone "America/Los_Angeles" --theme dark
```

For complete plan details, command aliases, and save flag settings, refer to the original [README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

---

## 🔧 Development Installation

For contributing or modifying the project:

```bash
git clone https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor.git
cd Claude-Code-Usage-Monitor
pip install -e .
python -m claude_monitor  # Run from source
```

For detailed instructions on testing, virtual environment setup, and development best practices, see the original [README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

---

## Troubleshooting

Refer to the original [README](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor) for solutions to installation and runtime issues.

---

## 📞 Contact

For questions, suggestions, or collaboration, contact:

**📧 Email**: [maciek@roboblog.eu](mailto:maciek@roboblog.eu)

---

## 📚 Additional Documentation

*   **Development Roadmap**:  [DEVELOPMENT.md](DEVELOPMENT.md)
*   **Contributing Guide**:  [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Troubleshooting**:  [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📝 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

---

## 🙏 Acknowledgments

A special thanks to our sponsors, Ed, and anyone else supporting the project.

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>