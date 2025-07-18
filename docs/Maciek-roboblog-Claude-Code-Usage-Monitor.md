# 🚀 Claude Code Usage Monitor: Real-time Token Tracking & AI-Powered Predictions

Tired of exceeding your Claude AI token limits? **Claude Code Usage Monitor** is your ultimate terminal companion, providing real-time tracking, advanced analytics, and intelligent session limit predictions to optimize your Claude AI usage. Check out the original repo [here](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

---

## Key Features:

*   **Real-time Monitoring:** Track token consumption, burn rate, and session duration in real-time.
*   **AI-Powered Predictions:** Leverage machine learning to forecast session limits and provide timely warnings.
*   **Advanced Analytics:** Gain insights into token usage, cost analysis, and overall efficiency.
*   **Rich Terminal UI:** Experience a beautiful, color-coded interface designed for optimal readability.
*   **Smart Auto-Detection:** Automatic plan switching with custom limit discovery.

## 🚀 Installation

### 1. Modern Installation (Recommended):

Use `uv` for effortless and isolated environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the monitor
uv tool install claude-monitor

# Run the monitor
claude-monitor  # or cmonitor, ccmonitor, ccm for short
```

### 2. Alternative Installations:

*   **pip:** `pip install claude-monitor`
*   **pipx:** `pipx install claude-monitor`
*   **conda/mamba:** `pip install claude-monitor`

### 3.  Check Command

```bash
# Show help information
claude-monitor --help
```

## 📖 Usage

### Key Features:

*   **Plan Selection**: Customize your monitoring to your Claude plan. (pro, max5, max20, custom)
*   **Customization**: Configure refresh rates, timezone, themes, and more.
*   **Saved Configuration**:  Save preferences like theme, timezone, and refresh rates for convenience.
*   **Aliases**: Use `cmonitor`, `ccmonitor`, or `ccm` for quick access.

### Command-Line Parameters:

| Parameter             | Type    | Default | Description                                   |
| :-------------------- | :------ | :------ | :-------------------------------------------- |
| `--plan`              | string  | custom  | Plan type: pro, max5, max20, or custom          |
| `--custom-limit-tokens` | int     | None    | Token limit for custom plan (must be > 0)     |
| `--timezone`          | string  | auto    | Timezone (auto-detected)                      |
| `--time-format`       | string  | auto    | Time format: 12h, 24h, or auto              |
| `--theme`             | string  | auto    | Display theme: light, dark, classic, or auto |
| `--refresh-rate`      | int     | 10      | Data refresh rate in seconds (1-60)           |
| `--refresh-per-second` | float   | 0.75    | Display refresh rate in Hz (0.1-20.0)        |
| `--reset-hour`        | int     | None    | Daily reset hour (0-23)                       |
| `--log-level`         | string  | INFO    | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL          |
| `--log-file`          | path  | None  | Log file path          |
| `--debug`          | flag  | False   | Enable debug logging   |
| `--version, -v`       | flag  | False   | Show version information |
| `--clear`          | flag  | False   | Clear saved configuration |

## 🛡️ Troubleshooting

*   **Installation Issues:** Check the "externally-managed-environment" error solutions.  Use `uv` or a virtual environment.
*   **Command Not Found:** Ensure your `PATH` is correctly configured.
*   **Runtime Issues:** Check your Python version and ensure the session is active.

## 📞 Contact

For questions, suggestions, or collaboration, contact [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## 📚 Additional Documentation

*   [Development Roadmap](DEVELOPMENT.md)
*   [Contributing Guide](CONTRIBUTING.md)
*   [Troubleshooting](TROUBLESHOOTING.md)

## 📝 License

[MIT License](LICENSE) - use and modify freely.

## 🤝 Contributors

*   [@adawalli](https://github.com/adawalli)
*   [@taylorwilsdon](https://github.com/taylorwilsdon)
*   [@moneroexamples](https://github.com/moneroexamples)

## 🙏 Acknowledgments

### Sponsors

A special thanks to our supporters who help keep this project going:

**Ed** - *Buy Me Coffee Supporter*
> "I appreciate sharing your work with the world. It helps keep me on track with my day. Quality readme, and really good stuff all around!"

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maciek-roboblog/Claude-Code-Usage-Monitor&type=Date)](https://www.star-history.com/#Maciek-roboblog/Claude-Code-Usage-Monitor&Date)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Request Feature](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor/issues) • [Contribute](CONTRIBUTING.md)

</div>