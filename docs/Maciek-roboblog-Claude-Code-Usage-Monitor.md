# 🚀 Claude Code Usage Monitor: Track, Analyze, and Optimize Your Claude AI Token Usage

Tired of exceeding your Claude AI token limits? **Claude Code Usage Monitor** is a powerful, real-time terminal tool that helps you track, analyze, and optimize your token consumption with advanced analytics and intelligent predictions. View the original repo [here](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor).

[![PyPI Version](https://img.shields.io/pypi/v/claude-monitor.svg)](https://pypi.org/project/claude-monitor/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![codecov](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Maciek-roboblog/Claude-Code-Usage-Monitor)

![Claude Token Monitor Screenshot](https://raw.githubusercontent.com/Maciek-roboblog/Claude-Code-Usage-Monitor/main/doc/scnew.png)

## ✨ Key Features

*   **🔮 ML-based Predictions:** Leverage machine learning for intelligent session limit detection.
*   **🔄 Real-time Monitoring:** Configurable refresh rates with dynamic display updates.
*   **📊 Advanced Rich UI:** Beautiful color-coded progress bars, tables, and layouts.
*   **🤖 Smart Auto-Detection:** Automatic plan switching and custom limit discovery.
*   **📋 Enhanced Plan Support:** Updated limits: Pro, Max5, Max20, and Custom (P90-based).
*   **⚠️ Advanced Warning System:** Multi-level alerts with cost and time predictions.
*   **🎨 Intelligent Theming:** Scientific color schemes with terminal background detection.
*   **📈 Cost Analytics:** Model-specific pricing with cache token calculations.
*   **⚡ Performance Optimized:** Advanced caching and efficient data processing.

## 🚀 Installation

### ⚡ Modern Installation with `uv` (Recommended)

`uv` provides a fast and reliable way to install and manage the monitor.

```bash
# Install with uv
uv tool install claude-monitor

# Run
claude-monitor  # or cmonitor, ccmonitor for short
```

*   **Advantages:** Creates isolated environments, handles Python versions, and simplifies updates.

### 📦 Installation with `pip`

```bash
pip install claude-monitor
claude-monitor  # or cmonitor, ccmonitor for short
```

### 🛠️ Other Package Managers

Install with `pipx` or `conda/mamba`:

```bash
# pipx
pipx install claude-monitor

# conda/mamba
pip install claude-monitor
```

## 📖 Usage

### Get Help

```bash
claude-monitor --help
```

### Basic Usage

```bash
claude-monitor  # or cmonitor, ccmonitor for short
```

## 💡 Key Configuration Options

*   **`--plan`**: `pro`, `max5`, `max20`, or `custom`.
*   **`--view`**: `realtime`, `daily`, or `monthly`.
*   **`--timezone`**: Set timezone (e.g., `America/New_York`).
*   **`--time-format`**: Set time format (`12h`, `24h`, or `auto`).
*   **`--theme`**: Set display theme (`light`, `dark`, `classic`, or `auto`).
*   **`--refresh-rate`**: Update interval in seconds (1-60).

## ✨ Understanding the Custom Plan

The **Custom plan** is the default and analyzes your usage over the last 192 hours (8 days) to calculate personalized limits based on your actual usage for accurate predictions and warnings.

## 📄 What's New in v3.0.0

*   **Complete Architecture Rewrite**: Improved modularity and stability.
*   **P90 Analysis:** Machine learning-based limit detection.
*   **Updated Plan Limits:** Expanded plan options.
*   **New CLI Options**: More control over the display and behavior.
*   **Breaking Changes**: Package name changes and updated defaults.

## 📞 Contact

Need help or have suggestions? Contact [maciek@roboblog.eu](mailto:maciek@roboblog.eu).

## 📝 License

MIT License - Use and modify freely.