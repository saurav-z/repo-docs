# TradingAgents-CN: 中文金融交易决策框架 🚀

**Unlock the power of AI for financial trading with TradingAgents-CN, an enhanced, Chinese-optimized framework built upon the foundation of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.12-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## Key Features:

*   **🧠 Advanced News Analysis:**  AI-powered filtering, quality assessment, and relevance analysis for Chinese financial news. (v0.1.12)
*   **🤖 Multi-LLM Support:**  Seamless integration with multiple LLM providers, including OpenAI, Anthropic, Google AI, and custom models via OpenRouter. (v0.1.11)
*   **💾 Persistent Model Selection:**  Save and share your preferred LLM configurations with URL-based persistence. (v0.1.11)
*   **🇨🇳 Chinese Market Focus:** Comprehensive support for A-shares, Hong Kong stocks, and US stocks, with Chinese language optimization.
*   **🐳 Docker Deployment:**  Easy setup with Docker, ensuring environment isolation and quick scaling.
*   **📊 Professional Reporting:** Generate reports in Markdown, Word, and PDF formats for in-depth analysis.
*   **🚀 Real-time Progress Updates:**  Monitor analysis steps with clear, asynchronous progress indicators. (v0.1.10)

## What's New in v0.1.12?

### 🧠 Enhanced News Analysis Module

*   **AI-Driven Filtering:**  News relevance scoring and quality evaluation powered by AI.
*   **Multi-Tier Filtering:**  Three-level filtering: Basic, Enhanced, and Integrated.
*   **Quality Assessment:** Automated identification and filtering of low-quality or irrelevant news.
*   **Unified News Tools:**  Consolidated access to multiple news sources through a single interface.

For other updates, please see [CHANGELOG.md](./docs/releases/CHANGELOG.md)

## Core Functionality

This project is a comprehensive **Chinese Financial Trading Decision-Making Framework** based on multi-agent Large Language Models. It offers:

*   **Multi-Agent Architecture:**  Specialized agents for Fundamental, Technical, News, and Social Media analysis.
*   **Structured Debate:**  Bullish and bearish researchers conduct in-depth analysis.
*   **Intelligent Decision-Making:** Traders make final investment recommendations based on all inputs.
*   **Risk Management:**  Multi-level risk assessment and management mechanisms.

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone the project
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables
cp .env.example .env
# Edit .env file and enter your API keys

# 3. Start the service
# First-time start or code changes (build image required)
docker-compose up -d --build

# Daily start (image exists, no code changes)
docker-compose up -d

# Intelligent start (automatically determines if a build is needed)
# Windows
powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1

# Linux/Mac
chmod +x scripts/smart_start.sh && ./scripts/smart_start.sh

# 4. Access the application
# Web Interface: http://localhost:8501
```

### 💻 Local Deployment

```bash
# 1. Upgrade pip (Important!)
python -m pip install --upgrade pip

# 2. Install dependencies
pip install -e .

# 3. Start the application
python start_web.py

# 4. Access http://localhost:8501
```

### 📊 Start Analyzing

1.  **Select Model:** DeepSeek V3 / Qwen / Gemini
2.  **Enter Stock Ticker:** `000001` (A-shares) / `AAPL` (US Stocks) / `0700.HK` (Hong Kong Stocks)
3.  **Start Analysis:** Click the "🚀 Start Analysis" button.
4.  **Real-time Tracking:** Observe the real-time progress and analysis steps.
5.  **View Report:** Click the "📊 View Analysis Report" button.
6.  **Export Report:** Supports Word/PDF/Markdown formats.

## 📚 Documentation & Support

*   **📖 Comprehensive Documentation:** [docs/](./docs/) - Installation guide, usage tutorials, API documentation.
*   **🚨 Troubleshooting:** [troubleshooting/](./docs/troubleshooting/) - Solutions to common problems.
*   **🔄 Changelog:** [CHANGELOG.md](./docs/releases/CHANGELOG.md) - Detailed version history.
*   **🚀 Quick Start:** [QUICKSTART.md](./QUICKSTART.md) - 5-minute quick deployment guide.

## 🙏 Acknowledgements

We are deeply grateful to the [Tauric Research](https://github.com/TauricResearch) team for creating the groundbreaking [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework.

We also thank all the contributors and users who have helped build and improve TradingAgents-CN.

## 🤝 Contribute

We welcome contributions of all kinds. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file.

---

<div align="center">

**🌟 If you find this project helpful, please give us a Star!**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)

</div>