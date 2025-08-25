# TradingAgents-CN: 中文金融交易决策框架 - 基于大语言模型的AI股票分析 (🚀 增强版)

> 💡 **Unlock AI-powered insights for the Chinese stock market!** TradingAgents-CN is a cutting-edge framework built on multi-agent LLMs, specifically designed for Chinese users and optimized for A-share/Hong Kong/US stock analysis.  **[Explore the original repo](https://github.com/hsliuping/TradingAgents-CN)**.

## ✨ Key Features

*   **🤖 Native OpenAI & Google AI Integration**: Seamlessly supports OpenAI and Google AI models, including Gemini 2.5 series
*   **🇨🇳 Optimized for Chinese Markets**: Comprehensive A-share, Hong Kong, and US stock analysis.
*   **🧠 Intelligent News Analysis**:  AI-driven news filtering, quality assessment, and sentiment analysis.
*   **🌐 Multi-LLM Provider Support**: Supports major LLM providers like DashScope, DeepSeek, Google AI, OpenAI, and OpenRouter.
*   **🚀 Streamlined Web Interface**: Intuitive Streamlit-based web UI for easy stock analysis and reporting.
*   **📈 Professional Reporting**: Generate detailed reports in Markdown, Word (.docx), or PDF format.
*   **🐳 Dockerized Deployment**: Easy one-click deployment with Docker for a hassle-free experience.
*   **💾 Persistent Configuration**:  Model selection and settings are saved using URL parameters, keeping your settings with every refresh!
*   **🎯 Enhanced Chinese Support**: Built-in Chinese language support for a superior user experience.

## 🌟 What's New in v0.1.13?

This release is a **preview** of the exciting upcoming features!

*   **🤖 Native OpenAI and Google AI Integration:** Custom OpenAI endpoints, 9 Google AI models, and  adapter optimization.
*   **🧠 Google AI Ecosystem Integration**: Full integration of langchain-google-genai, google-generativeai, and google-genai packages.
*   **🔧 Optimized LLM Adapter Architecture**:  Enhanced error handling and performance monitoring.
*   **🎨 Intelligent Web Interface Enhancements:** Smart model selection, UI improvements, and friendlier error handling.

## 🚀 Quick Start

1.  **Docker (Recommended):** `docker-compose up -d --build` or follow the smart start script in the `scripts` folder
2.  **Local Installation:** `pip install -e .` then  `python start_web.py`
3.  **Access the Web Interface:** `http://localhost:8501`
4.  **Enter Stock Code:** e.g., `000001` (A-Share), `AAPL` (US), `0700.HK` (HK)
5.  **Start Analysis:** Choose your research depth and click the "Start Analysis" button.
6.  **View Results:**  Monitor progress and access your professional report.

## 🖼️ Web Interface Screenshots

### 🏠 Main Interface - Analysis Configuration

![Analysis Configuration](images/README/1755003162925.png)

![Market Selection](images/README/1755002619976.png)

*Smart configuration panel supports multi-market stock analysis with 5 levels of research depth.*

### 📊 Real-time Analysis Progress

![Real-time Progress](images/README/1755002731483.png)

*Real-time progress tracking, visualized analysis process, and smart time estimation.*

### 📈 Analysis Results

![Detailed Results](images/README/1755002901204.png)

![Detailed Results](images/README/1755002924844.png)

![Detailed Results](images/README/1755002939905.png)

![Detailed Results](images/README/1755002968608.png)

![Detailed Results](images/README/1755002985903.png)

![Detailed Results](images/README/1755003004403.png)

![Detailed Results](images/README/1755003019759.png)

![Detailed Results](images/README/1755003033939.png)

![Detailed Results](images/README/1755003048242.png)

![Detailed Results](images/README/1755003064598.png)

![Detailed Results](images/README/1755003090603.png)

*Professional investment reports, multi-dimensional analysis results, and one-click export.*

## 🎯 Core Features at a Glance

*   **🤖 Multi-Agent Architecture**: Four specialized analysts (Technical, Fundamental, News, Social Media).
*   **📈 Bull/Bear Researchers**: In-depth analysis with structured debate.
*   **🎯 Trading Agent**: Makes final investment recommendations based on all inputs.
*   **🛡️ Risk Management**: Multi-layered risk assessment and management.
*   **🌐 Multi-Market Analysis**:  Support for US, A-share, and Hong Kong markets.

## 📚 Learn More

*   **Comprehensive Documentation:** [docs/](./docs/) - Installation, usage, and API documentation.
*   **Troubleshooting:** [docs/troubleshooting/](./docs/troubleshooting/) - Solutions to common issues.
*   **Changelog:** [CHANGELOG.md](./docs/releases/CHANGELOG.md) - Detailed release notes.
*   **Quickstart Guide:** [QUICKSTART.md](./QUICKSTART.md) - 5-minute deployment guide.

## 🤝  Special Thanks

Huge thanks to the [Tauric Research](https://github.com/TauricResearch/TradingAgents) team for creating the foundational TradingAgents framework!