# TradingAgents-CN: 中文金融交易决策框架 (AI-Powered)

> 🚀 **Empower your financial analysis with TradingAgents-CN, the AI-driven framework optimized for Chinese markets, offering comprehensive stock analysis and actionable insights.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based On](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

TradingAgents-CN is a powerful, AI-driven framework designed to help you make informed financial trading decisions, specifically tailored for the Chinese market.  Built upon the foundation of multi-agent LLMs, this framework provides in-depth analysis of A-share, Hong Kong, and US stocks, leveraging the latest advancements in AI and financial data.

**Key Features:**

*   🤖 **Native OpenAI & Google AI Integration:**  Seamlessly integrates with OpenAI and Google AI models.
*   🇨🇳 **Chinese Market Focus:** Complete support for A-shares, Hong Kong, and US stocks.
*   📰 **Smart News Analysis:** AI-powered news filtering and relevance assessment.
*   📊 **Professional Report Generation:** Generate reports in Markdown, Word, and PDF formats.
*   💻 **Docker Deployment:** Easy setup and environment management with Docker.
*   🌐 **Multi-LLM Provider Support**: Supports major LLM providers including OpenAI, Google AI, Alibaba Cloud's Qwen, DeepSeek and OpenRouter.

**[View the Original Repository](https://github.com/hsliuping/TradingAgents-CN)**

## Core Functionality:

TradingAgents-CN leverages a sophisticated multi-agent system to provide comprehensive stock analysis:

*   **Multi-Agent Architecture:**
    *   **Fundamental Analyst:** Evaluates financial statements and key metrics.
    *   **Technical Analyst:** Analyzes price charts and technical indicators.
    *   **News Analyst:** Gathers, filters, and interprets financial news.
    *   **Sentiment Analyst:** Assesses market sentiment from social media.
    *   **Bull/Bear Researchers:**  Conducts in-depth analysis to support buy/sell/hold recommendations.
    *   **Trader:** Synthesizes all information to generate investment recommendations.

*   **Investment Decision Making:**  Generate actionable buy/sell/hold recommendations.
*   **Risk Management:** Implement multi-layered risk assessment mechanisms.

## What's New in v0.1.13?

### 🤖 Native OpenAI Support

*   **Custom OpenAI Endpoints:** Configure any OpenAI-compatible API endpoint.
*   **Flexible Model Selection:** Use any OpenAI-format model, not just official ones.
*   **Intelligent Adapters:** New native OpenAI adapter for improved compatibility and performance.
*   **Configuration Management:** Unified endpoint and model configuration system.

### 🧠 Comprehensive Google AI Integration

*   **Support for Three Google AI Packages:** langchain-google-genai, google-generativeai, and google-genai.
*   **Nine Verified Models:** Including gemini-2.5-pro, gemini-2.5-flash, and gemini-2.0-flash.
*   **Google AI Tool Processor:** Dedicated processor for Google AI tool calls.
*   **Intelligent Fallback:** Automated fallback to basic functions when advanced functions fail.

### 🔧 LLM Adapter Architecture Optimization

*   **GoogleOpenAIAdapter:** New OpenAI-compatible adapter for Google AI.
*   **Unified Interface:** Consistent calling interface for all LLM providers.
*   **Enhanced Error Handling:** Improved exception handling and automatic retry mechanisms.
*   **Performance Monitoring:** Added LLM call performance monitoring and statistics.

### 🎨 Smart Web Interface Optimization

*   **Intelligent Model Selection:** Automatically selects the best model based on availability.
*   **KeyError Fix:** Resolved KeyError issues in model selection.
*   **UI Response Optimization:** Improved responsiveness and user experience for model switching.
*   **User-Friendly Error Prompts:** Clear error messages and troubleshooting suggestions.

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone the project
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables
cp .env.example .env
# Edit .env file to include your API keys

# 3. Start the service
# For initial builds or code changes
docker-compose up -d --build

# For daily use (existing image, no code changes)
docker-compose up -d

# Smart Start (automatic build detection)
# Windows
powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1

# Linux/Mac
chmod +x scripts/smart_start.sh && ./scripts/smart_start.sh

# 4. Access the application
# Web interface: http://localhost:8501
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

1.  **Choose Model:** DeepSeek V3 / Tongyi Qianwen / Gemini.
2.  **Enter Stock Ticker:** `000001` (A-Share) / `AAPL` (US Stock) / `0700.HK` (Hong Kong Stock).
3.  **Start Analysis:** Click the "🚀 Start Analysis" button.
4.  **Real-time Tracking:** Monitor progress and analysis steps.
5.  **View Report:** Click the "📊 View Analysis Report" button.
6.  **Export Report:** Supports Word/PDF/Markdown formats.

## 🎯 Key Features

*   **Multi-Agent System:** Independent analysts provide multifaceted perspectives.
*   **Market Coverage:** Supports A-shares, Hong Kong stocks, and US stocks.
*   **Comprehensive Data:** Access to financial data and news sources.
*   **Actionable Insights:** Generate buy/sell/hold recommendations with confidence scores.
*   **Flexible Depth of Analysis:** Choose from multiple analysis levels.
*   **Professional Reporting:** Export analysis results in various formats.

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contribute

We welcome contributions!  Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

## 📞 Contact

*   **GitHub Issues:** [Submit issues and suggestions](https://github.com/hsliuping/TradingAgents-CN/issues)
*   **Email:** hsliup@163.com
*   **Project QQ Group:** 782124367
*   **Original Project:** [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
*   **Documentation:** [Complete Documentation](./docs/)