# TradingAgents-CN: 中文金融交易决策框架 (增强版) 🚀

**Empowering Chinese investors with AI-driven financial analysis: A powerful, localized framework for stock market analysis.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

> **✨ What's New in cn-0.1.13-preview:** Native OpenAI & Comprehensive Google AI Integration!  Experience custom OpenAI endpoints, 9 Google AI models, and streamlined LLM adapter architecture.

**TradingAgents-CN** is a powerful, multi-agent framework built on large language models, specifically tailored for **Chinese-speaking users** to analyze and make informed decisions in the stock market. It provides in-depth analysis of **A-shares, Hong Kong stocks, and US stocks**.

## 🌟 Key Features

*   **🇨🇳 Complete Chinese Language Support:** Optimized for Chinese users and markets.
*   **🤖 Multi-Agent Architecture:**  Four specialized analysts for in-depth market analysis.
*   **🚀 Enhanced AI Integration:** Native OpenAI, and Comprehensive Google AI integration, supporting multiple models.
*   **📊 Comprehensive Market Coverage:**  A-shares, Hong Kong stocks, and US stocks.
*   **📈 Professional Reporting:** Generate investment reports in multiple formats (Markdown, Word, PDF).
*   **🐳 Docker Deployment:**  Easy to deploy and run with Docker.
*   **📰 Intelligent News Analysis:**  AI-powered news filtering and sentiment analysis.
*   **🔑 Persistent Configuration:**  Model and setting persistence for a seamless user experience.
*   **🌐 Multi-LLM Support**: Supports multiple LLM providers (OpenAI, Google, DashScope, DeepSeek, OpenRouter).

**[Learn more on GitHub](https://github.com/hsliuping/TradingAgents-CN)**.

## 🙏 Inspired by the Original

This project builds upon the groundbreaking work of [Tauric Research](https://github.com/TauricResearch) and their [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework. We extend its capabilities to provide a fully localized and enhanced experience for Chinese users.

## 🚀 Key Updates & Highlights

### 🤖 v0.1.13:  Native OpenAI & Google AI Integration

*   **OpenAI Integration:**
    *   Custom OpenAI Endpoint Support
    *   Flexible Model Selection (OpenAI models supported)
    *   Improved OpenAI Adapter for better compatibility.
    *   Unified Endpoint and Model Configuration.

*   **Google AI Ecosystem Integration:**
    *   Support for langchain-google-genai, google-generativeai, and google-genai packages.
    *   9 verified Google AI models, including Gemini 2.5 series.
    *   Dedicated Google AI tool processors
    *   Intelligent Fallback Mechanism

*   **LLM Adapter Enhancements:**
    *   GoogleOpenAIAdapter for OpenAI compatibility.
    *   Unified API for interacting with different LLM providers.
    *   Improved Error Handling and Retries
    *   Performance monitoring.

### 🧠 v0.1.12: Intelligent News Analysis

*   **Smart News Filtering:** AI-powered news relevance scoring and quality assessment.
*   **Multi-Layer Filtering:** Base, Enhanced, and Integrated filtering mechanisms.
*   **News Quality Assessment:** Automatic identification and filtering of low-quality, repetitive, and irrelevant news.
*   **Unified News Tools:** Integration of multiple news sources with a unified interface.

### 🔧 v0.1.11: Multi-LLM Integration & Persistence

*   **Multi-LLM Support**: Integrates with multiple LLM providers (OpenAI, Google, DashScope, DeepSeek, OpenRouter).
*   **Model Persistence**: Model settings are stored in the URL for easy sharing and recall.
*   **Quick Switch Buttons**: One-click selection of popular models.
*   **Real-Time Progress Display**: Asynchronous progress tracking for analysis steps.
*   **Session Management**: Saves progress and automatically recovers sessions.

### 📈 Core Features

*   **Multi-Agent Collaboration:** Specialized analysts (fundamental, technical, news, social media) work together.
*   **Structured Debate:** Bullish/bearish researchers provide in-depth analysis.
*   **Intelligent Decision-Making:** Traders make final investment recommendations.
*   **Risk Management:** Multi-layered risk assessment and management.

## 🖥️ Web Interface Showcase

### 📸 Example Web Interface Screenshots

>  **Modern Web Interface:** Streamlit-based responsive web application providing an intuitive stock analysis experience.

#### 🏠 Main Interface - Analysis Configuration

![1755003162925](images/README/1755003162925.png)

![1755002619976](images/README/1755002619976.png)

*Intelligent configuration panel, supporting multi-market stock analysis, and 5 research depth selections*

#### 📊 Real-Time Analysis Progress

![1755002731483](images/README/1755002731483.png)

*Real-time progress tracking, visual analysis process, smart time estimation*

#### 📈 Analysis Result Display

![1755002901204](images/README/1755002901204.png)

![1755002924844](images/README/1755002924844.png)

![1755002939905](images/README/1755002939905.png)

![1755002968608](images/README/1755002968608.png)

![1755002985903](images/README/1755002985903.png)

![1755003004403](images/README/1755003004403.png)

![1755003019759](images/README/1755003019759.png)

![1755003033939](images/README/1755003033939.png)

![1755003048242](images/README/1755003048242.png)

![1755003064598](images/README/1755003064598.png)

![1755003090603](images/README/1755003090603.png)

*Professional investment reports, multi-dimensional analysis results, one-click export function*

### 🎯 Core Functionality Highlights

#### 📋 **Smart Analysis Configuration**

*   **🌍 Multi-Market Support:** US, A-shares, and Hong Kong stocks.
*   **🎯 5 Research Depths:** From quick 2-minute analysis to comprehensive 25-minute studies.
*   **🤖 Agent Selection:** Technical, fundamental, news, and social media analysts.
*   **📅 Flexible Time Settings:** Supports analysis at historical time points.

#### 🚀 **Real-Time Progress Tracking**

*   **📊 Visual Progress:** Real-time display of analysis progress and remaining time.
*   **🔄 Intelligent Step Recognition:** Automatic recognition of the current analysis stage.
*   **⏱️ Accurate Time Estimation:** Intelligent time calculation based on historical data.
*   **💾 State Persistence:** Analysis progress is maintained even after page refresh.

#### 📈 **Professional Result Display**

*   **🎯 Investment Decisions:** Clear buy/hold/sell recommendations.
*   **📊 Multi-Dimensional Analysis:** Integrated technical, fundamental, and news assessments.
*   **🔢 Quantitative Indicators:** Confidence levels, risk scores, and target price levels.
*   **📄 Professional Reports:** Supports Markdown/Word/PDF report exports.

#### 🤖 **Multi-LLM Model Management**

*   **🌐 4 Providers:** DashScope, DeepSeek, Google AI, OpenRouter.
*   **🎯 60+ Model Choices:** Covers economical to flagship models.
*   **💾 Configuration Persistence:** URL parameter storage, settings are retained after refresh.
*   **⚡ Quick Switching:** One-click model selection for 5 popular models.

### 🎮 Web Interface User Guide

#### 🚀 **Quick Start Guide**

1.  **Start the application:** `python start_web.py` or `docker-compose up -d`
2.  **Access the interface:** Open `http://localhost:8501` in your browser.
3.  **Configure the model:** Select the LLM provider and model in the sidebar.
4.  **Enter the stock:** Input the stock code (e.g., AAPL, 000001, 0700.HK).
5.  **Select depth:** Choose research depth from 1 to 5.
6.  **Start analysis:** Click the "🚀 Start Analysis" button.
7.  **View the results:** Track progress and view the analysis report.
8.  **Export the report:** Export professional format reports with one click.

#### 📊 **Supported Stock Code Formats**

*   **🇺🇸 US Stocks:** `AAPL`, `TSLA`, `MSFT`, `NVDA`, `GOOGL`
*   **🇨🇳 A-shares:** `000001`, `600519`, `300750`, `002415`
*   **🇭🇰 Hong Kong Stocks:** `0700.HK`, `9988.HK`, `3690.HK`, `1810.HK`

#### 🎯 **Research Depth Explanation**

*   **Level 1 (2-4 minutes):** Quick overview, basic technical indicators.
*   **Level 2 (4-6 minutes):** Standard analysis, technical + fundamentals.
*   **Level 3 (6-10 minutes):** In-depth analysis, including news sentiment. ⭐ **Recommended**
*   **Level 4 (10-15 minutes):** Comprehensive analysis, multi-round agent debate.
*   **Level 5 (15-25 minutes):** Most in-depth analysis, complete research report.

#### 💡 **Tips**

*   **🔄 Real-Time Refresh:** Refresh the page during analysis without losing progress.
*   **📱 Mobile Adaptability:** Supports mobile and tablet access.
*   **🎨 Dark Mode:** Automatically adapts to system theme settings.
*   **⌨️ Shortcuts:** Supports the Enter key for quick analysis submissions.
*   **📋 History:** Automatically saves recent analysis configurations.

> 📖 **Detailed Guide:** For complete web interface usage, please refer to [🖥️ Web Interface Detailed Guide](docs/usage/web-interface-detailed-guide.md)

## 🎯 Feature Highlights

### 🚀  Intelligent News Analysis✨ **v0.1.12 Major Upgrade**

| Feature               | Status        | Description                                     |
| ---------------------- | ----------- | ----------------------------------------------- |
| **🧠 Smart News Analysis**    | 🆕 v0.1.12  | AI news filtering, quality assessment, relevance analysis  |
| **🔧 News Filter**      | 🆕 v0.1.12  | Multi-level filtering, base/enhanced/integrated processing       |
| **📰 Unified News Tools**    | 🆕 v0.1.12  | Integrates multiple news sources, unified interface, intelligent retrieval         |
| **🤖 Multi-LLM Providers**     | 🆕 v0.1.11  | 4 major providers, 60+ models, smart classification management        |
| **💾 Model Selection Persistence**  | 🆕 v0.1.11  | URL parameter storage, keeps settings after refresh, configuration sharing        |
| **🎯 Quick Selection Buttons**    | 🆕 v0.1.11  | One-click switching of popular models, improves operating efficiency        |
| **📊 Real-time Progress Display**    | ✅ v0.1.10  | Asynchronous progress tracking, intelligent step identification, accurate time calculation |
| **💾 Smart Session Management**    | ✅ v0.1.10  | State persistence, automatic downgrading, cross-page recovery         |
| **🎯 One-Click Report View**    | ✅ v0.1.10  | One-click view after analysis, intelligent result restoration         |
| **🖥️ Streamlit Interface** | ✅ Full Support | Modern, responsive interface, real-time interaction and data visualization   |
| **⚙️ Configuration Management**      | ✅ Full Support | Web-based API key management, model selection, parameter configuration     |

### 🎨 CLI User Experience ✨ **v0.1.9 Optimization**

| Feature                | Status        | Description                             |
| ----------------------- | ----------- | ------------------------------------ |
| **🖥️ Interface and Log Separation** | ✅ Full Support | Clean and beautiful user interface, independent management of technical logs   |
| **🔄 Smart Progress Display**     | ✅ Full Support | Multi-stage progress tracking, prevents repeated prompts         |
| **⏱️ Time Estimation Function**   | ✅ Full Support | Displays the estimated time for each analysis stage             |
| **🌈 Rich Color Output**     | ✅ Full Support | Color progress indicators, status icons, improved visual effects |

### 🧠 LLM Model Support ✨ **v0.1.13 Full Upgrade**

| Model Provider        | Supported Models                     | Special Features                | New Features |
| ----------------- | ---------------------------- | ----------------------- | -------- |
| **🇨🇳 Alibaba Baichuan** | qwen-turbo/plus/max          | Chinese optimization, cost-effective    | ✅ Integrated  |
| **🇨🇳 DeepSeek** | deepseek-chat                | Tool calling, extremely cost-effective    | ✅ Integrated  |
| **🌍 Google AI**  | **9 Verified Models**              | Latest Gemini 2.5 series      | 🆕 Upgraded  |
| ├─**Latest Flagship**  | gemini-2.5-pro/flash         | Latest flagship, super fast response      | 🆕 New  |
| ├─**Stable Recommendation**  | gemini-2.0-flash             | Recommended use, balance performance      | 🆕 New  |
| ├─**Classic and Powerful**  | gemini-1.5-pro/flash         | Classic and stable, high-quality analysis    | ✅ Integrated  |
| └─**Lightweight and Fast**  | gemini-2.5-flash-lite        | Lightweight tasks, fast response    | 🆕 New  |
| **🌐 Native OpenAI** | **Custom Endpoint Support**           | Any OpenAI compatible endpoint      | 🆕 New  |
| **🌐 OpenRouter** | **60+ Model Aggregation Platform**          | One API access to all mainstream models | ✅ Integrated  |
| ├─**OpenAI**    | o4-mini-high, o3-pro, GPT-4o | Latest o series, professional reasoning   | ✅ Integrated  |
| ├─**Anthropic** | Claude 4 Opus/Sonnet/Haiku   | Top performance, balanced versions      | ✅ Integrated  |
| ├─**Meta**      | Llama 4 Maverick/Scout       | Latest Llama 4 series         | ✅ Integrated  |
| └─**Custom**    | Any OpenRouter Model ID         | Unlimited expansion, personalized selection    | ✅ Integrated  |

**🎯 Quick Selection**: 5 hot model quick buttons | **💾 Persistence**: URL parameter storage, refresh keeps | **🔄 Smart Switching**: One-click switch different providers

### 📊 Data Sources & Markets

| Market Type      | Data Source                   | Coverage                     |
| ------------- | ------------------------ | ---------------------------- |
| **🇨🇳 A-shares**  | Tushare, AkShare, Tongdaxin | Shanghai and Shenzhen Stock Exchanges, real-time quotes, financial data |
| **🇭🇰 Hong Kong Stocks** | AkShare, Yahoo Finance   | Hong Kong Stock Exchange, real-time quotes, fundamentals     |
| **🇺🇸 US Stocks** | FinnHub, Yahoo Finance   | NYSE, NASDAQ, real-time data       |
| **📰 News**   | Google News              | Real-time news, multi-language support         |

### 🤖 Agent Team

**Analyst Team**: 📈 Market Analysis | 💰 Fundamental Analysis | 📰 News Analysis | 💬 Sentiment Analysis
**Research Team**: 🐂 Bullish Researcher | 🐻 Bearish Researcher | 🎯 Trading Decision Maker
**Management**: 🛡️ Risk Manager | 👔 Research Director

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone the project
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables
cp .env.example .env
# Edit the .env file and fill in your API keys.

# 3. Start the service
# For the first startup or when code changes (requires building the image)
docker-compose up -d --build

# Daily startup (image exists, no code changes)
docker-compose up -d

# Intelligent startup (automatically determines if building is required)
# Windows
powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1

# Linux/Mac
chmod +x scripts/smart_start.sh && ./scripts/smart_start.sh

# 4. Access the application
# Web interface: http://localhost:8501
```

### 💻 Local Deployment

```bash
# 1. Upgrade pip (Important! Avoid installation errors)
python -m pip install --upgrade pip

# 2. Install dependencies
pip install -e .

# 3. Start the application
python start_web.py

# 4. Access http://localhost:8501
```

### 📊 Start Analyzing

1.  **Select Model**: DeepSeek V3 / Tongyi Qianwen / Gemini
2.  **Enter Stock**: `000001` (A-shares) / `AAPL` (US stocks) / `0700.HK` (Hong Kong stocks)
3.  **Start Analysis**: Click the "🚀 Start Analysis" button
4.  **Real-Time Tracking**: Observe real-time progress and analysis steps
5.  **View Report**: Click the "📊 View Analysis Report" button
6.  **Export Report**: Supports Word/PDF/Markdown format

## 🎯 Key Advantages

*   🧠 **Intelligent News Analysis:**  AI-driven news filtering and quality assessment system in v0.1.12.
*   🔧 **Multi-Layer Filtering:** Base, enhanced, and integrated filtering for news.
*   📰 **Unified News Tools:** Integrates multiple news sources, providing a unified intelligent retrieval interface.
*   🆕 **Multi-LLM Integration:** v0.1.11: Integrates with 4 providers, 60+ models, for a one-stop AI experience.
*   💾 **Configuration Persistence:** Model selection truly persistent, URL parameter storage, refresh retains settings.
*   🎯 **Quick Switching:** 5 quick buttons for popular models, one-click switching between different AIs.
*   🆕 **Real-Time Progress:** v0.1.10: Asynchronous progress tracking, no more waiting in the dark.
*   💾 **Smart Session:** Session persistence, analysis results are not lost upon page refresh.
*   🇨🇳 **Chinese Optimized:** A-shares/Hong Kong stock data + domestic LLMs + Chinese interface.
*   🐳 **Containerization:** Docker one-click deployment, environment isolation, rapid expansion.
*   📄 **Professional Reports:** Multi-format export, automated investment recommendations.
*   🛡️ **Stable and Reliable:** Multi-layered data sources, smart downgrading, error recovery.

## 🔧 Technical Architecture

**Core Technologies**: Python 3.10+ | LangChain | Streamlit | MongoDB | Redis
**AI Models**: DeepSeek V3 | Alibaba Baichuan | Google AI | OpenRouter(60+ models) | OpenAI
**Data Sources**: Tushare | AkShare | FinnHub | Yahoo Finance
**Deployment**: Docker | Docker Compose | Local Deployment

## 📚 Documentation and Support

*   **📖 Complete Documentation**: [docs/](./docs/) - Installation Guide, Usage Tutorials, API Documentation
*   **🚨 Troubleshooting**: [troubleshooting/](./docs/troubleshooting/) - Solutions to Common Problems
*   **🔄 Changelog**: [CHANGELOG.md](./docs/releases/CHANGELOG.md) - Detailed Version History
*   **🚀 Quick Start**: [QUICKSTART.md](./QUICKSTART.md) - 5-Minute Quick Deployment Guide

## 🆚 Chinese Enhanced Features

**Compared to the Original**: Intelligent News Analysis | Multi-layered News Filtering | News Quality Assessment | Unified News Tools | Multi-LLM Provider Integration | Model Selection Persistence | Quick Switch Buttons | Real-time Progress Display | Smart Session Management | Chinese Interface | A-Share Data | Domestic LLMs | Docker Deployment | Professional Report Export | Unified Log Management | Web Configuration Interface | Cost Optimization

**Docker Deployment Includes Services**:

*   🌐 **Web Application**: TradingAgents-CN main program
*   🗄️ **MongoDB**: Data persistence storage
*   ⚡ **Redis**: High-speed cache
*   📊 **MongoDB Express**: Database management interface
*   🎛️ **Redis Commander**: Cache management interface

---

**[Get Started with TradingAgents-CN!](https://github.com/hsliuping/TradingAgents-CN)**