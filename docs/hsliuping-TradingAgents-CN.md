# TradingAgents-CN: 中文金融交易决策框架 (Enhanced)

> **Unlock the power of AI for Chinese financial markets!** TradingAgents-CN is a framework built upon multi-agent LLMs, optimized for Chinese users, and offering comprehensive A-share, H-share, and US stock analysis capabilities.  Explore the original project: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).

## Key Features

*   **🤖 Multi-LLM Provider Support:** Integrated with leading LLM providers including DashScope (Aliyun), DeepSeek V3, Google AI, and OpenRouter, offering a wide selection of over 60 AI models.
*   **💾 Persistent Model Selection:** Stores and restores LLM configurations via URL parameters, ensuring that your model choices are saved across sessions.
*   **🚀 Enhanced Web Interface:** Improved UI with a 320px sidebar, quick-select buttons, responsive design, and optimized memory management.
*   **📊 Detailed Analysis & Reporting:** Generates professional reports in Markdown, Word, and PDF formats.
*   **🇨🇳 A-Share, H-Share, US Stock Support:** Comprehensive data integration for A-shares, H-shares, and US stocks.
*   **🐳 Docker Deployment:** Easy one-click deployment with Docker for a streamlined setup.

## What's New in v0.1.11

*   **🤖 Expanded LLM Support:** Integration with four major LLM providers and a selection of 60+ AI models, including the latest from Claude 4 Opus, GPT-4o, Llama 4, and Gemini 2.5.
*   **💾 Configuration Persistence:** Model selections are now saved via URL parameters and automatically restored, with detailed logging for debugging.
*   **🎨 Improved Web Interface:** Enhancements include a 320px sidebar for better space utilization, quick-select buttons for easy model switching, and improved responsiveness.

## Core Features

*   **🤖 Multi-Agent Collaboration:** A team of analysts (Fundamental, Technical, News, Sentiment) and researchers (Bullish/Bearish) work together to provide expert financial analysis.
*   **🎯 Structured Debate:** Bullish and bearish researchers conduct in-depth analysis, leading to better-informed decision-making.
*   **💰 Intelligent Decision-Making:** Traders make final investment recommendations based on the combined analysis.
*   **🛡️ Risk Management:** Multi-layered risk assessment and management mechanisms are in place.

## Core Functionality Highlights

### Web Interface
| Feature                | Status          | Details                                                                                                   |
| ----------------------- | --------------- | --------------------------------------------------------------------------------------------------------- |
| **🤖 Multi-LLM Providers** | 🆕 v0.1.11       | Supports 4 providers, 60+ models, and intelligent categorization.                                       |
| **💾 Model Persistence**  | 🆕 v0.1.11       | Stores selections in URL parameters for persistence across refreshes and easy configuration sharing.  |
| **🎯 Quick Select Buttons**| 🆕 v0.1.11       | Easily switch between popular models.                                                                       |
| **📐 320px Sidebar**     | 🆕 v0.1.11       | Optimized for better space utilization and improved responsiveness.                                          |
| **📊 Real-time Progress**  | ✅ v0.1.10       | Asynchronous progress tracking, intelligent step identification, and accurate time calculation.            |
| **💾 Session Management**  | ✅ v0.1.10       | Persistent sessions, automatic downgrading, and cross-page restoration.                                    |
| **🎯 Report Generation**    | ✅ v0.1.10       | Generate and review analysis reports with a single click, with intelligent result recovery.                |
| **🖥️ Streamlit Interface**| ✅ Complete Support| Modern, responsive interface with real-time interaction and data visualization.                           |
| **⚙️ Configuration Management**| ✅ Complete Support| Web-based API key management, model selection, and parameter configuration.                             |

### CLI Experience
| Feature               | Status          | Details                                                                  |
| ---------------------- | --------------- | ------------------------------------------------------------------------ |
| **🖥️ Interface/Logs Separation** | ✅ Complete Support | Clean and beautiful user interface, with separate technical logs.            |
| **🔄 Intelligent Progress**    | ✅ Complete Support | Multi-stage progress tracking, preventing repeated prompts.                |
| **⏱️ Time Estimation** | ✅ Complete Support | Intelligent analysis stage, showing estimated time for completion.       |
| **🌈 Rich Output**     | ✅ Complete Support | Color-coded progress indicators, status icons, and enhanced visual effects. |

### LLM Model Support
| Provider        | Supported Models                                              | Features                      | New Feature |
| --------------- | ----------------------------------------------------------- | ----------------------------- | ----------- |
| **🇨🇳 Aliyun**   | qwen-turbo/plus/max                                        | Chinese optimized, cost-effective | ✅ Integrated|
| **🇨🇳 DeepSeek**  | deepseek-chat                                               | Tool calling, excellent value       | ✅ Integrated|
| **🌍 Google AI**  | gemini-2.0-flash/1.5-pro                                   | Multimodal support, strong inference  | ✅ Integrated|
| **🌐 OpenRouter** | **60+ Model Aggregation Platform**                         | One API access to all mainstream models| 🆕 Added     |
| ├─**OpenAI**    | o4-mini-high, o3-pro, GPT-4o                                  | Latest 'o' series, Professional inference | 🆕 Added     |
| ├─**Anthropic** | Claude 4 Opus/Sonnet/Haiku                                  | Top performance, balanced versions   | 🆕 Added     |
| ├─**Meta**      | Llama 4 Maverick/Scout                                      | Latest Llama 4 series            | 🆕 Added     |
| ├─**Google**    | Gemini 2.5 Pro/Flash                                       | Multimodal Professional        | 🆕 Added     |
| └─**Custom**    | Any OpenRouter Model ID                                    | Unlimited expansion, personalized selection  | 🆕 Added     |

**🎯 Quick Selection**: 5 quick buttons for popular models | **💾 Persistence**: URL parameter storage, refresh retention | **🔄 Smart Switching**: Switch providers with a single click

### Data Sources & Markets

| Market Type     | Data Source                   | Coverage                                |
| --------------- | ----------------------------- | --------------------------------------- |
| **🇨🇳 A-Shares**  | Tushare, AkShare, TongDaXin | Shanghai and Shenzhen Stock Exchanges, real-time market data, financial reports  |
| **🇭🇰 H-Shares** | AkShare, Yahoo Finance        | Hong Kong Stock Exchange, real-time data, fundamentals    |
| **🇺🇸 US Stocks** | FinnHub, Yahoo Finance        | NYSE, NASDAQ, real-time data             |
| **📰 News**      | Google News                  | Real-time news, multi-language support    |

### Agent Team

**Analyst Team**: 📈 Market Analysis | 💰 Fundamental Analysis | 📰 News Analysis | 💬 Sentiment Analysis
**Research Team**: 🐂 Bullish Researchers | 🐻 Bearish Researchers | 🎯 Trading Decision Maker
**Management**: 🛡️ Risk Manager | 👔 Research Director

## Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone the project
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables
cp .env.example .env
# Edit .env and enter your API keys

# 3. Start the service
docker-compose up -d --build

# 4. Access the application
# Web interface: http://localhost:8501
```

### 💻 Local Deployment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the application
python start_web.py

# 3. Access http://localhost:8501
```

### 📊 Start Analyzing

1.  **Select Model**: DeepSeek V3 / Tongyi Qianwen / Gemini
2.  **Enter Stock Ticker**: `000001` (A-Shares) / `AAPL` (US Stocks) / `0700.HK` (H-Shares)
3.  **Start Analysis**: Click the "🚀 Start Analysis" button
4.  **Monitor Real-time Progress**: Observe the real-time progress and analysis steps.
5.  **View Report**: Click the "📊 View Analysis Report" button
6.  **Export Report**: Supports Word/PDF/Markdown formats.

## 🎯 Key Advantages

*   **🆕 Multi-LLM Integration**: v0.1.11 includes 4 providers and 60+ models for a one-stop AI experience.
*   **💾 Configuration Persistence**: Your model selection is truly persistent, with URL-based storage.
*   **🎯 Fast Switching**: 5 quick select buttons let you instantly switch between different AIs.
*   **📐 Interface Optimization**: Features a 320px sidebar, responsive design, and more efficient space use.
*   **🆕 Real-time Progress**: v0.1.10 asynchronous progress tracking, so you're never left waiting in the dark.
*   **💾 Intelligent Session Management**: Persistent sessions, so you won't lose your analysis results after a refresh.
*   **🇨🇳 Chinese-Optimized**: A-share/H-share data, Chinese LLMs, and a Chinese-language interface.
*   **🐳 Containerized**: One-click Docker deployment for easy environment isolation and scalability.
*   **📄 Professional Reports**: Multi-format export with automated investment recommendations.
*   **🛡️ Reliable**: Multi-layer data sources, intelligent downgrading, and error recovery.

## 📚 Documentation and Support

*   **📖 Full Documentation**: [docs/](./docs/) - Installation guides, usage tutorials, and API documentation.
*   **🚨 Troubleshooting**: [troubleshooting/](./docs/troubleshooting/) - Solutions to common problems.
*   **🔄 Changelog**: [CHANGELOG.md](./docs/releases/CHANGELOG.md) - Detailed version history.
*   **🚀 Quick Start**: [QUICKSTART.md](./QUICKSTART.md) - A 5-minute quick deployment guide.

## 🆚 Chinese-Enhanced Features

**New features compared to the original**: Multi-LLM provider integration | Persistent model selection | Quick select buttons | 320px sidebar | Real-time progress display | Intelligent session management | Chinese interface | A-share data | Chinese LLMs | Docker deployment | Professional report export | Unified log management | Web configuration interface | Cost optimization

**Docker Deployment Includes:**

*   🌐 **Web Application**: TradingAgents-CN main program
*   🗄️ **MongoDB**: Data persistence storage
*   ⚡ **Redis**: High-speed cache
*   📊 **MongoDB Express**: Database management interface
*   🎛️ **Redis Commander**: Cache management interface

---