# 🚀 TradingAgents-CN: 中文金融AI交易决策框架

**Unlock the power of AI for Chinese financial markets with TradingAgents-CN, a cutting-edge framework built for in-depth stock analysis and informed trading decisions.  Based on the revolutionary [TradingAgents](https://github.com/TauricResearch/TradingAgents) and optimized for the Chinese market!**

## 🔑 Key Features

*   **🇨🇳 Chinese-Optimized:** Full A-share (A股), Hong Kong Stock (港股) support, and optimized for Chinese users.
*   **🤖 Multi-Agent Architecture:** Leverage specialized AI agents for fundamental, technical, news, and sentiment analysis.
*   **🚀 Real-time Progress Display:** v0.1.10 introduces asynchronous progress tracking for a seamless experience.
*   **💾 Intelligent Session Management:**  Preserve analysis state and reports across sessions.
*   **📊 Professional Report Export:** Generate insightful reports in Word, PDF, and Markdown formats.
*   **🐳 Docker Deployment:**  Easy one-command setup for rapid deployment and scalability.
*   **🧠 LLM Integration:** Supports leading LLMs including DeepSeek, Ali Tongyi Qianwen, Google AI, and OpenAI.

**[Visit the original repository](https://github.com/hsliuping/TradingAgents-CN) to get started!**

## ✨ What's New in v0.1.10?

*   **🚀 Real-time Progress Display:**
    *   Asynchronous progress tracking for transparent analysis.
    *   Accurate time calculation for better insight.
    *   Multiple display modes (Streamlit, static, unified).
*   **📊 Intelligent Session Management:**
    *   Persist analysis state and reports on page refresh.
    *   Automatic fallback to file storage if Redis is unavailable.
    *   "View Analysis Report" button after analysis completion.
*   **🎨 Enhanced User Experience:**
    *   Simplified interface, removing redundant buttons for clarity.
    *   Responsive design for mobile and various screen sizes.
    *   Improved error handling and user-friendly error messages.

## 🎯 Core Features

### 🤖 Multi-Agent Collaboration

*   **Specialized Analysts:**  Fundamental, Technical, News, and Sentiment analysts.
*   **Structured Debate:** Bullish and bearish researchers provide in-depth analysis.
*   **Intelligent Decision-Making:** Trader agent formulates investment recommendations based on all inputs.
*   **Risk Management:** Multi-layered risk assessment and management mechanisms.

## 🚀 Get Started Quickly

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables
cp .env.example .env
# Edit .env to include your API keys.

# 3. Start the service
docker-compose up -d --build

# 4. Access the application
# Web Interface: http://localhost:8501
```

### 💻 Local Deployment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the application
python start_web.py

# 3. Access the application at http://localhost:8501
```

### 📊 Analysis Steps

1.  **Select Model:** DeepSeek V3 / Ali Tongyi Qianwen / Gemini
2.  **Input Stock Symbol:** `000001` (A-Share) / `AAPL` (US Stock) / `0700.HK` (HK Stock)
3.  **Start Analysis:** Click the "🚀 Start Analysis" button
4.  **Real-time Tracking:** Monitor the progress and steps.
5.  **View Report:** Click the "📊 View Analysis Report" button
6.  **Export Report:** Supports Word/PDF/Markdown formats

## 🎯 Key Advantages

*   **🆕 Real-time Progress:** v0.1.10 adds asynchronous progress tracking.
*   **💾 Intelligent Sessions:** Session persistence ensures analysis results aren't lost.
*   **🇨🇳 China-Optimized:** A-Share/HK Stock data, Chinese UI, and local LLMs.
*   **🐳 Containerization:** Docker for one-click deployment, easy scalability.
*   **📄 Professional Reports:** Export to multiple formats, auto-generated investment advice.
*   **🛡️ Robustness:** Multi-layer data sources, intelligent fallback, and error recovery.

## ⚙️ Technical Architecture

**Core Technologies:** Python 3.10+ | LangChain | Streamlit | MongoDB | Redis
**AI Models:** DeepSeek V3 | Ali Tongyi Qianwen | Google AI | OpenAI
**Data Sources:** Tushare | AkShare | FinnHub | Yahoo Finance
**Deployment:** Docker | Docker Compose | Local Deployment

## 📚 Documentation & Support

*   **📖 Complete Documentation:** [docs/](./docs/) - Installation, usage tutorials, and API documentation.
*   **🚨 Troubleshooting:** [troubleshooting/](./docs/troubleshooting/) - Solutions to common issues.
*   **🔄 Changelog:** [CHANGELOG.md](./docs/releases/CHANGELOG.md) - Detailed version history.
*   **🚀 Quick Start:** [QUICKSTART.md](./QUICKSTART.md) - 5-minute deployment guide.

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file.

---

<div align="center">

**🌟 If you find this project helpful, please give us a Star!**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)

</div>