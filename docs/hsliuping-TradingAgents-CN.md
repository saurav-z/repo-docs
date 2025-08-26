# TradingAgents-CN:  🚀  AI-Powered Financial Trading for Chinese Markets

**Empower your trading strategy with AI!** TradingAgents-CN is a **Chinese-language enhanced financial trading decision-making framework** based on multi-agent large language models. Designed for Chinese users, it provides comprehensive analysis capabilities for **China A-shares, Hong Kong stocks, and US stocks**, with native OpenAI and Google AI integrations.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

[**View the Original Repo: TauricResearch/TradingAgents**](https://github.com/TauricResearch/TradingAgents)

## ✨ Key Features:

*   **Native OpenAI & Google AI Integration:** Utilizing the latest advancements in LLMs.
*   **Comprehensive Market Coverage:** Analyze China A-shares, Hong Kong stocks, and US stocks.
*   **Multi-Agent Architecture:** Fundamental, Technical, News, and Sentiment Analysts collaborate.
*   **Web Interface:** Intuitive Streamlit-based UI for easy stock analysis.
*   **Report Generation:** Generate professional reports in Markdown, Word, and PDF formats.
*   **Docker Deployment:** Easy setup with containerization.
*   **Chinese Language Support:** Optimized for Chinese users and markets.
*   **LLM Provider Flexibility:** Supports multiple LLM providers including OpenAI, Google AI, and more!

## 🚀 What's New in cn-0.1.13-preview:

*   **🚀  Native OpenAI Integration**
    *   Custom OpenAI endpoint support
    *   Flexible model selection: Use any OpenAI-compatible model
    *   Improved compatibility and performance via the native OpenAI adapter
    *   Unified endpoint and model configuration system
*   **🧠 Google AI Ecosystem Integration**
    *   Three Google AI package support
    *   Nine validated models like Gemini 2.5 Pro/Flash
    *   Google AI tool processor
    *   Smart fallback mechanism
*   **🔧 Optimized LLM Adapter Architecture**
    *   Google AI OpenAI adapter
    *   Unified invocation interfaces
    *   Enhanced error handling
    *   Performance monitoring
*   **🎨 Intelligent Web Interface Enhancements**
    *   Automatic model selection
    *   KeyError fix
    *   UI optimization
    *   Improved error messages

## 🎯 Core Features:

*   **Multi-Agent Collaboration:**
    *   Specialized analysts for fundamental, technical, news, and social media analysis.
    *   Bullish/Bearish analysts for in-depth analysis.
    *   Trader makes final investment recommendations based on all inputs.
    *   Multi-layered risk assessment and management.
*   **Web Interface (Streamlit):**
    *   Modern and responsive web interface built with Streamlit.
    *   Provides an intuitive stock analysis experience.
    *   Real-time progress tracking.
    *   Professional investment reports with multi-dimensional results.
    *   Support for various stock codes.

## 💻 Quick Start (Docker Recommended):

```bash
# 1. Clone the repository
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables
cp .env.example .env
# Edit .env file and fill in API keys.

# 3. Start the service
docker-compose up -d --build # First time or after code changes
# OR
docker-compose up -d # For daily use (no code changes)

# 4. Access the application
# Web Interface: http://localhost:8501
```

## 📝  Documentation:

*   **Comprehensive Chinese Documentation:**  Explore detailed guides, tutorials, and API references in our comprehensive Chinese-language documentation.
*   [View Documentation](./docs/)

---

**Disclaimer:** *This framework is for research and educational purposes only and does not constitute financial advice. Investing in the stock market involves risk, and decisions should be made with caution.  Always consult with a financial professional.*