# 🚀 TradingAgents-CN: 中文金融交易决策框架 (基于 TradingAgents)

**Unlock the power of AI for financial trading with TradingAgents-CN, a Chinese-optimized framework built upon the groundbreaking work of [Tauric Research's TradingAgents](https://github.com/TauricResearch/TradingAgents).**

## ✨ Key Features

*   **🇨🇳 Chinese Language & Market Focus:** Optimized for Chinese users, with full support for A-shares (A股), Hong Kong stocks (港股).
*   **🤖 Advanced LLM Integration:** Seamlessly integrates with multiple LLM providers, including OpenAI and Google AI, plus easy configuration of custom endpoints.
*   **🧠 Intelligent News Analysis:** AI-driven news filtering and sentiment analysis for informed decision-making.
*   **📈 Multi-Agent Architecture:** Four specialized analysts (fundamental, technical, news, and sentiment) and a trader, working collaboratively.
*   **📊 Streamlined Web Interface:** Intuitive web interface (built with Streamlit) for real-time progress tracking and professional report generation.
*   **🐳 Docker Ready:** Deploy quickly and easily with Docker for environment isolation and scalability.

## 🚀 What's New in v0.1.13 - Preview Release

*   **🤖 Native OpenAI Support:**
    *   Customizable OpenAI endpoints for maximum flexibility.
    *   Broad model compatibility (supports any OpenAI-compatible model).
    *   Improved compatibility and performance with a new OpenAI adapter.
    *   Unified endpoint and model configuration management.
*   **🧠 Comprehensive Google AI Integration:**
    *   Full support for the langchain-google-genai, google-generativeai, and google-genai packages.
    *   Access to 9 verified Google AI models, including gemini-2.5-pro, gemini-2.5-flash, etc.
    *   Dedicated Google AI tool call processor.
    *   Intelligent fallback mechanisms.
*   **🔧 Improved LLM Adapter Architecture:**
    *   GoogleOpenAIAdapter enables OpenAI compatibility for Google AI.
    *   Unified calling interface across all LLM providers.
    *   Enhanced error handling and automatic retry mechanisms.
    *   LLM performance monitoring and statistics.
*   **🎨 Intelligent Web Interface Improvements:**
    *   Smart model selection based on availability.
    *   Fixed KeyError issues in model selection.
    *   Faster response times and improved UI experience.
    *   More user-friendly error messages and helpful suggestions.

## 🎯 Core Functionality

*   **Automated Financial Analysis:**
    *   **Stock Analysis:** AAPL, 000001, 0700.HK (and more).
    *   **Research Depth:** 5 levels of analysis from quick overview to in-depth reports.
    *   **Expert Agents:** Fundamental, Technical, News, and Sentiment Analysts, working as a team.
    *   **Buy/Sell/Hold Recommendations:** Clear investment guidance.
    *   **Professional Reports:** Export reports in Markdown, Word, or PDF format.

## 💻 Web Interface

**(See screenshots in original README for visual examples.)**

*   **Smart Configuration Panel:** Supports multi-market stock analysis.
*   **Real-time Progress Tracking:** Visualizes analysis progress.
*   **Professional Analysis Reports:** Multi-dimensional analysis results, export options.

## 🐳 Docker Quickstart

1.  `git clone https://github.com/hsliuping/TradingAgents-CN.git`
2.  `cd TradingAgents-CN`
3.  `cp .env.example .env` (Edit `.env` with your API keys).
4.  `docker-compose up -d --build` (First time or code changes).
    `docker-compose up -d` (Subsequent starts).
5.  Access the web interface at `http://localhost:8501`.

## 📚 Full Documentation

Access comprehensive documentation in the `docs/` directory or at [TradingAgents-CN documentation](docs/README.md), including quickstarts, architecture, configuration, and detailed guides.

## 🔗 Get Started Now!

[View the original repository on GitHub](https://github.com/hsliuping/TradingAgents-CN)