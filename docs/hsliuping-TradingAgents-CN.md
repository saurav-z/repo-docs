# TradingAgents-CN: 中文金融交易决策框架 🚀

> Unleash the power of AI for financial analysis with TradingAgents-CN, a cutting-edge framework optimized for Chinese markets, offering comprehensive A-share, Hong Kong stock, and US stock analysis.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**[Explore the original repo](https://github.com/hsliuping/TradingAgents-CN)**

---

## Key Features 🌟

*   **Comprehensive Chinese Market Support:** Analyze A-shares, Hong Kong stocks, and US stocks.
*   **Multi-Agent Architecture:** Leverages specialized AI agents for in-depth analysis.
*   **AI-Powered News Analysis:** Includes intelligent news filtering and quality assessment.
*   **Native OpenAI & Google AI Integration:** Supports custom endpoints and a wide array of models.
*   **Flexible LLM Provider Support:** Offers diverse LLM options for analysis.
*   **User-Friendly Web Interface:** Provides an intuitive platform for analysis and reporting.
*   **Professional Report Export:** Generates reports in Markdown, Word, and PDF formats.
*   **Dockerized Deployment:** Simplifies setup and scaling.

---

## What's New in v0.1.13 (Preview) 🚀

*   **Native OpenAI Integration:** Custom endpoint support, model flexibility, and a new adapter.
*   **Google AI Ecosystem Integration:** Support for multiple Google AI models and tools.
*   **Enhanced LLM Adapter Architecture:** Unified interfaces, improved error handling, and performance monitoring.
*   **Smart Web Interface Optimization:** Automated model selection, improved UI, and enhanced error messaging.

---

## Core Functionality 🎯

TradingAgents-CN employs a multi-agent, Large Language Model (LLM)-based architecture to provide comprehensive financial analysis. Key features include:

*   **Multi-Agent Collaboration:** Specialized agents for fundamental, technical, news, and sentiment analysis.
*   **Structured Debate:** Bullish and bearish researchers offer in-depth analysis.
*   **Intelligent Decision-Making:** Traders make investment recommendations based on all inputs.
*   **Risk Management:** Multi-level risk assessment and management mechanisms.
*   **Streamlit Web Interface**: Interactive platform for configuration, real-time analysis, and reporting.

---

## Getting Started 🚀

### 🐳 Docker Deployment (Recommended)

1.  **Clone the repository:** `git clone https://github.com/hsliuping/TradingAgents-CN.git` & `cd TradingAgents-CN`
2.  **Configure environment variables:**  `cp .env.example .env` and edit the `.env` file with your API keys.
3.  **Start the service:**  `docker-compose up -d --build` (first-time setup or code changes) or `docker-compose up -d` (for regular use).
4.  **Access the application:**  Open your web browser to `http://localhost:8501`

### 💻 Local Deployment

1.  **Upgrade pip:** `python -m pip install --upgrade pip`
2.  **Install dependencies:** `pip install -e .`
3.  **Start the application:** `python start_web.py`
4.  **Access the web interface:** Open your web browser to `http://localhost:8501`

---

## Core Features in Detail 💡

### 🤖 Multi-Agent System

*   **Specialized Analysts:** Fundamental, Technical, News, and Sentiment analysis agents.
*   **Bulls and Bears:**  Bullish/Bearish researchers perform deep dives.
*   **Trader Agent:** Makes final investment decisions based on all information.
*   **Risk Management:** Ensures compliance and minimizes risk.

### 🖥️ Web Interface

[Include 1-2 of the most visually appealing and informative screenshots here.  Provide short captions.]

### 📈 Analysis Results & Reporting

*   **Actionable Recommendations:**  Clear buy, hold, or sell suggestions.
*   **Multi-Dimensional Analysis:**  Technical, fundamental, and news-based evaluations.
*   **Quantitative Metrics:** Confidence levels, risk scores, and target prices.
*   **Professional Reports:**  Export results in Markdown, Word, and PDF formats.

### 🤖 Model Management

*   **Diverse Providers:** Supports DeepSeek, Ali Baichuan, Google AI, OpenRouter and native OpenAI.
*   **Extensive Model Selection:**  Choose from a wide variety of models.
*   **Persistent Configuration:**  URL parameter-based settings for ease of sharing and reuse.
*   **Rapid Switching:** Quick access to favored models.

---

## Data Sources and Supported Markets 📊

*   **A-Shares:** Tushare, AkShare, and TongdaXin.
*   **Hong Kong Stocks:** AkShare and Yahoo Finance.
*   **US Stocks:** FinnHub and Yahoo Finance.
*   **News:** Google News and other sources.

---

## Documentation and Support 📚

*   **Detailed Documentation:** [./docs/](docs/) - Comprehensive guides, tutorials, and API references.
*   **Troubleshooting:**  [./docs/troubleshooting/](docs/troubleshooting/) - Solutions to common issues.
*   **Release Notes:** [CHANGELOG.md](./docs/releases/CHANGELOG.md) - Detailed version history and updates.

---

## Contributing 🤝

Contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License 📄

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

**If this project helps you, please consider giving it a star!** ⭐
```

Key improvements:

*   **SEO Optimization:**  Keywords like "financial analysis," "AI," "A-share," "Hong Kong stock," and "US stock" are integrated throughout.
*   **Clear Structure:** Headings, bullet points, and concise paragraphs make the information easy to scan.
*   **Concise Language:**  Replaced wordy phrases with more impactful statements.
*   **Actionable Content:**  Focuses on what the tool *does* and how to use it.
*   **Call to Action:**  Encourages users to explore the documentation and give the project a star.
*   **Prioritized Key Info:** Highlights the most important features first.
*   **Clearer Instructions:** Streamlined "Getting Started" sections.
*   **Reduced Redundancy:**  Eliminated repeated explanations.
*   **Visual Appeal:**  Added emojis to emphasize key points.  The inclusion of actual screenshots is extremely valuable for the user.
*   **Focus on Benefits:**  Emphasizes what the user *gains* from using the tool.  The "Key Features" section clearly states the value proposition.
*   **Improved Organization:**  Better use of headings and subheadings to guide the reader.
*   **Community Building:** Includes information about contributing.
*   **Emphasis on Chinese Users:** Throughout the text, the focus is kept on Chinese market features.
*   **Original Repo Link:** The link to the original repository has been clearly displayed.