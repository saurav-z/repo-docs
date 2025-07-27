# 🚀 TradingAgents-CN: 中文金融交易决策框架

**Unlock the power of AI in financial trading with TradingAgents-CN, an enhanced,中文-optimized framework built upon the groundbreaking work of [Tauric Research](https://github.com/TauricResearch/TradingAgents).** This project provides a complete AI-driven solution for analyzing the A-share, Hong Kong, and US stock markets, empowering you with intelligent trading insights and automated reports.

[<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">](https://opensource.org/licenses/Apache-2.0)
[<img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python">](https://www.python.org/)
[<img src="https://img.shields.io/badge/Version-cn--0.1.10-green.svg" alt="Version">](./VERSION)
[<img src="https://img.shields.io/badge/Docs-中文文档-green.svg" alt="Documentation">](./docs/)
[<img src="https://img.shields.io/badge/Based%20on-TauricResearch%2FTradingAgents-orange.svg" alt="Based on">](https://github.com/TauricResearch/TradingAgents)

## ✨ Key Features

*   **🇨🇳 Enhanced for Chinese Users**: Optimized for A-share, Hong Kong, and US stock market analysis with a fully localized experience.
*   **🤖 Multi-Agent Architecture**:  Four analysts (Fundamental, Technical, News, Sentiment) collaborate with Bull/Bear researchers and a Trader for comprehensive analysis.
*   **📈 Real-time Progress Display**:  New in v0.1.10: Track analysis steps and progress with AsyncProgressTracker.
*   **💾 Intelligent Session Management**: New in v0.1.10: Session persistence and automated fallback.
*   **🚀 Seamless Web Interface**: v0.1.10 upgrade: Streamlined UI, responsive design, and improved error handling.
*   **🐳 Docker Deployment**: Simplify setup and ensure consistent environments.
*   **📄 Professional Report Generation**:  Generate insightful reports in Word, PDF, and Markdown formats.
*   **🧠 Native LLM Support**:  Integrates with DeepSeek V3, Alibaba's Qwen, Google AI, and OpenAI models.

## 🆕 What's New in v0.1.10?

*   **🚀 Real-time progress display**: Async progress tracking and smart time calculations.
*   **💾 Intelligent Session Management**: State persistence and automatic fallback mechanisms.
*   **🎨 Optimized User Experience**: Simplified UI and error handling improvements.

## 🎯 Core Capabilities

*   **Multi-Agent Collaboration:**  Analysts specialize in Fundamental, Technical, News, and Sentiment analysis.
*   **Structured Debate:**  Bull and Bear researchers provide in-depth analysis.
*   **Intelligent Decision-Making:**  A Trader makes final investment recommendations based on all inputs.
*   **Risk Management:**  Multi-layered risk assessment and management mechanisms.

## 💻 Get Started

### 🐳 Docker (Recommended)

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/hsliuping/TradingAgents-CN.git
    cd TradingAgents-CN
    ```

2.  **Configure environment variables:**

    ```bash
    cp .env.example .env
    # Edit .env with your API keys.
    ```

3.  **Build and run:**

    ```bash
    docker-compose up -d --build
    ```

4.  **Access the web interface:**  `http://localhost:8501`

### 💻 Local Deployment

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**

    ```bash
    python start_web.py
    ```

3.  **Access the web interface:**  `http://localhost:8501`

**[Visit the original repository](https://github.com/hsliuping/TradingAgents-CN) for detailed instructions and further information.**