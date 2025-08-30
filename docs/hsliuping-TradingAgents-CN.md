# TradingAgents-CN: 中文金融交易决策框架 🚀

**Unlock the power of AI for Chinese financial markets with TradingAgents-CN, an enhanced and optimized framework for analyzing A-shares, H-shares, and US stocks.**  This project builds upon the groundbreaking work of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents), providing a comprehensive, localized experience with enhanced features and Chinese language support.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

> **✨ v0.1.13 Preview: Native OpenAI & Google AI Integration!**  Experience the future of AI-driven financial analysis with support for custom OpenAI endpoints, nine Google AI models, and optimized LLM adapter architecture.

**Key Features:**

*   🤖 **Native OpenAI Support:** Customize your OpenAI experience.
*   🧠 **Google AI Integration:** Leverage the power of Google's AI ecosystem.
*   🇨🇳 **A-Share, H-Share, & US Stock Support:** Comprehensive market coverage.
*   🌐 **Multi-LLM Provider Support:**  DashScope, DeepSeek, OpenAI, OpenRouter & Google AI.
*   📊 **Professional Report Export:** Generate insightful reports in multiple formats.
*   🐳 **Dockerized Deployment:**  Easy setup and scaling.
*   📰 **Smart News Analysis**: AI-driven news filtering and quality assessment.

## 🎯 Core Functionality

TradingAgents-CN leverages multi-agent large language models to create a sophisticated framework for financial trading decisions, optimized for Chinese users. It provides comprehensive analysis capabilities for A-shares, H-shares, and US stocks.

## 🙏  Acknowledgements

We extend our sincere gratitude to the [Tauric Research](https://github.com/TauricResearch) team for their pioneering work in developing the original [TradingAgents](https://github.com/TauricResearch/TradingAgents) framework.

**Our Mission:** To provide Chinese users with a complete localized experience, supporting the A-share and H-share markets, integrating domestic large language models, and promoting the popularization and application of AI financial technology within the Chinese community.

## 🆕 What's New

### ✨ v0.1.13: OpenAI & Google AI Integration Preview

*   🤖 **Native OpenAI Endpoints:** Support for custom OpenAI-compatible API endpoints.
*   🧠 **Comprehensive Google AI Integration:**  Support for langchain-google-genai, google-generativeai, and google-genai packages, including nine validated models (e.g. gemini-2.5-pro).
*   🔧 **Optimized LLM Adapter Architecture:** Refined architecture with GoogleOpenAIAdapter and unified calling interfaces.
*   🎨 **Smart UI Enhancements:** Intelligent model selection and UI/UX improvements.

### ✨ v0.1.12:  Smart News Analysis & More

*   🧠 **Smart News Analysis Module:** AI-powered news filtering and quality assessment.
*   🔧 **Technical Fixes & Optimizations:** Enhancements to LLM tool calls and more.
*   📚 **Enhanced Testing & Documentation:** Extensive testing and documentation.
*   🗂️ **Project Structure Optimization:** Improved organization for maintainability.

## 🎯 Core Features

### 🤖 Multi-Agent Collaboration

*   **Specialized Analysts:** Fundamental, Technical, News, and Social Media analysts.
*   **Structured Debate:** Bullish and bearish researchers conduct in-depth analysis.
*   **Intelligent Decision-Making:** Traders make final investment recommendations based on combined input.
*   **Risk Management:** Multi-layered risk assessment and management mechanisms.

## 🖥️ Web Interface

### 📸 Screenshots

> 🎨  **Modern Web Interface:** A responsive Streamlit-based web application providing an intuitive stock analysis experience.

**[Insert Image Examples from original repo for the Interface with captions]**

## 🎯 Key Functionalities

#### 📋 Smart Analysis Configuration

*   🌍 **Multi-Market Support:** Analysis for US, A-shares, and H-shares.
*   🎯 **5 Research Depths:** From quick analysis (2 minutes) to in-depth reports (25 minutes).
*   🤖 **Analyst Selection:** Choose market technicians, fundamental analysts, news analysts, and social media analysts.
*   📅 **Flexible Time Settings:** Analyze data from any historical point in time.

#### 🚀 Real-Time Progress Tracking

*   📊 **Visual Progress:** Displays real-time analysis progress and estimated time remaining.
*   🔄 **Intelligent Step Recognition:** Automatically identifies the current analysis stage.
*   ⏱️ **Accurate Time Estimation:** Uses historical data to intelligently calculate time estimates.
*   💾 **Persistent State:** Analysis progress is not lost on page refresh.

#### 📈 Professional Result Display

*   🎯 **Investment Decisions:** Clear buy/hold/sell recommendations.
*   📊 **Multi-Dimensional Analysis:** Comprehensive assessment of technical, fundamental, and news data.
*   🔢 **Quantitative Indicators:** Confidence levels, risk scores, and target price estimates.
*   📄 **Professional Reports:** Export reports in Markdown/Word/PDF formats.

#### 🤖 Multi-LLM Model Management

*   🌐 **4 Providers:** DashScope, DeepSeek, Google AI, and OpenRouter
*   🎯 **60+ Model Choices:** From economy to flagship models.
*   💾 **Persistent Configuration:** URL parameter storage, refresh to maintain settings.
*   ⚡ **Fast Switching:** One-click selection for 5 popular models.

### 🎮 Web Interface Operation Guide

#### 🚀 Quick Start

1.  **Start the application:** `python start_web.py` or `docker-compose up -d`
2.  **Access the interface:** Open `http://localhost:8501` in your browser.
3.  **Configure models:** Select the LLM provider and model from the sidebar.
4.  **Enter the stock:** Enter the stock code (e.g., AAPL, 000001, 0700.HK).
5.  **Choose depth:** Select research depth (1-5 levels).
6.  **Start analysis:** Click the "🚀 Start Analysis" button.
7.  **View results:** Track progress in real-time, and view the analysis report.
8.  **Export the report:** Export professional format reports with one click.

#### 📊 Supported Stock Code Formats

*   🇺🇸 **US Stocks:** `AAPL`, `TSLA`, `MSFT`, `NVDA`, `GOOGL`
*   🇨🇳 **A-Shares:** `000001`, `600519`, `300750`, `002415`
*   🇭🇰 **H-Shares:** `0700.HK`, `9988.HK`, `3690.HK`, `1810.HK`

#### 🎯 Research Depth Explanation

*   **Level 1 (2-4 minutes):** Quick overview, basic technical indicators
*   **Level 2 (4-6 minutes):** Standard analysis, technical + fundamentals
*   **Level 3 (6-10 minutes):** In-depth analysis, incorporating news sentiment ⭐ **Recommended**
*   **Level 4 (10-15 minutes):** Comprehensive analysis, multi-round agent debate
*   **Level 5 (15-25 minutes):** Most in-depth analysis, complete research report

#### 💡 Tips

*   🔄 **Real-time Refresh:** Refresh the page at any time during the analysis without losing progress.
*   📱 **Mobile Adaptive:** Supports mobile and tablet device access.
*   🎨 **Dark Mode:** Automatically adapts to system theme settings.
*   ⌨️ **Shortcuts:** Supports the Enter key for quick analysis submission.
*   📋 **History:** Automatically saves recent analysis configurations.

> 📖 **Detailed Guide:** For complete web interface usage instructions, refer to [🖥️ Web Interface Detailed Guide](docs/usage/web-interface-detailed-guide.md).

## 🎯 Core Features

### 🚀 Smart News Analysis ✨ **v0.1.12 Major Upgrade**

| Feature | Status | Description |
|---|---|---|
| **🧠 Smart News Analysis** | 🆕 v0.1.12 | AI news filtering, quality assessment, relevance analysis |
| **🔧 News Filter** | 🆕 v0.1.12 | Multi-level filtering, basic/enhanced/integrated processing |
| **📰 Unified News Tool** | 🆕 v0.1.12 | Integrates multiple news sources, unified interface, smart retrieval |
| **🤖 Multi-LLM Providers** | 🆕 v0.1.11 | 4 major providers, 60+ models, smart categorization management |
| **💾 Model Selection Persistence** | 🆕 v0.1.11 | URL parameter storage, refresh to maintain, configuration sharing |
| **🎯 Quick Select Buttons** | 🆕 v0.1.11 | One-click switching of popular models, improving operational efficiency |
| **📊 Real-time Progress Display** | ✅ v0.1.10 | Asynchronous progress tracking, intelligent step recognition, accurate time calculation |
| **💾 Smart Session Management** | ✅ v0.1.10 | Persistent state, automatic degradation, cross-page recovery |
| **🎯 One-Click Report View** | ✅ v0.1.10 | View results with one click after analysis, intelligent result recovery |
| **🖥️ Streamlit Interface** | ✅ Full Support | Modern responsive interface, real-time interaction and data visualization |
| **⚙️ Configuration Management** | ✅ Full Support | Web-end API key management, model selection, parameter configuration |

### 🎨 CLI User Experience ✨ **v0.1.9 Optimization**

| Feature | Status | Description |
|---|---|---|
| **🖥️ Interface and Log Separation** | ✅ Full Support | Clean and beautiful user interface, independent management of technical logs |
| **🔄 Smart Progress Display** | ✅ Full Support | Multi-stage progress tracking, preventing repeated prompts |
| **⏱️ Time Estimation Feature** | ✅ Full Support | Intelligent analysis phase display and estimated time consumption |
| **🌈 Rich Color Output** | ✅ Full Support | Color progress indicators, status icons, visual enhancement |

### 🧠 LLM Model Support ✨ **v0.1.13 Full Upgrade**

| Model Provider | Supported Models | Featured Functionality | New Features |
|---|---|---|---|
| **🇨🇳 Alibaba Baichuan** | qwen-turbo/plus/max | Chinese optimization, cost-effectiveness | ✅ Integrated |
| **🇨🇳 DeepSeek** | deepseek-chat | Tool calling, high cost-effectiveness | ✅ Integrated |
| **🌍 Google AI** | **9 Verified Models** | Latest Gemini 2.5 Series | 🆕 Upgraded |
| ├─**Latest Flagship** | gemini-2.5-pro/flash | Latest flagship, super-fast response | 🆕 New |
| ├─**Stable Recommendation** | gemini-2.0-flash | Recommended use, balanced performance | 🆕 New |
| ├─**Classic & Powerful** | gemini-1.5-pro/flash | Classic and stable, high-quality analysis | ✅ Integrated |
| └─**Lightweight & Fast** | gemini-2.5-flash-lite | Lightweight tasks, fast response | 🆕 New |
| **🌐 Native OpenAI** | **Custom Endpoint Support** | Any OpenAI compatible endpoint | 🆕 New |
| **🌐 OpenRouter** | **60+ Model Aggregation Platform** | One API accesses all major models | ✅ Integrated |
| ├─**OpenAI** | o4-mini-high, o3-pro, GPT-4o | Latest o series, professional reasoning | ✅ Integrated |
| ├─**Anthropic** | Claude 4 Opus/Sonnet/Haiku | Top performance, balanced versions | ✅ Integrated |
| ├─**Meta** | Llama 4 Maverick/Scout | Latest Llama 4 Series | ✅ Integrated |
| └─**Custom** | Any OpenRouter Model ID | Infinite expansion, personalized selection | ✅ Integrated |

**🎯 Quick Selection**: 5 Popular Model Buttons | **💾 Persistence**: URL Parameter Storage, Refresh to Maintain | **🔄 Intelligent Switching**: One-Click Switching of Different Providers

### 📊 Data Sources and Markets

| Market Type | Data Source | Coverage |
|---|---|---|
| **🇨🇳 A-Shares** | Tushare, AkShare, Tongdaxin | Shanghai and Shenzhen Stock Exchanges, real-time quotes, financial data |
| **🇭🇰 H-Shares** | AkShare, Yahoo Finance | Hong Kong Stock Exchange, real-time quotes, fundamentals |
| **🇺🇸 US Stocks** | FinnHub, Yahoo Finance | NYSE, NASDAQ, real-time data |
| **📰 News** | Google News | Real-time news, multi-language support |

### 🤖 Agent Team

**Analyst Team:** 📈 Market Analysis | 💰 Fundamental Analysis | 📰 News Analysis | 💬 Sentiment Analysis
**Research Team:** 🐂 Bullish Researcher | 🐻 Bearish Researcher | 🎯 Trading Decision Maker
**Management:** 🛡️ Risk Manager | 👔 Research Director

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone the project
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables
cp .env.example .env
# Edit the .env file, fill in your API keys

# 3. Start the service
# For the first startup or code changes (requires image build)
docker-compose up -d --build

# For daily startup (image already exists, no code changes)
docker-compose up -d

# Smart startup (automatically determines whether a build is needed)
# Windows environment
powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1

# Linux/Mac environment
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

1.  **Select a model:** DeepSeek V3 / Tongyi Qianwen / Gemini
2.  **Enter a stock:** `000001` (A-shares) / `AAPL` (US stocks) / `0700.HK` (H-shares)
3.  **Start Analysis:** Click the "🚀 Start Analysis" button.
4.  **Real-time Tracking:** Observe real-time progress and analysis steps.
5.  **View Report:** Click the "📊 View Analysis Report" button.
6.  **Export Report:** Supports Word/PDF/Markdown format.

## 🎯 Core Advantages

*   🧠 **Smart News Analysis:** AI-driven news filtering and quality assessment (v0.1.12).
*   🔧 **Multi-Level Filtering:** Base, enhanced, and integrated three-level news filtering mechanism.
*   📰 **Unified News Tools:** Integrates multiple news sources, providing a unified smart retrieval interface.
*   🆕 **Multi-LLM Integration:** New 4 major providers, 60+ models, all-in-one AI experience (v0.1.11).
*   💾 **Persistent Configuration:** Truly persistent model selection, URL parameter storage, and refresh to maintain settings.
*   🎯 **Quick Switching:** One-click switching of 5 popular models, different AIs.
*   🆕 **Real-time Progress:** v0.1.10 asynchronous progress tracking, no more black box waiting.
*   💾 **Smart Session:** State persistence, page refresh doesn't lose analysis results.
*   🇨🇳 **China Optimized:** A-share/H-share data + domestic LLMs + Chinese interface.
*   🐳 **Containerized:** Docker one-click deployment, environment isolation, and rapid expansion.
*   📄 **Professional Report:** Multi-format export, automatic generation of investment advice.
*   🛡️ **Stable & Reliable:** Multi-layered data sources, smart degradation, and error recovery.

## 🔧 Technical Architecture

**Core Technologies:** Python 3.10+ | LangChain | Streamlit | MongoDB | Redis
**AI Models:** DeepSeek V3 | Alibaba Baichuan | Google AI | OpenRouter (60+ models) | OpenAI
**Data Sources:** Tushare | AkShare | FinnHub | Yahoo Finance
**Deployment:** Docker | Docker Compose | Local Deployment

## 📚 Documentation and Support

*   📖 **Complete Documentation:** [docs/](./docs/) - Installation guide, usage tutorial, API documentation
*   🚨 **Troubleshooting:** [troubleshooting/](./docs/troubleshooting/) - Common issue solutions
*   🔄 **Change Log:** [CHANGELOG.md](./docs/releases/CHANGELOG.md) - Detailed version history
*   🚀 **Quick Start:** [QUICKSTART.md](./QUICKSTART.md) - 5-minute quick deployment guide

## 🆚 Chinese Enhancement Features

**Compared to the original version:** Smart News Analysis | Multi-level News Filtering | News Quality Assessment | Unified News Tool | Multi-LLM Provider Integration | Model Selection Persistence | Quick Switch Buttons | Real-Time Progress Display | Smart Session Management | Chinese Interface | A-Share Data | Domestic LLMs | Docker Deployment | Professional Report Export | Unified Log Management | Web Configuration Interface | Cost Optimization

**Docker deployment services:**

*   🌐 **Web Application:** TradingAgents-CN main program
*   🗄️ **MongoDB:** Data persistence storage
*   ⚡ **Redis:** High-speed cache
*   📊 **MongoDB Express:** Database management interface
*   🎛️ **Redis Commander:** Cache management interface

#### 💻 Method 2: Local Deployment

**Applicable scenarios:** development environment, custom configuration, offline use

### Environment Requirements

*   Python 3.10+ (recommended 3.11)
*   4GB+ RAM (8GB+ recommended)
*   Stable network connection

### Installation Steps

```bash
# 1. Clone the project
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Create a virtual environment
python -m venv env
# Windows
env\Scripts\activate
# Linux/macOS
source env/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install all dependencies
pip install -r requirements.txt
# Or use pip install -e .
pip install -e .

# Note: requirements.txt already contains all necessary dependencies:
# - Database support (MongoDB + Redis)
# - Multi-market data sources (Tushare, AKShare, FinnHub, etc.)
# - Web interface and report export functionality
```

### Configure API Keys

#### 🇨🇳 Recommended: Use Alibaba Baichuan (domestic large model)

```bash
# Copy the configuration template
cp .env.example .env

# Edit the .env file, configure the following required API keys:
DASHSCOPE_API_KEY=your_dashscope_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here

# Recommended: Tushare API (professional A-share data)
TUSHARE_TOKEN=your_tushare_token_here
TUSHARE_ENABLED=true

# Optional: Other AI model APIs
GOOGLE_API_KEY=your_google_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Database configuration (optional, improve performance)
# Local deployment uses standard ports
MONGODB_ENABLED=false  # Set to true to enable MongoDB
REDIS_ENABLED=false    # Set to true to enable Redis
MONGODB_HOST=localhost
MONGODB_PORT=27017     # Standard MongoDB port
REDIS_HOST=localhost
REDIS_PORT=6379        # Standard Redis port

# Docker deployment requires modification of the hostname
# MONGODB_HOST=mongodb
# REDIS_HOST=redis
```

#### 📋 Deployment Mode Configuration Instructions

**Local deployment mode**:

```bash
# Database configuration (local deployment)
MONGODB_ENABLED=true
REDIS_ENABLED=true
MONGODB_HOST=localhost      # Local host
MONGODB_PORT=27017         # Standard port
REDIS_HOST=localhost       # Local host
REDIS_PORT=6379           # Standard port
```

**Docker deployment mode**:

```bash
# Database configuration (Docker deployment)
MONGODB_ENABLED=true
REDIS_ENABLED=true
MONGODB_HOST=mongodb       # Docker container service name
MONGODB_PORT=27017        # Standard port
REDIS_HOST=redis          # Docker container service name
REDIS_PORT=6379          # Standard port
```

> 💡 **Configuration Tips**:
>
> - Local deployment: You need to manually start MongoDB and Redis services
> - Docker deployment: Database services are automatically started through docker-compose
> - Port conflict: If you already have database services locally, you can modify the port mapping in docker-compose.yml

#### 🌍 Optional: Use Foreign Models

```bash
# OpenAI (requires scientific internet access)
OPENAI_API_KEY=your_openai_api_key

# Anthropic (requires scientific internet access)
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 🗄️ Database Configuration (MongoDB + Redis)

#### High-Performance Data Storage Support

This project supports **MongoDB** and **Redis** databases, providing:

-   📊 **Stock data caching**: Reduce API calls, improve response speed
-   🔄 **Intelligent downgrade mechanism**: Multi-layer data source of MongoDB → API → local cache
-   ⚡ **High-performance caching**: Redis caches hot data, millisecond response
-   🛡️ **Data persistence**: MongoDB stores historical data, supports offline analysis

#### Database Deployment Method

**🐳 Docker deployment (Recommended)**

If you use Docker deployment, the database is already included:

```bash
# Docker deployment automatically starts all services, including:
docker-compose up -d --build
# - Web Application (port 8501)
# - MongoDB (port 27017)
# - Redis (port 6379)
# - Database management interface (ports 8081, 8082)
```

**💻 Local deployment - Database Configuration**

If you use local deployment, you can choose one of the following methods:

**Method 1: Only start the database service**

```bash
# Only start MongoDB + Redis services (without starting the Web application)
docker-compose up -d mongodb redis mongo-express redis-commander

# View service status
docker-compose ps

# Stop services
docker-compose down
```

**Method 2: Complete local installation**

```bash
# Database dependencies are already included in requirements.txt, no additional installation is needed

# Start MongoDB (default port 27017)
mongod --dbpath ./data/mongodb

# Start Redis (default port 6379)
redis-server
```

> ⚠️ **Important Notes**:
>
> - **🐳 Docker deployment**: The database is automatically included, no additional configuration is required
> - **💻 Local deployment**: You can choose to start only the database service or complete local installation
> - **📋 Recommended**: Use Docker deployment to get the best experience and consistency

#### Database Configuration Options

**Environment variable configuration** (Recommended):

```bash
# MongoDB configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=trading_agents
MONGODB_USERNAME=admin
MONGODB_PASSWORD=your_password

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_DB=0
```

**Configuration file method**:

```python
# config/database_config.py
DATABASE_CONFIG = {
    'mongodb': {
        'host': 'localhost',
        'port': 27017,
        'database': 'trading_agents',
        'username': 'admin',
        'password': 'your_password'
    },
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'password': 'your_redis_password',
        'db': 0
    }
}
```

#### Database Feature Characteristics

**MongoDB Features**:

-   ✅ Stock basic information storage
-   ✅ Historical price data cache
-   ✅ Analysis result persistence
-   ✅ User configuration management
-   ✅ Automatic data synchronization

**Redis Features**:

-   ⚡ Real-time price data caching
-   ⚡ API response result caching
-   ⚡ Session state management
-   ⚡ Hot data preloading
-   ⚡ Distributed lock support

#### Smart Downgrade Mechanism

The system adopts a multi-layer data source degradation strategy to ensure high availability:

```
📊 Data acquisition process:
1.  🔍 Check Redis cache (milliseconds)
2.  📚 Query MongoDB storage (seconds)
3.  🌐 Call TongdaXin API (seconds)
4.  💾 Local file cache (backup)
5.  ❌ Return error message
```

**Configure Degradation Strategy**:

```python
# Configure in the .env file
ENABLE_MONGODB=true
ENABLE_REDIS=true
ENABLE_FALLBACK=true

# Cache expiration time (seconds)
REDIS_CACHE_TTL=300
MONGODB_CACHE_TTL=3600
```

#### Performance Optimization Suggestions

**Production Environment Configuration**:

```bash
# MongoDB Optimization
MONGODB_MAX_POOL_SIZE=50
MONGODB_MIN_POOL_SIZE=5
MONGODB_MAX_IDLE_TIME=30000

# Redis Optimization
REDIS_MAX_CONNECTIONS=20
REDIS_CONNECTION_POOL_SIZE=10
REDIS_SOCKET_TIMEOUT=5
```

#### Database Management Tools

```bash
# Initialize the database
python scripts/setup/init_database.py

# System status check
python scripts/validation/check_system_status.py

# Cleanup cache tool
python scripts/maintenance/cleanup_cache.py --days 7
```

#### Troubleshooting

**Common Problem Solving**:

1.  🪟 **Windows 10 ChromaDB Compatibility Issue**

    **Problem Description**: A `Configuration error: An instance of Chroma already exists for ephemeral with different settings` error occurs on Windows 10, while Windows 11 is normal.

    **Quick Solution**:

    ```bash
    # Solution 1: Disable memory function (recommended)
    # Add to the .env file:
    MEMORY_ENABLED=false

    # Solution 2: Use a dedicated repair script
    powershell -ExecutionPolicy Bypass -File scripts\fix_chromadb_win10.ps1

    # Solution 3: Run with administrator privileges
    # Right-click PowerShell -> "Run as administrator"
    ```

    **Detailed solution**: Refer to [Windows 10 Compatibility Guide](docs/troubleshooting/windows10-chromadb-fix.md)
2.  **MongoDB Connection Failed**

    **Docker Deployment**:

    ```bash
    # Check service status
    docker-compose logs mongodb

    # Restart the service
    docker-compose restart mongodb
    ```

    **Local Deployment**:

    ```bash
    # Check MongoDB process
    ps aux | grep mongod

    # Restart MongoDB
    sudo systemctl restart mongod # Linux
    brew services restart mongodb # macOS
    ```
3.  **Redis Connection Timeout**

    ```bash
    # Check Redis status
    redis-cli ping

    # Clear Redis cache
    redis-cli flushdb
    ```
4.  **Cache Issues**

    ```bash
    # Check system status and cache
    python scripts/validation/check_system_status.py

    # Clear expired cache
    python scripts/maintenance/cleanup_cache.py --days 7
    ```

> 💡 **Tips**: Even without configuring the database, the system can still run normally, and will automatically degrade to the API direct call mode. Database configuration is an optional performance optimization feature.

> 📚 **Detailed Documents**: For more database configuration information, please refer to [Database Architecture Documentation](docs/architecture/database-architecture.md)

### 📤 Report Export Functionality

#### New Feature: Professional Analysis Report Export

This project now supports exporting stock analysis results to various professional formats:

**Supported export formats**:

-   📄 **Markdown (.md)** - Lightweight markup language, suitable for technical users and version control
-   📝 **Word (.docx)** - Microsoft Word documents, suitable for business reports and further editing
-   📊 **PDF (.pdf)** - Portable Document Format, suitable for formal sharing and printing

**Report Content Structure**:

-   🎯 **Investment Decision Summary** - Buy/hold/sell recommendations, confidence level, risk score
-   📊 **Detailed Analysis Report** - Technical analysis, fundamental analysis, market sentiment, news events
-   ⚠️ **Risk Disclosure** - Complete investment risk statements and disclaimers
-   📋 **Configuration Information** - Analysis parameters, model information, generation time

**How to use**:

1.  After completing the stock analysis, find the "📤 Export Report" section at the bottom of the results page
2.  Select the desired format: Markdown, Word, or PDF
3.  Click the export button, and the system automatically generates and provides the download

**Install export dependencies**:

```bash
# Install Python dependencies
pip install markdown pypandoc

# Install system tools (for PDF export)
# Windows: choco install pandoc wkhtmltopdf
# macOS: brew install pandoc wkhtmltopdf
# Linux: sudo apt-get install pandoc wkhtmltopdf
```

> 📚 **Detailed Documents**: For the complete export function usage guide, please refer to [Export Function Guide](docs/EXPORT_GUIDE.md)

### 🚀 Start the Application

#### 🐳 Docker Startup (Recommended)

If you are using Docker deployment, the application has already started automatically:

```bash
# The application is already running in Docker, access directly:
# Web interface: http://localhost:8501
# Database management: http://localhost:8081
# Cache management: http://localhost:8082

# View running status
docker-compose ps

# View logs
docker-compose logs -f web
```

#### 💻 Local Startup

If you are using local deployment:

```bash
# 1. Activate the virtual environment
# Windows
.\env\Scripts\activate
# Linux/macOS
source env/bin/activate

# 2. Install the project into the virtual environment (Important!)
pip install -e .

# 3. Start the Web management interface
# Method 1: Use the project startup script (recommended)
python start_web.py

# Method 2: Use the original startup script
python web/run_web.py

# Method 3: Use streamlit directly (requires the project to be installed first)
streamlit run web/app.py
```

Then access `http://localhost:8501` in your browser

**Web Interface Feature Functions**:

-   🇺🇸 **US Stock Analysis**: Supports AAPL, TSLA, NVDA and other US stock codes
-   🇨🇳 **A-Share Analysis**: Supports 000001, 600519, 300750 and other A-share codes
-   📊 **Real-time Data**: TongdaXin API provides real-time A-share market data
-   🤖 **Agent Selection**: Different analyst combinations can be selected
-   📤 **Report Export**: One-click export of professional analysis reports in Markdown/Word/PDF formats
-   🎯 **5 Research Depths**: From quick analysis (2-4 minutes) to comprehensive analysis (15-25 minutes)
-   📊 **Smart Analyst Selection**: Market technology, fundamental analysis, news, and social media analysts
-   🔄 **Real-time Progress Display**: Visualize the analysis process and avoid waiting anxiety
-   📈 **Structured Results**: Investment recommendations, target prices, confidence levels, risk assessment
-   🇨🇳 **Fully Chinese**: The interface and analysis results are displayed in Chinese

**Research Depth Level Explanation**:

-   **Level 1 - Quick Analysis** (2-4 minutes): Daily monitoring, basic decision-making
-   **Level 2 - Basic Analysis** (4-6 minutes): Routine investment, balance speed
-   **Level 3 - Standard Analysis** (6-10 minutes): Important decision-making, recommended by default
-   **Level 4 - In-depth Analysis** (10-15 minutes): Major investment, detailed research
-   **Level 5 - Comprehensive Analysis** (15-25 minutes): Most important decisions, most comprehensive analysis

#### 💻 Code Invocation (Suitable for Developers)

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Configure Alibaba Baichuan
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "dashscope"
config["deep_think_llm"] = "qwen-plus"  # Deep analysis
config["quick_think_llm"] = "qwen-turbo"  # Fast tasks

# Create trading agents
ta = TradingAgentsGraph(debug=True, config=config)

# Analyze stocks (taking Apple Inc. as an example)
state, decision = ta.propagate("AAPL", "2024-01-15")

# Output analysis results
print(f"Recommended Action: {decision['action']}")
print(f"Confidence Level: {decision['confidence']:.1%}")
print(f"Risk Score: {decision['risk_score']:.1%}")
print(f"Reasoning Process: {decision['reasoning']}")
```

#### Quick Start Script

```bash
# Alibaba Baichuan demo (recommended for Chinese users)
python examples/dashscope/demo_dashscope_chinese.py

# Alibaba Baichuan complete demo
python examples/dashscope/demo_dashscope.py

# Alibaba Baichuan simplified test
python examples/dashscope/demo_dashscope_simple.py

# OpenAI demo (requires a foreign API)
python examples/openai/demo_openai.py

# Integration test
python tests/integration/test_dashscope_integration.py
```

#### 📁 Data Directory Configuration

**New Feature**: Flexible configuration of data storage paths, supporting multiple configuration methods:

```bash
# View current data directory configuration
python -m cli.main data-config --show

# Set a custom data directory
python -m cli.main data-config --set /path/to/your/data

# Reset to default configuration
python -m cli.main data-config --reset
```

**Environment variable configuration**:

```bash
# Windows
set TRADING_AGENTS_DATA_DIR=C:\MyTradingData

# Linux/macOS
export TRADING_AGENTS_DATA_DIR=/home/user/trading_data
```

**Programmatic configuration**:

```python
from tradingagents.config_manager import ConfigManager

# Set the data directory
config_manager = ConfigManager()
config_manager.set_data_directory("/path/to/data")

# Get configuration
data_dir = config_manager.get_data_directory()
print(f"Data Directory: {data_dir}")
```

**Configuration Priority**: Program setting > Environment variable > Configuration file > Default value

For detailed information, please refer to: [📁