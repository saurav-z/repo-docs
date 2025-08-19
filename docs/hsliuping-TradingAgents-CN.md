# TradingAgents-CN: 中文金融交易决策框架 (增强版) 🚀

**Unleash the power of AI for financial trading with TradingAgents-CN, a Chinese-optimized framework built on multi-agent LLMs, providing comprehensive A-share, H-share, and US stock analysis.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

> **🌟 Key Feature Highlights:**
>
> *   Native OpenAI Support with Custom Endpoints
> *   Comprehensive Google AI Integration
> *   Intelligent Model Selection and LLM Provider Support
> *   Web UI for Real-time Analysis and Reporting
> *   Complete A-Share/H-Share/US Stock Coverage
>
> 🚀 **Latest Release: cn-0.1.13-preview** -  Preview version with native OpenAI support and full Google AI integration!  Includes custom OpenAI endpoint configuration, 9 Google AI models, and LLM adapter architecture optimizations!

---

## 🔑 Core Features

*   **Multi-Agent Architecture:**  Leverages specialized agents for fundamental, technical, news, and sentiment analysis.
*   **Buy/Sell/Hold Recommendations:** Provides clear and concise investment advice.
*   **Comprehensive Market Coverage:** Analyzes A-shares, H-shares, and US stocks.
*   **Web-Based Interface:** User-friendly Streamlit-based interface for easy access and analysis.
*   **Professional Reporting:** Generates detailed investment reports in Markdown, Word, and PDF formats.
*   **LLM Agnostic:** Supports multiple LLM providers, including OpenAI, Google AI, and more.

---

## ✨ What's New in v0.1.13 - Major Updates

### 🤖 Native OpenAI Support

*   **Custom OpenAI Endpoints**: Configure any OpenAI-compatible API endpoint.
*   **Flexible Model Selection**: Utilize any OpenAI-formatted model, not just official ones.
*   **Smart Adapter**:  New native OpenAI adapter provides better compatibility and performance.
*   **Configuration Management**: Unified endpoint and model configuration system.

### 🧠 Full Google AI Ecosystem Integration

*   **Three Google AI Package Support**: langchain-google-genai, google-generativeai, google-genai
*   **9 Verified Models**: Latest models like gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash
*   **Google Tools Processor**: Specialized Google AI tool calling processor.
*   **Intelligent Fallback Mechanism**: Automatically downgrades to basic functionality when advanced features fail.

### 🔧 LLM Adapter Architecture Optimization

*   **GoogleOpenAIAdapter**: New OpenAI-compatible adapter for Google AI.
*   **Unified Interface**: All LLM providers use a consistent calling interface.
*   **Enhanced Error Handling**: Improved exception handling and automatic retry mechanisms.
*   **Performance Monitoring**: Adds LLM call performance monitoring and statistics.

### 🎨 Web Interface Smart Optimizations

*   **Intelligent Model Selection**: Automatically selects the best model based on availability.
*   **KeyError Fix**:  Resolves KeyError issues in model selection.
*   **UI Response Optimization**: Improves model switching responsiveness and user experience.
*   **Error Prompt**:  More user-friendly error messages and suggestions for solutions.

---

## 🆕 Key Features in v0.1.12

### 🧠 Intelligent News Analysis Module

*   **AI-Powered News Filtering**: AI-based news relevance scoring and quality assessment.
*   **Multi-Level Filtering**: Basic, Enhanced, and Integrated filtering for robust processing.
*   **News Quality Evaluation**: Automatic detection and filtering of low-quality, duplicate, and irrelevant news.
*   **Unified News Tools**: Integrates multiple news sources, providing a unified news retrieval interface.

### 🔧 Technical Fixes and Optimizations

*   **DashScope Adapter Fix**: Resolves tool-calling compatibility issues.
*   **DeepSeek Infinite Loop Fix**: Fixes infinite loops in the news analyst.
*   **Enhanced LLM Tool Calling**: Improves the reliability and stability of tool calling.
*   **News Retriever Optimization**: Enhances news data acquisition and processing capabilities.

### 📚 Comprehensive Testing and Documentation

*   **Extensive Test Coverage**: Over 15 new test files covering all new features.
*   **Detailed Technical Documentation**: Added 8 technical analysis reports and fix documentation.
*   **Improved User Guide**: Added news filtering usage guides and best practices.
*   **Demonstration Scripts**: Provides a complete news filtering feature demo.

### 🗂️ Project Structure Optimization

*   **Documentation Categorization**: Classifies documents by function into the docs subdirectory.
*   **Example Code Placement**: Demonstration scripts are unified in the examples directory.
*   **Root Directory Cleanliness**: Maintains a clean root directory for improved project professionalism.

---

## 🖥️ Web Interface Screenshots

> 🎨 **Modern Web Interface:** A responsive web application built with Streamlit, providing an intuitive stock analysis experience.

#### 🏠 Main Interface - Analysis Configuration

![1755003162925](images/README/1755003162925.png)

![1755002619976](images/README/1755002619976.png)

*Intelligent configuration panel, supports stock analysis for multiple markets, 5 levels of research depth selection*

#### 📊 Real-time Analysis Progress

![1755002731483](images/README/1755002731483.png)

*Real-time progress tracking, visualizes the analysis process, intelligent time estimation*

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

---

## 🎯 Core Features in Detail

### 🤖 Multi-Agent Collaboration Architecture

*   **Specialized Analysts:** Fundamental, Technical, News, and Social Media Analysts.
*   **Structured Debate:** Bullish and Bearish Researchers conduct in-depth analysis.
*   **Intelligent Decision-Making:** Traders make final investment recommendations based on all inputs.
*   **Risk Management:** Multi-level risk assessment and management mechanisms.

---

### 🎯 Core Feature Highlights

#### 📋 **Smart Analysis Configuration**

*   **🌍 Multi-Market Support**: US, A-Share, and H-Share analysis in one place.
*   **🎯 5 Research Depths**: From 2-minute quick analysis to 25-minute in-depth research.
*   **🤖 Agent Selection**: Market Technical, Fundamental, News, and Social Media Analysts.
*   **📅 Flexible Time Settings**: Supports historical analysis at any point in time.

#### 🚀 **Real-Time Progress Tracking**

*   **📊 Visual Progress**: Displays analysis progress and estimated time remaining in real-time.
*   **🔄 Intelligent Step Recognition**: Automatically identifies the current analysis stage.
*   **⏱️ Accurate Time Estimation**: Intelligent time calculation based on historical data.
*   **💾 State Persistence**: Analysis progress is preserved even after page refresh.

#### 📈 **Professional Results Display**

*   **🎯 Investment Decisions**: Clear Buy/Hold/Sell recommendations.
*   **📊 Multi-Dimensional Analysis**: Integrated assessment of technical, fundamental, and news aspects.
*   **🔢 Quantitative Indicators**: Confidence levels, risk scores, and target price.
*   **📄 Professional Reports**: Supports exporting reports in Markdown/Word/PDF format.

#### 🤖 **Multi-LLM Model Management**

*   **🌐 4 Providers**: DashScope, DeepSeek, Google AI, and OpenRouter.
*   **🎯 60+ Model Choices**: Comprehensive coverage from economic to flagship models.
*   **💾 Configuration Persistence**: URL parameter storage, settings are saved on refresh.
*   **⚡ Quick Switching**: One-click buttons for 5 popular model selections.

---

## 🎮 Web Interface Operation Guide

#### 🚀 **Quick Start**

1.  **Start the Application**: `python start_web.py` or `docker-compose up -d`
2.  **Access the Interface**: Open `http://localhost:8501` in your browser.
3.  **Configure Model**: Select LLM provider and model in the sidebar.
4.  **Enter Stock**: Enter the stock code (e.g., AAPL, 000001, 0700.HK).
5.  **Select Depth**: Choose research depth from 1-5 based on your needs.
6.  **Start Analysis**: Click the "🚀 Start Analysis" button.
7.  **View Results**: Track progress in real-time and view the analysis report.
8.  **Export Report**: One-click export of professional format reports.

#### 📊 **Supported Stock Code Formats**

*   **🇺🇸 US Stocks**: `AAPL`, `TSLA`, `MSFT`, `NVDA`, `GOOGL`
*   **🇨🇳 A-Shares**: `000001`, `600519`, `300750`, `002415`
*   **🇭🇰 H-Shares**: `0700.HK`, `9988.HK`, `3690.HK`, `1810.HK`

#### 🎯 **Research Depth Explanation**

*   **Level 1 (2-4 minutes)**: Quick overview, basic technical indicators
*   **Level 2 (4-6 minutes)**: Standard analysis, technical + fundamentals
*   **Level 3 (6-10 minutes)**: In-depth analysis, including news sentiment ⭐ **Recommended**
*   **Level 4 (10-15 minutes)**: Comprehensive analysis, multi-round agent debate
*   **Level 5 (15-25 minutes)**: Most in-depth analysis, complete research report

#### 💡 **Tips**

*   **🔄 Real-time Refresh**: Refresh the page at any time during the analysis without losing progress
*   **📱 Mobile Adaptation**: Supports access on mobile phones and tablets
*   **🎨 Dark Mode**: Automatically adapts to system theme settings
*   **⌨️ Shortcuts**: Supports the Enter key to quickly submit analysis
*   **📋 History**: Automatically saves recent analysis configurations

> 📖 **Detailed Guide**: For complete web interface usage instructions, please refer to [🖥️ Web Interface Detailed Guide](docs/usage/web-interface-detailed-guide.md)

---

## 🎯 Features

### 🚀  Smart News Analysis✨ **v0.1.12 Major Upgrade**

| Feature               | Status        | Details                                 |
| ---------------------- | ----------- | ---------------------------------------- |
| **🧠 Smart News Analysis**    | 🆕 v0.1.12  | AI News Filtering, Quality Assessment, Relevance Analysis         |
| **🔧 News Filter**      | 🆕 v0.1.12  | Multi-level Filtering, Basic/Enhanced/Integrated Three-Level Processing       |
| **📰 Unified News Tools**    | 🆕 v0.1.12  | Integrates Multiple News Sources, Unified Interface, Intelligent Retrieval         |
| **🤖 Multi-LLM Providers**     | 🆕 v0.1.11  | 4 Major Providers, 60+ Models, Smart Classification Management         |
| **💾 Model Selection Persistence**  | 🆕 v0.1.11  | URL Parameter Storage, Refresh Retention, Configuration Sharing          |
| **🎯 Quick Selection Buttons**    | 🆕 v0.1.11  | One-Click Hot Model Switching, Improved Operational Efficiency           |
| **📊 Real-time Progress Display**    | ✅ v0.1.10  | Asynchronous Progress Tracking, Smart Step Recognition, Accurate Time Calculation |
| **💾 Smart Session Management**    | ✅ v0.1.10  | State Persistence, Automatic Downgrading, Cross-Page Recovery         |
| **🎯 One-Click View Report**    | ✅ v0.1.10  | One-Click View After Analysis, Intelligent Result Recovery         |
| **🖥️ Streamlit Interface** | ✅ Full Support | Modern Responsive Interface, Real-time Interaction and Data Visualization   |
| **⚙️ Configuration Management**      | ✅ Full Support | Web-Side API Key Management, Model Selection, Parameter Configuration     |

### 🎨 CLI User Experience ✨ **v0.1.9 Optimization**

| Feature                | Status        | Details                             |
| ----------------------- | ----------- | ------------------------------------ |
| **🖥️ Interface and Log Separation** | ✅ Full Support | User Interface Clean and Beautiful, Technical Logs Managed Independently   |
| **🔄 Intelligent Progress Display**     | ✅ Full Support | Multi-Stage Progress Tracking, Prevents Duplicate Prompts         |
| **⏱️ Time Estimation Feature**   | ✅ Full Support | Intelligent Analysis Stage Displays Estimated Time Consumption             |
| **🌈 Rich Color Output**     | ✅ Full Support | Color Progress Indicators, Status Icons, Visual Effect Enhancement |

### 🧠 LLM Model Support ✨ **v0.1.13 Comprehensive Upgrade**

| Model Provider        | Supported Models                     | Feature Functions                | New Function |
| ----------------- | ---------------------------- | ----------------------- | -------- |
| **🇨🇳 Alibaba Baichuan** | qwen-turbo/plus/max          | Chinese Optimization, High Cost-Effectiveness    | ✅ Integrated  |
| **🇨🇳 DeepSeek** | deepseek-chat                | Tool Calling, High Cost Performance    | ✅ Integrated  |
| **🌍 Google AI**  | **9 Verified Models**              | Latest Gemini 2.5 Series      | 🆕 Upgraded  |
| ├─**Latest Flagship**  | gemini-2.5-pro/flash         | Latest Flagship, Super Fast Response      | 🆕 New  |
| ├─**Stable Recommendation**  | gemini-2.0-flash             | Recommended for Use, Balance Performance      | 🆕 New  |
| ├─**Classic and Powerful**  | gemini-1.5-pro/flash         | Classic and Stable, High-Quality Analysis    | ✅ Integrated  |
| └─**Lightweight and Fast**  | gemini-2.5-flash-lite        | Lightweight Tasks, Fast Response    | 🆕 New  |
| **🌐 Native OpenAI** | **Custom Endpoint Support**           | Any OpenAI-Compatible Endpoint      | 🆕 New  |
| **🌐 OpenRouter** | **60+ Model Aggregation Platform**          | One API Access for All Mainstream Models | ✅ Integrated  |
| ├─**OpenAI**    | o4-mini-high, o3-pro, GPT-4o | Latest o-Series, Professional Inference Version   | ✅ Integrated  |
| ├─**Anthropic** | Claude 4 Opus/Sonnet/Haiku   | Top Performance, Balanced Version      | ✅ Integrated  |
| ├─**Meta**      | Llama 4 Maverick/Scout       | Latest Llama 4 Series         | ✅ Integrated  |
| └─**Custom**    | Any OpenRouter Model ID         | Unlimited Expansion, Personalized Choice    | ✅ Integrated  |

**🎯 Quick Selection**: 5 Quick Hot Model Buttons | **💾 Persistence**: URL Parameter Storage, Refresh Retention | **🔄 Smart Switching**: One-Click Switching Between Different Providers

### 📊 Data Sources and Markets

| Market Type      | Data Source                   | Coverage                     |
| ------------- | ------------------------ | ---------------------------- |
| **🇨🇳 A-Shares**  | Tushare, AkShare, Tongdaxin | Shanghai and Shenzhen, Real-time Quotes, Financial Reports |
| **🇭🇰 H-Shares** | AkShare, Yahoo Finance   | Hong Kong Stock Exchange, Real-time Quotes, Fundamentals     |
| **🇺🇸 US Stocks** | FinnHub, Yahoo Finance   | NYSE, NASDAQ, Real-time Data       |
| **📰 News**   | Google News              | Real-time News, Multilingual Support         |

### 🤖 Agent Team

**Analyst Team**: 📈 Market Analysis | 💰 Fundamental Analysis | 📰 News Analysis | 💬 Sentiment Analysis
**Research Team**: 🐂 Bullish Researchers | 🐻 Bearish Researchers | 🎯 Trading Decision Maker
**Management**: 🛡️ Risk Manager | 👔 Research Director

---

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone the project
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables
cp .env.example .env
# Edit the .env file and fill in the API keys

# 3. Start the service
# First start or code change (requires image building)
docker-compose up -d --build

# Daily start (image exists, no code change)
docker-compose up -d

# Smart start (automatically determines whether to build)
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

### 📊 Start Analysis

1.  **Select Model**: DeepSeek V3 / Tongyi Qianwen / Gemini
2.  **Enter Stock**: `000001` (A-Share) / `AAPL` (US Stock) / `0700.HK` (H-Share)
3.  **Start Analysis**: Click the "🚀 Start Analysis" button
4.  **Real-time Tracking**: Observe real-time progress and analysis steps
5.  **View Report**: Click the "📊 View Analysis Report" button
6.  **Export Report**: Supports Word/PDF/Markdown formats

---

## 🎯 Core Advantages

*   **🧠 Smart News Analysis**:  v0.1.12 introduces an AI-driven news filtering and quality assessment system.
*   **🔧 Multi-Level Filtering**: Basic, Enhanced, and Integrated three-level news filtering mechanism.
*   **📰 Unified News Tools**: Integrates multiple news sources, providing a unified intelligent search interface.
*   **🆕 Multi-LLM Integration**: v0.1.11 introduces 4 major providers, 60+ models, for an all-in-one AI experience.
*   **💾 Configuration Persistence**: Model selection is truly persistent, URL parameter storage, and refresh retention.
*   **🎯 Quick Switching**: 5 hot model quick buttons for one-click switching between different AIs.
*   **🆕 Real-time Progress**: v0.1.10 asynchronous progress tracking, bid farewell to the black box wait.
*   **💾 Smart Sessions**: State persistence, page refresh does not lose analysis results.
*   **🇨🇳 Chinese Optimization**: A-Share/H-Share data + domestic LLM + Chinese interface
*   **🐳 Containerization**: Docker one-click deployment, environment isolation, and rapid expansion.
*   **📄 Professional Reports**: Multi-format export, automated investment advice.
*   **🛡️ Stable and Reliable**: Multi-layer data sources, intelligent downgrading, and error recovery.

---

## 🔧 Technical Architecture

**Core Technologies**: Python 3.10+ | LangChain | Streamlit | MongoDB | Redis
**AI Models**: DeepSeek V3 | Alibaba Baichuan | Google AI | OpenRouter(60+ models) | OpenAI
**Data Sources**: Tushare | AkShare | FinnHub | Yahoo Finance
**Deployment**: Docker | Docker Compose | Local Deployment

---

## 📚 Documentation and Support

*   **📖 Complete Documentation**: [docs/](./docs/) - Installation guide, usage tutorials, API documentation
*   **🚨 Troubleshooting**: [troubleshooting/](./docs/troubleshooting/) - Solutions to common problems
*   **🔄 Changelog**: [CHANGELOG.md](./docs/releases/CHANGELOG.md) - Detailed version history
*   **🚀 Quick Start**: [QUICKSTART.md](./QUICKSTART.md) - 5-minute quick deployment guide

---

## 🆚 Chinese Enhanced Features

**Compared to the original version, new features**: Intelligent News Analysis | Multi-level News Filtering | News Quality Assessment | Unified News Tools | Multi-LLM Provider Integration | Model Selection Persistence | Quick Switch Buttons | Real-time Progress Display | Smart Session Management | Chinese Interface | A-Share Data | Domestic LLM | Docker Deployment | Professional Report Export | Unified Log Management | Web Configuration Interface | Cost Optimization

**Services included in Docker deployment**:

-   🌐 **Web Application**: TradingAgents-CN main program
-   🗄️ **MongoDB**: Data persistence storage
-   ⚡ **Redis**: High-speed cache
-   📊 **MongoDB Express**: Database management interface
-   🎛️ **Redis Commander**: Cache management interface

#### 💻 Method 2: Local Deployment

**Applicable scenarios**: Development environment, custom configuration, offline use

### Environment Requirements

-   Python 3.10+ (recommended 3.11)
-   4GB+ RAM (recommended 8GB+)
-   Stable network connection

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
# or use pip install -e .
pip install -e .

# Note: requirements.txt already includes all necessary dependencies:
# - Database Support (MongoDB + Redis)
# - Multi-Market Data Sources (Tushare, AKShare, FinnHub, etc.)
# - Web Interface and Report Export Functionality
```

### Configure API Keys

#### 🇨🇳 Recommended: Use Alibaba Baichuan (Domestic Large Model)

```bash
# Copy configuration template
cp .env.example .env

# Edit the .env file and configure the following required API keys:
DASHSCOPE_API_KEY=your_dashscope_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here

# Recommended: Tushare API (Professional A-Share Data)
TUSHARE_TOKEN=your_tushare_token_here
TUSHARE_ENABLED=true

# Optional: Other AI Model APIs
GOOGLE_API_KEY=your_google_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Database configuration (optional, to improve performance)
# Use standard ports for local deployment
MONGODB_ENABLED=false  # Set to true to enable MongoDB
REDIS_ENABLED=false    # Set to true to enable Redis
MONGODB_HOST=localhost
MONGODB_PORT=27017     # Standard MongoDB port
REDIS_HOST=localhost
REDIS_PORT=6379        # Standard Redis port

# Docker deployment requires modifying the hostname
# MONGODB_HOST=mongodb
# REDIS_HOST=redis
```

#### 📋 Deployment Mode Configuration Instructions

**Local Deployment Mode**:

```bash
# Database configuration (local deployment)
MONGODB_ENABLED=true
REDIS_ENABLED=true
MONGODB_HOST=localhost      # Local host
MONGODB_PORT=27017         # Standard Port
REDIS_HOST=localhost       # Local host
REDIS_PORT=6379           # Standard Port
```

**Docker Deployment Mode**:

```bash
# Database configuration (Docker deployment)
MONGODB_ENABLED=true
REDIS_ENABLED=true
MONGODB_HOST=mongodb       # Docker container service name
MONGODB_PORT=27017        # Standard Port
REDIS_HOST=redis          # Docker container service name
REDIS_PORT=6379          # Standard Port
```

> 💡 **Configuration Hints**:
>
> - Local deployment: You need to manually start MongoDB and Redis services
> - Docker deployment: Database services are automatically started via docker-compose
> - Port conflicts: If you already have a database service locally, you can modify the port mapping in docker-compose.yml

#### 🌍 Optional: Use Foreign Models

```bash
# OpenAI (requires a VPN)
OPENAI_API_KEY=your_openai_api_key

# Anthropic (requires a VPN)
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 🗄️ Database Configuration (MongoDB + Redis)

#### High-Performance Data Storage Support

This project supports **MongoDB** and **Redis** databases, providing:

-   **📊 Stock Data Caching**: Reduces API calls, improves response speed
-   **🔄 Intelligent Downgrading Mechanism**: MongoDB → API → Local Cache Multi-Layer Data Source
-   **⚡ High-Performance Caching**: Redis caches hot data, millisecond response
-   **🛡️ Data Persistence**: MongoDB stores historical data, supports offline analysis

#### Database Deployment Method

**🐳 Docker Deployment (Recommended)**

If you use Docker deployment, the database is already included:

```bash
# Docker deployment automatically starts all services, including:
docker-compose up -d --build
# - Web application (port 8501)
# - MongoDB (port 27017)
# - Redis (port 6379)
# - Database Management Interface (ports 8081, 8082)
```

**💻 Local Deployment - Database Configuration**

If you use local deployment, you can choose the following methods:

**Method 1: Only Start Database Services**

```bash
# Only start MongoDB + Redis services (do not start the Web application)
docker-compose up -d mongodb redis mongo-express redis-commander

# View service status
docker-compose ps

# Stop the service
docker-compose down
```

**Method 2: Fully Local Installation**

```bash
# Database dependencies are already included in requirements.txt, no need to install extra
# Start MongoDB (default port 27017)
mongod --dbpath ./data/mongodb

# Start Redis (default port 6379)
redis-server
```

> ⚠️ **Important Note**:
>
> - **🐳 Docker Deployment**: The database is automatically included, no additional configuration is required
> - **💻 Local Deployment**: You can choose to only start the database service or fully install locally
> - **📋 Recommendation**: Use Docker deployment to get the best experience and consistency

#### Database Configuration Options

**Environment Variable Configuration** (Recommended):

```bash
# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=trading_agents
MONGODB_USERNAME=admin
MONGODB_PASSWORD=your_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_DB=0
```

**Configuration File Method**:

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

#### Database Feature Attributes

**MongoDB Features**:

-   ✅ Stock Basic Information Storage
-   ✅ Historical Price Data Caching
-   ✅ Analysis Result Persistence
-   ✅ User Configuration Management
-   ✅ Automatic Data Synchronization

**Redis Features**:

-   ⚡ Real-time Price Data Caching
-   ⚡ API Response Result Caching
-   ⚡ Session State Management
-   ⚡ Hot Data Preloading
-   ⚡ Distributed Lock Support

#### Intelligent Downgrading Mechanism

The system uses a multi-layer data source downgrading strategy to ensure high availability:

```
📊 Data Acquisition Process:
1. 🔍 Check Redis Cache (milliseconds)
2. 📚 Query MongoDB Storage (seconds)
3. 🌐 Call Tongdaxin API (seconds)
4. 💾 Local File Cache (backup)
5. ❌ Return Error Information
```

**Configure the Downgrading Strategy**:

```python
# Configure in the .env file
ENABLE_MONGODB=true
ENABLE_REDIS=true
ENABLE_FALLBACK=true

# Cache expiration time (seconds)
REDIS_CACHE_TTL=300
MONGODB_CACHE_TTL=3600
```

#### Performance Optimization Recommendations

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

# Cleanup cache tools
python scripts/maintenance/cleanup_cache.py --days 7
```

#### Troubleshooting

**Common Problem Solving**:

1.  **🪟 Windows 10 ChromaDB Compatibility Issue**

    **Problem Phenomenon**: On Windows 10, the error `Configuration error: An instance of Chroma already exists for ephemeral with different settings` appears, while Windows 11 is normal.

    **Quick Solution**:

    ```bash
    # Solution 1: Disable memory functionality (recommended)
    # Add to the .env file:
    MEMORY_ENABLED=false

    # Solution 2: Use a dedicated repair script
    powershell -ExecutionPolicy Bypass -File scripts\fix_chromadb_win10.ps1

    # Solution 3: Run with administrator privileges
    # Right-click PowerShell -> "Run as administrator"
    ```

    **Detailed Solution**: Refer to [Windows 10 Compatibility Guide](docs/troubleshooting/windows10-chromadb-fix.md)
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
    # Check the MongoDB process
    ps aux | grep mongod

    # Restart MongoDB
    sudo systemctl restart mongod  # Linux
    brew services restart mongodb  # macOS
    ```
3.  **Redis Connection Timeout**

    ```bash
    # Check Redis status
    redis-cli ping

    # Clear Redis cache
    redis-cli flushdb
    ```
4.  **Cache Problem**

    ```bash
    # Check system status and cache
    python scripts/validation/check_system_status.py

    # Clear expired cache
    python scripts/maintenance/cleanup_cache.py --days 7
    ```

> 💡 **Hint**: Even if you don't configure the database, the system can still run normally and will automatically downgrade to the API direct call mode. Database configuration is an optional performance optimization function.

> 📚 **Detailed Documents**: For more database configuration information, please refer to [Database Architecture Documentation](docs/architecture/database-architecture.md)

### 📤 Report Export Function

#### New Feature: Professional Analysis Report Export

This project now supports exporting stock analysis results to a variety of professional formats:

**Supported Export Formats**:

-   📄 **Markdown (.md)** - Lightweight markup language, suitable for technical users and version control
-   📝 **Word (.docx)** - Microsoft Word documents, suitable for business reports and further editing
-   📊 **PDF (.pdf)** - Portable Document Format, suitable for formal sharing and printing

**Report Content Structure**:

-   🎯 **Investment Decision Summary** - Buy/Hold/Sell recommendations, confidence levels, risk scores
-   📊 **Detailed Analysis Report** - Technical analysis, fundamental analysis, market sentiment, news events
-   ⚠️ **Risk Warning** - Complete investment risk statement and disclaimer
-   📋 **Configuration Information** - Analysis parameters, model information, generation time

**How to Use**:

1.  After completing the stock analysis, find the "📤 Export Report" section at the bottom of the results page
2.  Select the desired format: Markdown, Word, or PDF
3.  Click the export button, and the system automatically generates and provides a download

**Install Export Dependencies**:

```bash
# Install Python dependencies
pip install markdown pypandoc

# Install system tools (for PDF export)
# Windows: choco install pandoc wkhtmltopdf
# macOS: brew install pandoc wkhtmltopdf
# Linux: sudo apt-get install pandoc wkhtmltopdf
```

> 📚 **Detailed Documentation**: For a complete guide to using the export function, please refer to [Export Function Guide](docs/EXPORT_GUIDE.md)

### 🚀 Start the Application

#### 🐳 Docker Start (Recommended)

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

#### 💻 Local Start

If you are using local deployment:

```bash
# 1. Activate the virtual environment
# Windows
.\env