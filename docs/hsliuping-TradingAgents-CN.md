# TradingAgents-CN: 中文金融AI交易决策框架

> **🚀 借助AI的力量，轻松驾驭中文金融市场！** TradingAgents-CN是基于多智能体大语言模型的中文金融交易决策框架，专为中国用户优化，提供全面的A股/港股/美股分析能力，助您在金融市场中做出更明智的决策。  [访问原项目](https://github.com/TauricResearch/TradingAgents)

## ✨ Key Features

*   🤖 **多智能体协作架构**: 专业的分析师团队，结构化辩论，智能决策和风险管理。
*   🇨🇳 **中文支持**:  深度优化，支持A股/港股/美股市场。
*   🧠 **智能新闻分析 (v0.1.12)**: AI驱动的新闻过滤、质量评估和相关性分析。
*   🆕 **多LLM集成 (v0.1.11)**:  支持阿里百炼、DeepSeek、Google AI、OpenRouter(60+模型) 等多LLM提供商，灵活选择。
*   💾 **模型选择持久化 (v0.1.11)**:  URL参数存储，刷新配置保持，方便分享。
*   🐳 **Docker部署**: 一键部署，环境隔离，快速启动。
*   📄 **专业报告**:  多格式 (Markdown, Word, PDF) 导出，提供投资建议。
*   🔄 **实时进度**:  异步进度跟踪，告别黑盒等待。
*   🇨🇳 **中国优化**: A股/港股数据 + 国产LLM + 中文界面。

## 🆕 What's New in v0.1.12

### 🧠 智能新闻分析 
*   **AI新闻过滤**:  基于AI的新闻相关性评分和质量评估。
*   **多层次过滤机制**: 基础过滤、增强过滤、集成过滤。
*   **新闻质量评估**: 自动识别并过滤低质量、重复和无关新闻。
*   **统一新闻工具**: 整合多个新闻源，提供统一新闻获取接口。

## 🛠️ How to Get Started

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. Configure environment variables (API keys)
cp .env.example .env
# Edit .env file to add your API keys

# 3. Build and Start Services
docker-compose up -d --build  # First time or code changes
docker-compose up -d           # Subsequent runs

# 4. Access the Application
# Web Interface: http://localhost:8501
```

### 💻 Local Deployment

```bash
# 1. Upgrade pip
pip install --upgrade pip

# 2. Install dependencies
pip install -e .

# 3. Start the application
python start_web.py

# 4. Access the application
# Open your web browser and go to http://localhost:8501
```

## 📚 Comprehensive Documentation

Explore our detailed documentation to understand the framework thoroughly:
*   [docs/](docs/) - Detailed documentation covering installation, usage, and API.
*   [QUICKSTART.md](./QUICKSTART.md) - Quick start guide in 5 minutes.

## 🤝 Contributing

We welcome contributions!

*   Fork the repository.
*   Create a feature branch (`git checkout -b feature/AmazingFeature`).
*   Commit your changes (`git commit -m 'Add some AmazingFeature'`).
*   Push to the branch (`git push origin feature/AmazingFeature`).
*   Create a pull request.

[CONTRIBUTORS.md](CONTRIBUTORS.md) - See the list of our contributors.

## 📄 License

Licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file.