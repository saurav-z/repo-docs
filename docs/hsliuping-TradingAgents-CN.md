# TradingAgents-CN: 中文金融交易决策框架 - 🚀 开启智能投资新时代！

基于多智能体大语言模型的中文金融交易决策框架，专为中国市场量身打造，提供**A股/港股/美股全市场分析能力**。 **[访问原项目](https://github.com/hsliuping/TradingAgents-CN)**，体验智能投资的魅力！

**Key Features:**

*   🧠 **智能新闻分析**: AI驱动的过滤、评估和相关性分析 (v0.1.12)
*   🆕 **多层次新闻过滤**: 基础/增强/集成三级过滤，精准筛选
*   📰 **统一新闻工具**: 整合多源新闻，提供智能检索
*   🤖 **多LLM提供商**: 阿里云、DeepSeek、Google AI、OpenRouter 等 (v0.1.11)
*   💾 **模型选择持久化**: 刷新页面配置不丢失 (v0.1.11)
*   🇨🇳 **中国市场支持**: 深度支持 A股、港股和美股
*   🐳 **Docker 部署**: 快速搭建和环境隔离
*   📊 **专业报告导出**: 一键生成 Markdown/Word/PDF 报告

## 💡 核心优势

*   **中文优化**: 专为中文用户设计，更友好的界面和分析结果。
*   **多模型支持**: 集成多种 LLM 模型，满足不同需求。
*   **实时数据**: 连接 Tushare, AkShare 等数据源，提供实时行情。
*   **多智能体协作**: 市场、基本面、新闻、情绪分析师协同工作。
*   **一键部署**: 使用 Docker 快速搭建，轻松体验。
*   **数据持久化**: 历史数据、分析结果保存，便于复盘。
*   **报告导出**: 生成专业报告，方便分享和进一步分析。

## 🚀 快速开始

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
docker-compose up -d --build
# 访问: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 安装依赖
pip install -e .

# 2. 启动应用
python start_web.py
# 访问: http://localhost:8501
```

### 📈 开始分析

1.  **选择模型**: DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票**: `000001` (A股) / `AAPL` (美股) / `0700.HK` (港股)
3.  **开始分析**: 点击 "🚀 开始分析" 按钮
4.  **查看报告**: 点击 "📊 查看分析报告" 按钮

## 📚 详细文档

我们提供**超过 50,000 字的详细中文文档**，涵盖：

*   [项目概述](docs/overview/project-overview.md)
*   [系统架构](docs/architecture/system-architecture.md)
*   [智能体架构](docs/architecture/agent-architecture.md)
*   [核心特性详解](docs/)
*   [常见问题](docs/faq/faq.md)

## 🤝 贡献

欢迎贡献代码、文档、问题反馈等！

## 📄 许可证

本项目基于 [Apache 2.0](LICENSE) 许可证开源。

---