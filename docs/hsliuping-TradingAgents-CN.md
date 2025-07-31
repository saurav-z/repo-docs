# TradingAgents-CN: AI赋能的中文金融交易决策框架

> **解锁智能金融交易新境界！** TradingAgents-CN是基于[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)的中文增强版，专为中国市场优化，提供全面的A股、港股、美股分析能力，助您洞悉市场脉搏。

## 🚀 主要特点

*   **🇨🇳 中文优化**：专为中文用户设计，界面、数据源及模型全面支持中文。
*   **🧠 智能新闻分析**：**v0.1.12重大升级**，AI驱动的新闻过滤、质量评估与相关性分析，助力精准决策。
    *   **📰 智能新闻过滤**：基础、增强、集成多层过滤机制。
    *   **📰 统一新闻工具**：整合多源新闻，智能检索。
*   **🤖 多LLM提供商**：**v0.1.11全面升级**，支持阿里云百炼、DeepSeek、Google AI、OpenRouter，60+模型任您选择。
*   **💾 模型持久化**：模型选择持久化，通过URL分享配置。
*   **🐳 一键部署**：Docker一键部署，快速搭建，轻松上手。
*   **📊 专业报告**：自动生成Word/PDF/Markdown格式的专业分析报告，便于分享和查阅。
*   **🇨🇳 A股全面支持**：提供全面的A股市场数据和分析能力。
*   **⚙️ 易于配置**：Web端API密钥管理，模型选择和参数配置。

## ✨ 最新版本更新

### 🧠 智能新闻分析模块 v0.1.12

*   **🚀 智能新闻过滤**：基于AI的新闻相关性评分和质量评估。
*   **🔧 新闻过滤器**：多层次过滤，基础/增强/集成三级处理。
*   **📰 统一新闻工具**：整合多源新闻，提供统一接口，智能检索。

### 🤖 多LLM提供商 v0.1.11

*   **4大提供商支持**：阿里云百炼、DeepSeek、Google AI、OpenRouter。
*   **60+模型选择**：涵盖最新模型，如Claude 4 Opus、GPT-4o。
*   **智能模型分类**：OpenRouter支持多种模型类别。
*   **💾 模型选择持久化**：URL参数存储，刷新保持配置。
*   **🎯 快速选择按钮**：一键切换热门模型，提升操作效率。

## 📚 核心功能

*   **多智能体协作**：基本面、技术面、新闻面、情绪面分析师协同工作，给出综合投资建议。
*   **支持市场**：涵盖A股、港股、美股市场，提供实时数据和分析。
*   **风险管理**：多层次风险评估和管理机制，降低投资风险。
*   **自定义配置**：支持多种LLM模型和参数配置，满足个性化需求。

## 🚀 快速开始

### 🐳 Docker部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
docker-compose up -d --build

# 4. 访问应用
# Web界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 升级pip
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -e .

# 3. 启动应用
python start_web.py

# 4. 访问 http://localhost:8501
```

## 💡 核心优势

*   **易于上手**：一键部署，快速搭建，用户友好的Web界面。
*   **功能丰富**：多LLM集成、智能新闻分析、A股全面支持、专业报告导出等。
*   **性能卓越**：多层缓存、异步处理，提升运行效率。
*   **持续更新**：持续优化和更新，紧跟技术发展趋势。
*   **社区支持**：活跃的社区，提供技术支持和交流。

## 📚 文档与支持

*   **完整文档**：[docs/](./docs/) - 详细的安装、使用、API文档。
*   **故障排除**：[troubleshooting/](./docs/troubleshooting/) - 常见问题解决方案。
*   **更新日志**：[CHANGELOG.md](./docs/releases/CHANGELOG.md) - 详细的版本更新历史。
*   **联系方式**:  GitHub Issues, 邮箱 (hsliup@163.com), QQ群 (782124367)。

## 🔗 贡献

我们欢迎社区贡献，一起完善TradingAgents-CN！

*   **贡献指南**：[CONTRIBUTING.md](CONTRIBUTING.md)

## 📜 许可证

本项目基于 [Apache 2.0](LICENSE) 许可证开源。

---

**立即体验AI驱动的智能金融交易，开启您的投资新篇章！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN)