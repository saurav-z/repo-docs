# TradingAgents-CN: 中文金融交易决策框架 🚀

**利用尖端AI技术，助力您在股票市场做出更明智的决策，为中文用户量身打造的金融交易框架，深度支持A股/港股/美股市场分析。**  [查看原始项目](https://github.com/hsliuping/TradingAgents-CN)

## 🌟 核心特性

*   **📰 智能新闻分析**: AI驱动的新闻过滤、质量评估和相关性分析，助您把握市场动态。
*   **🤖 多LLM支持**: 集成4大LLM提供商，支持60+模型，包括最新的GPT-4o和Claude 4 Opus。
*   **🇨🇳 中文优化**: 专为中文用户优化，提供A股/港股/美股全面分析能力。
*   **🐳 Docker部署**: 一键部署，环境隔离，轻松搭建和扩展。
*   **💾 模型持久化**:  模型配置保存，方便复用，并通过URL分享。
*   **📄 专业报告**: 导出分析结果为Markdown、Word、PDF等格式，方便分享。
*   **🚀 实时进度**:  告别黑盒等待，异步进度跟踪，让您随时了解分析进度。

## ✨ 主要更新

*   **v0.1.12 (最新)**: 智能新闻分析模块全面升级，新增多层次新闻过滤、质量评估、相关性分析。
    *   **核心亮点**: AI新闻过滤、多层次过滤机制、统一新闻工具。
*   **v0.1.11**: 多LLM提供商集成，模型选择持久化，带来更强大的模型支持和配置管理。
    *   **核心亮点**:  多LLM集成，模型持久化，快速模型切换。

## 📚 深入了解

*   **快速上手**:  [快速开始](docs/overview/quick-start.md)
*   **完整文档**:  [文档](docs/) - 超过50,000字的中文文档，涵盖从入门到精通的全面指南
*   **贡献指南**:  [贡献指南](CONTRIBUTING.md)

## 🔑 关键功能

*   **🤖 多智能体协作**: 市场、基本面、新闻、情绪四大分析师协同工作。
*   **📈 市场数据**: 支持 A股、港股、美股市场数据。
*   **🎨 Web界面**: 提供直观友好的用户界面。
*   **🛡️ 风险管理**:  多层风险评估与管理机制。
*   **🐳 容器化部署**:  Docker一键部署，方便快捷。

## 🚀 快速开始 (Docker 部署)

```bash
# 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 配置API密钥 (编辑 .env 文件)
cp .env.example .env
# 填入您的API密钥 (如 DashScope, Tushare 等)

# 启动
docker-compose up -d --build  # 首次构建
# docker-compose up -d  #  日常启动
#  或运行脚本:
#  Windows: powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1
#  Linux/Mac: chmod +x scripts/smart_start.sh && ./scripts/smart_start.sh

# 访问
# Web界面: http://localhost:8501
```

## 🛠️ 技术栈

*   **编程语言**: Python 3.10+
*   **核心框架**: LangChain
*   **前端框架**: Streamlit
*   **数据库**: MongoDB, Redis
*   **AI模型**:  DeepSeek V3, 阿里百炼, Google AI, OpenRouter (60+模型)

## 📝 许可证

本项目基于 [Apache 2.0](LICENSE) 许可证。

## 🙏 致谢

感谢 [Tauric Research](https://github.com/TauricResearch) 团队提供的开源框架 [TradingAgents](https://github.com/TauricResearch/TradingAgents)。

---

[⭐  点击Star，支持我们!](https://github.com/hsliuping/TradingAgents-CN)
```

Key improvements and summaries:

*   **SEO Optimization**:  Includes keywords like "中文金融", "交易决策", "A股", "港股", "美股", "AI" to enhance searchability.
*   **Clear Headings**: Organizes information with clear, concise headings and subheadings.
*   **One-Sentence Hook**: Starts with a compelling sentence to grab the reader's attention.
*   **Bulleted Key Features**: Highlights the most important features in easy-to-scan bullet points.
*   **Concise Summaries**: Condenses large blocks of text, focusing on the most important details.
*   **Actionable Call to Action**: Encourages the user to take action (e.g., "Star this repo").
*   **Complete Links**:  Provides links to the original repository and crucial sections of the documentation.
*   **Emphasis on Benefits**: Focuses on *what* the project does and *why* it's useful.
*   **Clear Versioning**:  Highlights the latest version and major updates.
*   **Removed redundant information:** Streamlined repeated information and condensed lengthy sections.
*   **Simplified and Focused Start Instructions:** Focused on most common install method using docker.
*   **Consistent formatting**: Overall improved readability with consistent markdown formatting.