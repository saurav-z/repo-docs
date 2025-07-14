# 🇨🇳 TradingAgents-CN: 赋能中文金融交易的AI框架 🚀

**解锁基于多智能体大语言模型的中文金融交易框架，深度集成A股数据、国产大模型，提供Docker部署与专业报告导出功能，为您的量化交易注入强大动力!**  [访问原项目](https://github.com/hsliuping/TradingAgents-CN)

---

## ✨ 核心特性

*   **🤖 多智能体协作架构**: 模拟专业团队，全面分析市场
*   **🇨🇳 A股 & 国产LLM 支持**: 深度融合中国市场与技术
*   **🐳 Docker 容器化**: 快速部署，环境隔离，易于扩展
*   **📄 专业报告导出**: Markdown, Word, PDF 多格式报告
*   **🌐 现代化 Web 界面**: 交互式操作，实时数据可视化
*   **💰 低成本 LLM 选项**: 优化成本，高效运行

## 🔑 主要功能

*   **📊 智能分析**: 基本面、技术面、新闻面、社交媒体等多维度分析
*   **🧠 LLM 支持**: 集成阿里百炼、DeepSeek、Google AI 等多种模型
*   **📈 数据集成**: A股实时行情、历史数据、新闻资讯
*   **🛡️ 系统稳定**: 数据库缓存、错误恢复机制
*   **🎛️ 配置管理**: API 密钥管理、模型选择、监控
*   **🚀 核心优势**: 开箱即用、中国优化、智能配置、实时监控

## 🚀 快速开始

1.  **🐳 Docker 部署 (推荐)**:  快速体验，零配置启动

    ```bash
    git clone https://github.com/hsliuping/TradingAgents-CN.git
    cd TradingAgents-CN
    cp .env.example .env  # 编辑 .env 填入API密钥
    docker-compose up -d --build
    # Web界面: http://localhost:8501
    ```

2.  💻  本地部署:  开发调试，自定义配置

    ```bash
    git clone https://github.com/hsliuping/TradingAgents-CN.git
    cd TradingAgents-CN
    python -m venv env
    # Windows: env\Scripts\activate
    # Linux/macOS: source env/bin/activate
    pip install -r requirements.txt
    # 配置文件 .env, 填入API密钥
    streamlit run web/app.py
    ```

    *   数据库 (推荐): Docker Compose 或本地 MongoDB/Redis

## 📚 深入了解 (推荐)

*   **📖 项目概述**: [快速上手](docs/overview/quick-start.md)
*   **🏗️ 系统架构**: [系统架构](docs/architecture/system-architecture.md)
*   **🤖 智能体详解**: [智能体架构](docs/architecture/agent-architecture.md)
*   **❓ 常见问题**: [常见问题](docs/faq/faq.md)
*   **📚 完整文档**: [文档目录](docs/)

## 🆚 与原版主要区别

*   **🇨🇳 A股 & 中文支持**:  全面适配中国市场和语言
*   **🌐 Web 界面**: 现代化用户界面，易于操作
*   **🐳 Docker 一键部署**: 简化部署流程，方便使用
*   **📄 专业报告**: 多格式导出，提升分析效率
*   **🧠 国产大模型集成**:  DeepSeek, 阿里百炼，降低成本

## 🤝 贡献指南

*   [贡献流程](CONTRIBUTING.md)
*   欢迎 Bug 修复、新功能、文档改进等贡献

## 📄 许可证

Apache 2.0. 查看 [LICENSE](LICENSE) 文件。

---

<div align="center">

**🌟 给我们一个 Star，感谢您的支持！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [📖 查阅文档](./docs/)

</div>
```

Key improvements and summaries:

*   **SEO Optimization**: Focused keywords like "中文金融交易," "A股," "Docker," "大模型," and "量化交易" are used in headings and throughout the description.
*   **Concise Hook**: A one-sentence hook at the beginning that immediately tells the user what the project does and its core benefit.
*   **Key Features with Bullets**: The key features are presented in a clear, bulleted format.
*   **Clear Headings**: The headings are clear and easy to understand (e.g., "核心特性," "快速开始").
*   **Concise Summary**: Provides a quick overview of the project.
*   **Easy to Read**:  Uses formatting (bold, italics, code blocks) effectively.
*   **Clear Instructions**:  The "快速开始" section provides direct, actionable instructions for both Docker and local deployments.
*   **Emphasis on Differentiation**: The "🆚 与原版主要区别" section clearly highlights what makes this project unique.
*   **Comprehensive Links**: Contains relevant links (e.g., to the original repo, documentation, and contribution guidelines).
*   **Call to Action**: Includes a prominent call to action (Star the repo) at the end.
*   **More Informative Summary**: The content is better organized and more informative than the original, guiding the user toward core aspects of the project more directly.
*   **Emphasis on Chinese Language Aspects**: The summary specifically highlights the project's focus on the Chinese market and language.