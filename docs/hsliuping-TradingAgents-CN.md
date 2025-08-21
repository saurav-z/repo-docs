# TradingAgents-CN: 🚀 AI驱动的中文金融交易框架，为A股/港股/美股分析提供支持

**解锁智能交易决策的强大力量！** TradingAgents-CN 是一个基于多智能体大语言模型的中文金融交易决策框架，专为中文用户优化，提供全面的A股/港股/美股分析能力，助您洞悉市场先机。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**⭐  最新版本 cn-0.1.13-preview**:  原生OpenAI支持与Google AI全面集成预览版！ 🚀

**核心优势**：

*   🧠 **AI驱动的智能分析**： 结合多智能体协同、新闻分析、以及 LLM 技术的强大能力
*   🇨🇳 **中文优化 & 全市场支持**： 专为中国市场设计，全面支持 A 股、港股和美股
*   🐳 **一键部署 & 易于使用**：  Docker 容器化，本地部署，Web 界面，操作简单
*   🌐 **多 LLM 支持**: 阿里百炼、DeepSeek、Google AI、OpenRouter、原生OpenAI等多种模型
*   📊 **专业报告**：生成详细的投资分析报告，包括技术面、基本面和新闻面评估，一键导出
*   🤝 **完善文档**： 超过 50,000 字的中文文档，手把手带你入门

**查看代码库:** [hsliuping/TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN)

##  ✨ 核心功能

*   🤖 **多智能体协作**: 基本面、技术面、新闻面、社交媒体分析师，协同分析
*   📰 **智能新闻分析**:  AI驱动的新闻过滤、质量评估和相关性分析
*   🌐 **多LLM模型支持**:  阿里百炼、DeepSeek、Google AI、OpenRouter、原生OpenAI 多种模型
*   📊 **专业分析报告**:  技术、基本面、新闻等多维度评估，生成专业报告，支持多种格式导出
*   💻 **Web 界面**:  基于 Streamlit 构建，提供直观的股票分析体验，包括实时进度跟踪和分析结果展示
*   🇨🇳 **中文支持**:  专为中文用户优化，支持A股/港股/美股，中文界面和分析结果展示
*   🐳 **Docker 支持**:  一键部署，环境隔离，快速扩展

## 🆕  版本更新 (cn-0.1.13-preview)

### 🤖 原生 OpenAI 与 Google AI 集成

*   **自定义 OpenAI 端点**: 支持配置任意 OpenAI 兼容的 API 端点
*   **灵活模型选择**: 可以使用任何 OpenAI 格式的模型
*   **智能适配器**: 新增原生 OpenAI 适配器
*   **三大 Google AI 包支持**: langchain-google-genai、google-generativeai、google-genai
*   **9 个验证模型**: 包括 gemini-2.5-pro, gemini-2.5-flash 等最新模型
*   **Google 工具处理器**: 专门的 Google AI 工具调用处理器
*   **智能降级机制**: 高级功能失败时自动降级到基础功能

## 🙏 致敬原项目

感谢 [Tauric Research](https://github.com/TauricResearch) 团队创建了革命性的多智能体交易框架 [TradingAgents](https://github.com/TauricResearch/TradingAgents)！

## 🚀  快速开始

1.  **Docker 部署（推荐）**:

    ```bash
    git clone https://github.com/hsliuping/TradingAgents-CN.git
    cd TradingAgents-CN
    cp .env.example .env  # 配置 API 密钥
    docker-compose up -d --build  # 首次启动或代码更改
    # or docker-compose up -d
    # 访问Web界面: http://localhost:8501
    ```

2.  **本地部署**:

    ```bash
    pip install -e .
    python start_web.py
    # 访问 Web 界面: http://localhost:8501
    ```

    **重要**:  务必配置 API 密钥，参考 `.env.example`。

## 📖 详细文档

我们提供了超过 **50,000 字** 的中文文档，包含：

*   [快速开始指南](docs/overview/quick-start.md)
*   [系统架构解析](docs/architecture/system-architecture.md)
*   [LLM 模型配置指南](docs/configuration/llm-config.md)
*   [Web 界面详细使用说明](docs/usage/web-interface-detailed-guide.md)

[访问完整文档](./docs/)

## 🤝 贡献

我们欢迎您的贡献！ 请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多信息。

## 📄 许可证

本项目基于 Apache 2.0 许可证开源，详见 [LICENSE](LICENSE)。

---