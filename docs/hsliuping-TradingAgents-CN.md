# TradingAgents-CN: 中文金融交易AI框架 - 🚀 开启您的智能交易之旅

**利用多智能体大语言模型，深度分析A股、港股和美股，为您的投资决策提供专业的支持。** 

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

> **v0.1.13 (Preview):** ⚡️ **全面支持OpenAI & Google AI！** 包括自定义OpenAI端点、9个Google AI模型、LLM适配器优化，带来更强大、更灵活的中文金融分析体验！

## 🚀 核心优势：

*   **多市场支持**: A股、港股、美股一站式分析。
*   **多智能体架构**: 基本面、技术面、新闻面、情绪面多维度分析。
*   **多LLM模型**: 支持阿里百炼、DeepSeek、Google AI、OpenAI & OpenRouter等。
*   **专业报告导出**: Markdown/Word/PDF格式，轻松分享分析结果。
*   **中文本地化**: 专为中文用户优化，提供完整A股数据支持。
*   **Web界面**: 直观的股票分析界面，实时进度跟踪，方便易用。
*   **快速部署**:  Docker 一键部署，快速启动，轻松体验。
*   **开源 & 免费**:  基于Apache 2.0协议，自由使用，持续更新。

## ✨ 主要功能

*   **智能新闻分析**: AI驱动的新闻过滤与质量评估。
*   **多层次过滤**: 基础、增强、集成三级新闻过滤。
*   **多LLM集成**: 4大提供商，60+模型，一站式AI体验。
*   **模型选择持久化**: URL参数存储，刷新保持。
*   **实时进度**: 异步进度跟踪，告别黑盒等待。
*   **智能会话**: 状态持久化，页面刷新不丢失。
*   **专业报告**: 多格式导出，自动生成投资建议。

## 🆕 最新更新: v0.1.13

### 🤖 全面支持OpenAI 与 Google AI

*   **自定义OpenAI端点**: 使用任何OpenAI兼容API。
*   **灵活模型选择**: 支持各种OpenAI格式的模型。
*   **Google AI 集成**: 9个Gemini模型，包括gemini-2.5-pro, gemini-2.5-flash。
*   **LLM适配器架构优化**: 新增Google AI的OpenAI兼容适配器。

## 💻 快速上手

1.  **Docker 部署 (推荐)**: 简单快速，一键启动！
    ```bash
    # 克隆项目
    git clone https://github.com/hsliuping/TradingAgents-CN.git
    cd TradingAgents-CN

    # 配置环境变量 (.env文件，填写API密钥，参考 .env.example)
    cp .env.example .env
    # 填写API密钥

    # 启动服务
    docker-compose up -d --build # 首次构建
    docker-compose up -d         # 再次启动
    ```
    访问Web界面: `http://localhost:8501`

2.  **本地部署**: 适合开发与自定义配置
    ```bash
    # 升级pip
    pip install --upgrade pip
    # 安装依赖
    pip install -e .
    # 启动应用
    python start_web.py
    ```
    访问 Web 界面: `http://localhost:8501`

3.  **分析步骤**:
    1.  选择 LLM 模型 (DeepSeek V3 / 通义千问 / Gemini等)
    2.  输入股票代码 (AAPL, 000001, 0700.HK)
    3.  点击"🚀 开始分析"
    4.  查看实时进度，获取分析报告，导出报告。

## 🔑 核心功能特色

| 功能             | 描述                                                     |
| ---------------- | -------------------------------------------------------- |
| **多市场支持**   | 美股、A股、港股一站式分析                                  |
| **研究深度选择** | 5级研究深度，从快速概览到深度分析                          |
| **智能体选择**   | 市场、基本面、新闻、社交媒体分析师                         |
| **实时进度跟踪** | 可视化分析过程，智能时间预估                              |
| **专业结果展示** | 投资建议、多维分析、量化指标、专业报告导出                 |
| **多LLM模型管理**| 4大提供商，60+模型选择，快速切换                           |
| **智能新闻分析**   | 🤖 新闻过滤，质量评估，相关性分析                           |

## 📚 完整文档 & 社区支持

*   **中文文档**: 深入了解 [docs/](./docs/) - 安装、使用、技术细节一应俱全！
*   **技术支持**: 欢迎在 [GitHub Issues](https://github.com/hsliuping/TradingAgents-CN/issues) 提出问题。
*   **贡献指南**:  [CONTRIBUTING.md](CONTRIBUTING.md) - 欢迎贡献代码，改进文档！

## 🙏 致敬与感谢

感谢 [Tauric Research](https://github.com/TauricResearch) 团队的开源项目 [TradingAgents](https://github.com/TauricResearch/TradingAgents)！

## 📄 许可证

本项目基于 [Apache 2.0](LICENSE) 许可证。

---

<div align="center">

**⭐  给个 Star 支持我们！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [📖 阅读中文文档](./docs/)

</div>