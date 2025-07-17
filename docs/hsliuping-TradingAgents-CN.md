# TradingAgents-CN: 中文金融交易决策框架 🚀

[![](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/Version-cn--0.1.9-green.svg)](./VERSION)
[![](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**使用基于多智能体大语言模型的TradingAgents-CN，让您轻松驾驭中文金融市场，实现智能化交易策略！**  基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 开发，TradingAgents-CN 旨在为中文用户提供全面的 AI 驱动的金融交易解决方案，包括 A股支持、国产 LLM 集成、以及便捷的本地化体验。

## 🌟 主要特点

*   🤖 **多智能体协作**: 模拟专业交易团队，AI 驱动决策。
*   🇨🇳 **中文支持**: 完整的中文文档、界面和 A股数据支持。
*   🧠 **模型集成**: 支持阿里百炼、DeepSeek、Google AI、OpenAI 等 LLM 模型。
*   📊 **数据全面**: 支持 A股、美股、新闻和社交情绪数据。
*   🐳 **容器化部署**: Docker 快速部署，环境隔离，快速扩展。
*   📄 **专业报告**:  多格式导出，自动生成专业分析报告。

## ✨ 主要更新

*   **v0.1.9 (最新版本)**: 🎨 CLI用户体验重大优化与统一日志管理
*   **v0.1.8**: 🎨 Web界面全面优化与用户体验提升
*   **v0.1.7**: 🐳 容器化部署与专业报告导出
*   **v0.1.6**: 🔧 阿里百炼修复与数据源升级

## 🎯 核心功能

*   **多智能体系统**:  由分析师、研究员、交易员和风险管理团队组成，协同工作，提升决策质量。

    *   **分析师团队**: 技术、基本面、新闻、社交媒体四大分析师。
    *   **研究员团队**: 看涨/看跌研究员辩论，进行结构化分析。
    *   **交易员智能体**: 综合决策制定，仓位建议，止损止盈。
    *   **风险管理**: 多层次风险评估和管理机制。

*   **LLM 模型集成**: 灵活的模型选择，支持多语言、多模态。

    *   🇨🇳 阿里百炼: qwen-turbo, qwen-plus-latest, qwen-max (已完整支持)
    *   DeepSeek: deepseek-chat (已完整支持)
    *   Google AI: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash (已完整支持)
    *   OpenAI: GPT-4o, GPT-4o-mini, GPT-3.5-turbo (配置即用)
    *   Anthropic: Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku (配置即用)
    *   智能混合: Google AI推理 + 阿里百炼嵌入 (已优化)

*   **全面数据集成**: 涵盖多种数据源，满足不同分析需求。

    *   🇨🇳 A股数据: 通达信API (实时行情和历史数据) - 已完整支持
    *   🇺🇸 美股数据: FinnHub, Yahoo Finance (实时行情) - 已完整支持
    *   📰 新闻数据: Google News、财经新闻API (实时新闻) - 已完整支持
    *   💬 社交数据: Reddit 情绪分析 - 已完整支持
    *   🗄️ 数据库支持: MongoDB 数据持久化 + Redis 高速缓存 - 已完整支持
    *   🔄 智能降级: MongoDB → 通达信API → 本地缓存 (多层数据源) - 已完整支持
    *   ⚙️ 统一配置: .env 文件统一管理，启用开关完全生效 - 已完整支持

*   **Web 管理界面**:  用户友好的 Web 界面，提供直观的操作体验和实时监控。

    *   🇨🇳 中文界面: 全中文化的用户界面和分析结果。
    *   🎛️ 配置管理:  API密钥管理、模型选择、系统配置，便捷管理。
    *   💰 Token 统计:  实时 Token 使用统计，成本追踪，透明高效。
    *   💾 缓存管理:  数据缓存状态监控和管理，提升性能。

*   **高性能特性**:  为用户提供极速、稳定的分析体验。

    *   ⚡️ 并行处理:  多智能体并行分析，提高分析效率。
    *   🔄 智能缓存:  多层缓存策略，减少 API 调用成本。
    *   🔄 高可用架构:  多层数据源降级，确保服务稳定性。

*   **专业报告导出**:  多种报告格式，满足不同场景需求。

    *   📝 Markdown: 轻量级格式，版本控制友好。
    *   📄 Word: 专业格式，可编辑，商务报告标准。
    *   📊 PDF:  固定格式，打印友好，正式发布。

*   **部署便捷**: Docker 部署简化流程，降低使用门槛。

    *   🐳 Docker 支持: 快速部署，环境隔离。
    *   🔧 Docker Compose: 一键部署，开发环境。

## 🚀 快速开始

### 🐳 Docker 部署 (推荐)

1.  **克隆项目:**  `git clone https://github.com/hsliuping/TradingAgents-CN.git`
2.  **配置环境变量:** `cp .env.example .env`，并编辑 .env 文件，填入 API 密钥。
3.  **构建并启动:**  `docker-compose up -d --build` (首次运行需 5-10 分钟构建镜像)。
4.  **访问应用:**
    *   Web 界面: `http://localhost:8501`
    *   数据库管理: `http://localhost:8081`
    *   缓存管理: `http://localhost:8082`

### 💻 本地部署

1.  **克隆项目:**  `git clone https://github.com/hsliuping/TradingAgents-CN.git`
2.  **创建虚拟环境:** `python -m venv env`  然后激活虚拟环境。
3.  **安装依赖:**  `pip install -r requirements.txt`
4.  **配置API密钥**: 复制 `.env.example` 为 `.env`，并编辑 .env 文件，配置 API 密钥。

    *   推荐使用阿里百炼，也支持 OpenAI、Google AI 等。
    *   配置 Tushare API 以获得专业的 A股数据。

5.  **启动 Web 界面:**  `streamlit run web/app.py`，在浏览器中访问 `http://localhost:8501`。

**请务必配置好 API 密钥和数据库，才能正常运行。**  更多详情请参阅 [快速开始](docs/overview/quick-start.md) 文档。

## 📚 详细文档

我们的文档提供了全面的指南，助您深入理解和使用 TradingAgents-CN。

*   **[项目概述](docs/overview/project-overview.md)**：介绍项目背景和核心价值。
*   **[快速开始](docs/overview/quick-start.md)**：10分钟上手指南。
*   **[系统架构](docs/architecture/system-architecture.md)**：深度解析系统架构。
*   **[智能体架构](docs/architecture/agent-architecture.md)**：详细介绍多智能体协作机制。
*   **[数据流架构](docs/architecture/data-flow-architecture.md)**：数据处理全流程。
*   **[配置指南](docs/configuration/config-guide.md)**：详细配置选项说明。

>   **我们与原版最大的区别是：我们提供了业界最完整的中文文档体系！**

## 🤝 贡献指南

我们欢迎各种形式的贡献！ 详见 [贡献指南](CONTRIBUTING.md)

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。

## 🙏 致谢

感谢 [Tauric Research](https://github.com/TauricResearch) 团队，感谢开源社区的贡献者，特别是 Docker容器化和报告导出功能的贡献者。

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)

</div>