# TradingAgents-CN: AI驱动的中文金融交易决策框架 🚀

> 🚀 **基于多智能体大语言模型的中文金融交易决策框架，支持A股、港股和美股，助您智能分析股票！**  [访问原始项目](https://github.com/hsliuping/TradingAgents-CN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

TradingAgents-CN 是一个**专为中文用户优化的金融交易决策框架**，它利用**多智能体大语言模型 (LLMs)** 来分析股票市场。  此项目基于 [Tauric Research 的 TradingAgents](https://github.com/TauricResearch/TradingAgents)，并针对中国市场进行了增强，提供**全面的A股/港股/美股分析能力**，并加入了**原生OpenAI 和 Google AI 支持**。

## ✨ 核心特性

*   🤖 **多智能体架构**: 模拟专业分析师进行全方位分析，包括基本面、技术面、新闻面、社交媒体分析，生成买入/持有/卖出建议。
*   🇨🇳 **中文优化**:  专为中文用户设计，提供流畅的中文界面和支持中文市场数据。
*   📊 **市场覆盖**:  全面支持A股、港股和美股市场。
*   🌐 **多LLM支持**:  支持阿里百炼、DeepSeek、Google AI、原生OpenAI、OpenRouter等多种LLM提供商，提供灵活选择。
*   🚀 **Web界面**:  提供直观的Web界面，方便用户配置、分析和查看结果。
*   🐳 **Docker 部署**:  一键Docker部署，快速搭建分析环境。
*   🎯 **智能新闻分析**: 引入AI新闻过滤和质量评估，提高分析准确性（v0.1.12）。
*   📈 **报告导出**: 支持生成Markdown, Word, 和 PDF 格式的专业分析报告。
*   🔑 **模型选择持久化**:  URL参数存储，刷新保持，配置分享 (v0.1.11)。
*   ⚡ **实时进度**: 异步进度跟踪，实时查看分析过程 (v0.1.10)。

## 🆕 最新更新 (v0.1.13 - 预览版)

*   🤖 **原生OpenAI支持**:  自定义OpenAI端点、灵活模型选择、智能适配器。
*   🧠 **Google AI全面集成**:  集成三大Google AI包，支持9个验证模型。
*   🔧 **LLM适配器架构优化**: 统一接口和增强的错误处理。
*   🎨 **Web界面智能优化**:  智能模型选择、UI响应速度优化。

## 🚀 快速开始 (Docker部署)

1.  **克隆项目**: `git clone https://github.com/hsliuping/TradingAgents-CN.git`
2.  **进入目录**: `cd TradingAgents-CN`
3.  **配置 API 密钥**:  编辑 `.env` 文件，填写您的API密钥。
4.  **启动服务**:  `docker-compose up -d --build` (首次或代码变更), 或 `docker-compose up -d` (日常启动)。
5.  **访问 Web界面**:  在浏览器中打开 `http://localhost:8501`

## 📚 文档和支持

*   📖 **完整文档**:  [查看文档](docs/) - 包含安装、使用教程、API文档等。
*   🚨 **故障排除**:  [查看常见问题](docs/troubleshooting/)
*   🔄 **更新日志**:  [查看更新日志](docs/releases/CHANGELOG.md)

## 🤝 贡献指南

我们欢迎贡献!  请参考 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 开源。

---

如果您觉得这个项目对您有帮助，请考虑给个 [⭐ Star](https://github.com/hsliuping/TradingAgents-CN)！