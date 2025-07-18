# TradingAgents-CN: 中文金融交易决策框架 ✨

> 🚀 **利用人工智能赋能您的金融交易！** TradingAgents-CN 基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)，专为中文用户优化，提供 A 股支持、国产大模型集成、专业报告导出和 Docker 容器化部署。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.9-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## 🌟 主要特性

*   **🇨🇳 A 股支持**: 实时行情、历史数据、国产数据源集成
*   **🧠 国产大模型**: 阿里云百炼、DeepSeek、Gemini 等大模型集成
*   **🌐 中文界面**: 全中文用户界面和分析结果
*   **🐳 Docker 部署**: 快速、便捷的容器化部署
*   **📄 专业报告导出**:  Markdown、Word、PDF 多种格式专业报告
*   **🤖 多智能体协作**: 模拟真实交易公司的专业分工与决策流程

## 🚀 核心优势

*   **开箱即用**:  完整的 Web 界面，无需命令行操作
*   **中国优化**: A 股数据、国产 LLM、中文界面
*   **智能配置**:  自动检测、智能降级、零配置启动
*   **实时监控**:  Token 使用统计、缓存状态、系统监控
*   **稳定可靠**:  多层数据源、错误恢复、生产就绪
*   **容器化**:  Docker 部署，环境隔离，快速扩展
*   **专业报告**: 多格式导出，自动生成

## ✨ 最新版本 v0.1.9  更新亮点

*   **🎨 CLI用户体验重构**: 界面与日志分离，提供清爽专业的用户体验
*   **🔄 智能进度显示**: 解决重复提示问题，添加多阶段进度跟踪
*   **⏱️ 时间预估功能**: 智能分析阶段显示"预计耗时约10分钟"，管理用户期望
*   **📝 统一日志管理**: LoggingManager + TOML配置 + 工具调用记录
*   **🇭🇰 港股数据优化**: 改进数据获取稳定性和容错机制
*   **🔑 配置问题修复**: 解决OpenAI配置混乱，统一API密钥管理

## 📚 详细文档

> **与原项目最大的区别！** 我们提供了业界最完整的中文金融 AI 框架文档体系，包含超过 **50,000 字** 的详细技术文档。

*   [🚀 快速开始](docs/overview/quick-start.md) - 快速上手指南
*   [🏛️ 系统架构](docs/architecture/system-architecture.md) - 深度理解系统设计
*   [🤖 智能体详解](docs/agents/analysts.md) - 核心组件详解
*   [🌐 Web界面指南](docs/usage/web-interface-guide.md) - 完整的 Web 界面使用教程
*   [❓ 常见问题](docs/faq/faq.md) - 详细的 FAQ 和故障排除指南

[📚 更多文档，请访问完整文档目录](./docs/)

## 🚀 快速开始

### 🐳 Docker 部署 (推荐)

```bash
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN
cp .env.example .env
# 编辑 .env 文件，填入API密钥
docker-compose up -d --build
# 访问: http://localhost:8501 (Web界面)
```

### 💻 本地部署

```bash
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN
python -m venv env
# ... (安装依赖, 配置API密钥, 参考readme.md)
streamlit run web/app.py
# 访问: http://localhost:8501 (Web界面)
```

## 🤝 贡献

我们欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何贡献。

## 📄 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 开源。

## 🙏 致谢

感谢 [Tauric Research](https://github.com/TauricResearch) 团队的贡献，以及所有贡献者和用户。

## 📞 联系

*   **GitHub Issues**: [提交问题和建议](https://github.com/hsliuping/TradingAgents-CN/issues)
*   **原项目**: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)

## ⚠️ 免责声明

本项目仅用于研究和教育目的，不构成投资建议。投资有风险，请谨慎决策。

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [📖 Read the docs](./docs/)

</div>