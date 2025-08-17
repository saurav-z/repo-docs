# TradingAgents-CN: 中文金融交易决策框架 (基于多智能体大语言模型)

> **解锁中国股市投资潜力！** TradingAgents-CN 是一个专为中文用户优化的金融交易决策框架，基于先进的多智能体大语言模型，提供全面的 A 股、港股和美股分析能力，助力您的投资决策。  [查看原项目](https://github.com/hsliuping/TradingAgents-CN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## ✨ 主要特点

*   **🤖  多智能体架构**:  专业分析师团队（基本面、技术面、新闻面、情绪面）协同工作，提供深度分析和投资建议。
*   **🇨🇳  中文优化**:  深度支持 A 股、港股市场，并集成国产大语言模型，打造本土化体验。
*   **🆕  最新版本 (cn-0.1.13-preview)**: 原生 OpenAI 和 Google AI 全面集成，支持自定义端点和多种模型。
*   **📊  Web 界面**: 简洁直观的 Streamlit 界面，方便您配置、分析和查看报告。
*   **🚀  快速部署**:  支持 Docker 一键部署，本地部署，轻松上手。
*   **📈  专业报告**:  生成 Markdown、Word 和 PDF 格式的专业投资报告，方便分享和归档。
*   **💰  成本控制**: 多种 LLM 模型选择，灵活配置，优化分析成本。

## 🔑 核心功能

*   **📰 智能新闻分析**:  基于 AI 的新闻过滤和质量评估，筛选关键信息。
*   **🤖  多 LLM 模型支持**:  支持 OpenAI、Google AI、阿里百炼、DeepSeek 和 OpenRouter 等多种模型。
*   **💾  模型选择持久化**:  URL 参数存储，方便您保存和分享配置。
*   **🚀  实时进度显示**:  清晰的分析进度跟踪，告别黑盒等待。
*   **🇨🇳  中国市场数据**:  全面支持 A 股、港股和美股，实时行情和基本面数据。
*   **🐳  Docker 部署**:  轻松实现环境隔离，快速部署和扩展。

## 🚀 快速上手

1.  **Docker 部署 (推荐)**
    ```bash
    git clone https://github.com/hsliuping/TradingAgents-CN.git
    cd TradingAgents-CN
    cp .env.example .env  # 编辑 .env 文件，配置 API 密钥
    docker-compose up -d --build # 首次启动或代码变更时
    docker-compose up -d # 日常启动
    # Web 界面: http://localhost:8501
    ```

2.  **本地部署**
    ```bash
    pip install -e .  # 安装依赖
    python start_web.py  # 启动应用
    # 访问 http://localhost:8501
    ```

## 📚 深度学习与资源

*   **📖 完整文档**:  深入了解项目的各个方面，包括安装、使用、架构和 API，文档目录: `./docs/`
*   **🚀 示例代码**:  提供多种代码示例，帮助您快速上手和扩展，示例目录: `./examples/`
*   **🤝 贡献指南**:  了解如何为本项目贡献代码，贡献者名单: `./CONTRIBUTORS.md`
*   **📜 许可证**: Apache 2.0 许可证，允许自由使用和修改，[LICENSE](LICENSE)

## 💡 最新更新 (cn-0.1.13-preview)

*   **🤖 原生OpenAI集成**: 支持自定义端点，灵活模型选择。
*   **🧠 Google AI 生态集成**:  全面支持 Google AI 模型，包含 Gemini 2.5 系列。
*   **🔧 LLM 适配器架构优化**:  统一接口，增强错误处理和性能监控。
*   **🎨 Web 界面优化**:  智能模型选择、KeyError 修复，提升用户体验。

## 📈 更多信息

*   **项目ＱＱ群：782124367**
*   **联系我们**: hsliup@163.com
*   **原项目**: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
*   **欢迎提交问题和建议**: [GitHub Issues](https://github.com/hsliuping/TradingAgents-CN/issues)

---

<div align="center">
  **如果您觉得这个项目对您有帮助，请给个 Star 吧！**
  <br>
  [⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN)
</div>