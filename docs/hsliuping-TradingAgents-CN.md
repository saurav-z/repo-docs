# TradingAgents-CN: 中文金融AI交易框架

🚀 **利用多智能体大语言模型，TradingAgents-CN 助力您深入分析 A 股、港股和美股，做出更明智的投资决策!**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**TradingAgents-CN** 是一个基于多智能体大语言模型的**中文金融交易决策框架**，专为中国用户优化，提供 A 股、港股和美股的全面分析能力。 它在原版 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 的基础上，针对中国市场和用户需求进行了增强和改进。

## ✨ 主要特性

*   **🇨🇳 中文支持**:  全面支持中文界面和A股、港股数据。
*   **🤖 多智能体架构**:  基本面、技术面、新闻面、社交媒体四大分析师协作。
*   **🌐 多 LLM 支持**:  集成多个大语言模型提供商，包括 OpenAI, Google AI 等。
*   **📈 智能新闻分析**:  AI驱动的新闻过滤和质量评估，快速获取市场信息。
*   **🐳 容器化部署**:  Docker 一键部署，环境隔离，轻松上手。
*   **📊 专业报告导出**:  支持 Markdown, Word, PDF 等格式，方便分享和分析。
*   **💻 现代化 Web 界面**: 直观的股票分析体验，实时进度跟踪。

## 🆕 最新版本 cn-0.1.13-preview 更新亮点

*   **🤖 原生 OpenAI 支持**:  支持自定义 OpenAI 端点，灵活选择模型，提供更好的兼容性和性能。
*   **🧠 Google AI 集成**:  全面支持 Google AI 生态系统，包括 Gemini 2.5 系列模型，提供更强大的分析能力。
*   **🔧 LLM 适配器架构优化**:  统一的 LLM 调用接口，增强错误处理和性能监控。
*   **🎨 Web 界面智能优化**:  智能模型选择、UI 响应优化、更友好的错误提示。

## 💡 核心功能

*   **智能分析配置**：支持 A 股、港股、美股，提供多级研究深度。
*   **实时进度跟踪**：可视化分析过程，预估剩余时间。
*   **专业结果展示**：明确的买入/持有/卖出建议，多维度分析结果，一键导出报告。
*   **多 LLM 模型管理**：灵活选择模型，配置持久化，快速切换。

## 🚀 快速开始 (Docker 部署)

```bash
# 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥 (强烈建议)

# 启动服务
docker-compose up -d --build  # 首次构建镜像
# 或 docker-compose up -d      # 之后启动

# 访问
# Web 界面: http://localhost:8501
```

## 📚 文档与支持

*   [📖 完整文档](./docs/):  全面、详细的中文文档，从入门到精通。
*   [🚨 故障排除](./docs/troubleshooting/):  常见问题解决方案。
*   [🚀 快速开始](./QUICKSTART.md):  5 分钟快速部署指南。
*   **GitHub Issues**: [提交问题和建议](https://github.com/hsliuping/TradingAgents-CN/issues)

## 🙏 致谢

感谢 [Tauric Research](https://github.com/TauricResearch) 团队创建了 [TradingAgents](https://github.com/TauricResearch/TradingAgents)。 TradingAgents-CN 旨在为中国用户提供一个更适合中文环境的金融交易决策框架，推动 AI 金融技术在中文社区的普及应用。

## 🤝 贡献

我们欢迎各种形式的贡献。  查看 [CONTRIBUTORS.md](CONTRIBUTORS.md) 了解贡献者名单。

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN)