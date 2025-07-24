# 🇨🇳 中文金融 AI 交易框架：TradingAgents-CN

> 🚀 **基于多智能体大语言模型的中文金融交易框架**，专为中文用户优化，提供 A 股/港股/美股分析能力，并集成实时进度显示和智能会话管理。 [查看原项目 (TauricResearch/TradingAgents)](https://github.com/TauricResearch/TradingAgents)。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.10-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## ✨ 核心特性

*   **🇨🇳 中文优化**: 深度适配中文语境，提供更流畅的使用体验。
*   **📊 全面市场支持**: 涵盖 A 股、港股、美股市场数据。
*   **🤖 多智能体架构**: 专业分析师团队协同工作，进行深度分析和决策。
*   **🚀 实时进度显示**: 告别黑盒，异步跟踪分析进度，掌控每一步。
*   **💾 智能会话管理**: 分析状态持久化，页面刷新不丢失结果。
*   **🐳 容器化部署**: Docker 一键部署，环境隔离，方便扩展。
*   **📄 专业报告导出**:  支持 Word/PDF/Markdown 格式，自动生成投资建议。
*   **🧠 LLM 模型支持**: 灵活集成多种大语言模型，包括 DeepSeek V3、阿里百炼、OpenAI 等。

## 🆕 v0.1.10 更新亮点

*   **🚀 实时进度显示系统**:  异步进度跟踪，智能时间计算，多种显示模式。
*   **📊 智能会话管理**: 状态持久化，自动降级，一键查看报告。
*   **🎨 用户体验优化**: 界面简化，响应式设计，增强错误处理。

## 🎯 核心功能

*   **🤖 多智能体协作架构**:  专业分析师团队（市场、基本面、新闻、情绪），研究团队（看涨/看跌），交易员决策，风险管理。
*   **🚀 Web 界面增强**: 实时进度、智能会话、一键报告查看、界面优化。
*   **🧠 LLM 模型支持**: 阿里百炼、DeepSeek、Google AI、OpenAI 等。
*   **📊 数据源与市场**: A 股 (Tushare, AkShare)、港股 (AkShare, Yahoo Finance)、美股 (FinnHub, Yahoo Finance)、新闻 (Google News)。

## 🚀 快速开始

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API 密钥

# 3. 启动服务
docker-compose up -d --build

# 4. 访问应用
# Web 界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python start_web.py

# 3. 访问 http://localhost:8501
```

### 📊 开始分析

1.  **选择模型**: DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票**: `000001` (A股) / `AAPL` (美股) / `0700.HK` (港股)
3.  **开始分析**: 点击"🚀 开始分析"按钮
4.  **实时跟踪**: 观察实时进度和分析步骤
5.  **查看报告**: 点击"📊 查看分析报告"按钮
6.  **导出报告**: 支持 Word/PDF/Markdown 格式

## 🔧 技术架构

*   **核心技术**: Python 3.10+ | LangChain | Streamlit | MongoDB | Redis
*   **AI 模型**: DeepSeek V3 | 阿里百炼 | Google AI | OpenAI
*   **数据源**: Tushare | AkShare | FinnHub | Yahoo Finance
*   **部署**: Docker | Docker Compose | 本地部署

## 📚 文档和支持

*   **📖 完整文档**: [docs/](./docs/) - 安装、使用、API 文档
*   **🚨 故障排除**: [troubleshooting/](./docs/troubleshooting/) - 常见问题解决方案
*   **🔄 更新日志**: [CHANGELOG.md](./docs/releases/CHANGELOG.md) - 版本历史
*   **🚀 快速开始**: [QUICKSTART.md](./QUICKSTART.md) - 快速部署指南

## 🆚 中文增强特色

*   实时进度显示 | 智能会话管理 | 中文界面 | A 股数据 | 国产 LLM | Docker 部署 | 专业报告导出 | 统一日志管理 | Web 配置界面 | 成本优化

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)

</div>