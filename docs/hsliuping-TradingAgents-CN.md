# TradingAgents-CN: 中文增强版 - 基于多智能体LLM的金融交易框架

> 🚀 **开启AI驱动的智能金融交易！** TradingAgents-CN 是一个强大的中文金融交易决策框架，基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 构建，为中文用户提供全面的文档、A股数据支持、国产LLM集成和便捷的Web界面。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.7-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**Key Features:**

*   **🇨🇳 中文支持**: 全中文文档、界面和分析结果，降低使用门槛。
*   **🧠 国产LLM集成**: 集成阿里百炼、Google AI等LLM，适应国内环境。
*   **📊 A股数据支持**: 实时行情、历史数据和技术指标。
*   **🌐 Web管理界面**: 现代化Streamlit界面，易于配置和监控。
*   **🗄️ 数据库集成**: MongoDB持久化、Redis缓存，提升性能。
*   **📤 专业报告导出**: 支持Markdown/Word/PDF等多格式报告导出。
*   **🚀 高性能架构**: 多智能体并行处理、智能缓存、数据降级机制。

## 核心优势

*   **开箱即用**: 完整的Web界面，无需命令行操作。
*   **中国优化**: A股数据 + 国产LLM + 中文界面。
*   **智能配置**: 自动检测、智能降级、零配置启动。
*   **实时监控**: Token使用统计、缓存状态、系统监控。
*   **稳定可靠**: 多层数据源、错误恢复、生产就绪。

## 🚀 项目目标

TradingAgents-CN旨在为中国金融领域提供一个先进的AI交易框架，实现以下目标：

*   **🇨🇳 技术本土化**: 提供中文文档、界面，支持A股、港股市场。
*   **🎓 教育推广**: 为高校和研究机构提供AI金融教学工具。
*   **🤝 社区建设**: 建立中文开发者社区，促进交流与合作。
*   **🚀 创新应用**: 推动AI技术在中国金融科技领域的应用。

## 核心特性

*   **🤖 多智能体协作架构**: 模拟真实交易公司的专业分工。
*   **🧠 多LLM模型支持**: 支持阿里百炼、Google AI、OpenAI、Anthropic。
*   **📊 全面数据集成**: A股、美股、新闻、社交数据，数据库支持。
*   **🚀 高性能特性**: 并行处理、智能缓存、灵活配置。
*   **🌐 Web管理界面**: 直观操作，实时进度，智能配置，token统计。

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 创建虚拟环境
python -m venv env
# Windows
env\Scripts\activate
# Linux/macOS
source env/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API 密钥
cp .env.example .env
# 编辑 .env 文件，配置API密钥，例如：
# DASHSCOPE_API_KEY=your_dashscope_api_key_here
# FINNHUB_API_KEY=your_finnhub_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here

# 5. 启动 Web 界面
streamlit run web/app.py
```

访问 `http://localhost:8501` 开始使用。

## 📚 深入了解

*   **文档**:  [中文文档](docs/) - 涵盖项目架构、使用指南、常见问题解答。
*   **示例**:  `examples/` - 提供多种使用示例。

## 🤝 贡献指南

欢迎贡献代码、文档、反馈和建议。请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。

---

**🌟 立即体验，开启智能金融交易之旅！**
[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [📖 阅读文档](./docs/)