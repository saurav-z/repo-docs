# TradingAgents-CN: AI驱动的中文金融交易框架

🤖 **利用多智能体大语言模型，提升您的中文金融交易决策！**  基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)，TradingAgents-CN 提供了针对中国市场的全面增强，包括 A 股支持、国产大模型集成和 Docker 容器化部署， 助您轻松驾驭金融市场。

## 🚀 核心特性

*   🇨🇳 **A 股市场支持**: 完整的 A 股数据、实时行情与历史数据。
*   🧠 **国产大模型集成**: 阿里百炼、DeepSeek 等模型，成本更低，响应更快。
*   🐳 **Docker 容器化**: 一键部署，环境隔离，方便快捷。
*   🌐 **Web 界面**: 直观的 Streamlit 界面，实时监控与配置管理。
*   📄 **专业报告导出**: Markdown, Word, PDF 多种格式，便于分享与分析。
*   💡 **智能体协作**: 基本面、技术面、新闻面、情绪面分析师协同决策。

## ✨ 主要优势

*   开箱即用：完整的 Web 界面，无需命令行操作。
*   中国优化：A 股数据 + 国产 LLM + 中文界面。
*   智能配置：自动检测、智能降级，零配置启动。
*   实时监控：Token 使用统计、缓存状态、系统监控。
*   稳定可靠：多层数据源、错误恢复、生产就绪。
*   容器化：Docker 部署，环境隔离，快速扩展。
*   专业报告：多格式导出，自动生成。

## 📚 快速上手 & 完整文档

快速开始：

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置API密钥 (编辑 .env.example)
cp .env.example .env
# 编辑 .env 文件，填入API密钥
# 3. 构建并启动所有服务 (推荐 Docker)
docker-compose up -d --build
```

查看详细文档，获取更深入的理解与操作指南:

*   [完整文档目录](docs/)

## 🆚 主要区别

*   **增强**: 完整的中文文档体系 + A 股支持 + 国产 LLM 集成 + Docker 部署 +  专业报告导出。
*   **优化**:  Web 界面、用户体验、配置管理、成本控制、架构。

## 🤝 贡献

我们欢迎贡献！  请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)