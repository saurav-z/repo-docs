# TradingAgents-CN: 中文金融交易决策框架 🚀

**利用先进的 AI，解锁中文金融市场的强大分析能力，基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 构建。**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.12-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

---

## 🌟 核心特性

*   **🤖 多智能体协作:**  基本面、技术面、新闻面、情绪面四大分析师协同工作。
*   **🧠 智能新闻分析 (v0.1.12):** AI驱动的新闻过滤、质量评估和相关性分析。
    *   **智能新闻过滤器:** 基于AI的新闻相关性评分和质量评估
    *   **多层次过滤机制:** 基础过滤、增强过滤、集成过滤三级处理
    *   **新闻质量评估:** 自动识别和过滤低质量、重复、无关新闻
    *   **统一新闻工具:** 整合多个新闻源，提供统一的新闻获取接口
*   **🆕 多 LLM 提供商 (v0.1.11):**  支持 4 大 LLM 提供商，60+ 模型。
    *   **LLM 支持**:  阿里百炼、DeepSeek、Google AI、OpenRouter (OpenAI, Anthropic, Meta, Google, Custom)。
*   **💾 模型选择持久化 (v0.1.11):**  基于 URL 的模型配置存储，刷新不丢失。
*   **🇨🇳 中国市场支持:** 完整 A 股、港股和美股市场数据。
*   **🐳 Docker 部署:** 一键式容器化部署，快速启动和扩展。
*   **📊 报告导出:** 支持 Word/PDF/Markdown 格式的专业分析报告。

## 🚀 主要更新 (v0.1.12)

*   **🧠 智能新闻分析模块**:  AI驱动的新闻过滤和质量评估系统，助力更精准的决策。
    *   **🔧 新闻过滤器**:  多层次过滤机制，基础、增强、集成三级处理。
    *   **📰 统一新闻工具**:  整合多源新闻，提供统一的智能检索接口。

## 快速开始 🚀

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2.  配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3.  构建并启动
docker-compose up -d --build  # 首次或代码变更
docker-compose up -d           # 日常启动

# 4.  访问应用
# Web界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1.  升级pip
python -m pip install --upgrade pip

# 2.  安装依赖
pip install -e .

# 3.  启动应用
python start_web.py

# 4.  访问: http://localhost:8501
```

## 📚 详细文档

本项目拥有 **最全面的中文文档**， 助您深入理解和定制:

*   [🚀 快速开始](docs/overview/quick-start.md):  5分钟上手指南
*   [🏛️ 系统架构](docs/architecture/system-architecture.md): 深入了解系统设计
*   [🤖 智能体架构](docs/architecture/agent-architecture.md):  多智能体协作机制
*   [📖 完整文档目录](docs/README.md):  查看所有文档

## 🤝 贡献

欢迎通过 Pull Requests 贡献代码、文档、翻译等。查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 Apache 2.0 许可证.

---
**[访问原始仓库](https://github.com/hsliuping/TradingAgents-CN) 了解更多!**