# 📈 TradingAgents-CN: 中文金融交易决策框架 -  基于AI的多智能体系统，专为中国市场优化，助力智能投资决策！

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.12-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

> **TradingAgents-CN** 🚀 基于多智能体大语言模型 (LLM)，为中国用户提供 **强大的中文金融交易决策框架**，支持 A股、港股、美股分析，并集成智能新闻分析模块，助力用户做出更明智的投资决策。

## ✨ 核心特性

*   **🤖 多智能体协作架构:** 模拟专业分析师团队，提供多角度分析和决策。
*   **🇨🇳 中国市场优化:** 专为A股/港股/美股市场设计，支持Tushare, AkShare等数据源。
*   **🧠 智能新闻分析 (v0.1.12):**  AI驱动的新闻过滤、质量评估与相关性分析，支持多层过滤机制。
*   **🆕 多LLM提供商集成 (v0.1.11):** 集成包括阿里百炼、DeepSeek、Google AI、OpenRouter在内的多种LLM提供商。
*   **💾 模型选择持久化 (v0.1.11):** 支持URL参数存储，方便分享模型配置。
*   **🐳 Docker 容器化部署:** 一键部署，环境隔离，易于扩展。
*   **📄 专业报告导出:** 支持多种格式 (Markdown, Word, PDF)，生成投资建议。
*   **📊 实时进度显示 (v0.1.10):** 异步进度跟踪，告别等待。

## 🚀 主要功能

*   **智能新闻分析:** 基于AI的新闻过滤、质量评估、相关性分析、多层过滤。
*   **多LLM支持:** 阿里百炼、DeepSeek、Google AI、OpenRouter (60+模型)。
*   **A股/港股/美股数据:** 实时行情、财务数据，全面覆盖。
*   **Web 界面:** 现代化响应式界面，实时交互，数据可视化。
*   **配置管理:** Web端API密钥管理，模型选择，参数配置。
*   **专业报告导出:** 生成Word/PDF/Markdown报告，包含决策建议。
*   **智能会话管理:** 状态持久化，页面刷新不丢失分析结果。
*   **用户友好的CLI界面 (v0.1.9):** 更清晰的输出，智能进度显示。

## 🆕 最新更新 (v0.1.12)

*   **🧠 智能新闻分析模块全面升级:**  新增多层次新闻过滤、质量评估、相关性分析，支持A股/港股/美股新闻智能处理！
*   **🔧 技术修复和优化:** 修复DeepSeek死循环问题，提升工具调用可靠性，增强新闻检索能力。
*   **📚 完善测试和文档:** 全面测试覆盖，详细技术分析报告和用户指南。

## 🚀 快速开始

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量 (API 密钥)
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
docker-compose up -d --build  # 首次或代码变更
# docker-compose up -d  # 日常启动

# 4. 访问应用
# Web界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 升级pip
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -e .

# 3. 启动应用
python start_web.py

# 4. 访问 http://localhost:8501
```

### 📈 立即开始分析

1.  选择模型 (DeepSeek V3 / 通义千问 / Gemini 等)
2.  输入股票代码 (e.g., 000001, AAPL, 0700.HK)
3.  点击 "🚀 开始分析"
4.  实时跟踪进度
5.  查看生成的 "📊 分析报告"
6.  导出报告

## 🤝 致敬与感谢

本项目的核心框架基于 [Tauric Research/TradingAgents](https://github.com/TauricResearch/TradingAgents)，感谢团队的开源贡献！

## 📚 详细文档

访问 [./docs/](docs/) 获取更详细的安装、使用、架构和技术文档。  **[50,000+ 字的中文文档，助你深入理解!](docs/)**

## 📄 许可证

本项目采用 Apache 2.0 许可证。 详见 [LICENSE](LICENSE)。

## 🔗 源码

访问 [hsliuping/TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN)  获取源代码。