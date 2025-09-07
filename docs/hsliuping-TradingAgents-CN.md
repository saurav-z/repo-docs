# TradingAgents-CN: 中文金融交易决策框架 (增强版) 📈

> 🚀  **使用AI驱动的智能体，分析股票市场并生成专业的投资建议！**  基于TauricResearch/TradingAgents，为中国市场量身定制，支持A股、港股和美股，提供全面的AI驱动金融分析。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/Based%20on-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

[**查看原始仓库: TauricResearch/TradingAgents**](https://github.com/TauricResearch/TradingAgents)

## 🔑 核心特性

*   **🌐 全面中文支持**: 专为中文用户设计，提供本地化体验。
*   **🧠 智能多智能体**:  基本面、技术面、新闻面、社交媒体分析师协同工作。
*   **📈 多市场分析**:  全面支持A股、港股和美股。
*   **🤖 多LLM模型**: 支持阿里百炼、DeepSeek、Google AI、OpenRouter 和 原生OpenAI。
*   **🚀 实时进度与报告**:  可视化分析进度，生成专业投资报告。
*   **🐳 Docker 部署**: 一键部署，环境隔离，快速扩展。

## ✨ 最新版本：cn-0.1.13-preview

### 🤖 原生OpenAI 支持和 Google AI 集成

*   **🆕 原生 OpenAI 支持**:  自定义 OpenAI 端点，灵活的模型选择。
*   **🧠 全面 Google AI 集成**:  支持 gemini-2.5-pro, gemini-2.5-flash 等模型。
*   **🔧 LLM 适配器优化**:  统一的调用接口，增强的错误处理。
*   **🎨 Web 界面优化**:  智能模型选择，提升用户体验。

###  主要更新 (v0.1.12)

*   **🧠 智能新闻分析**: AI 驱动的新闻过滤和相关性评估。
*   **📰 统一新闻工具**: 整合多源新闻，提供智能检索。
*   **📚 完善文档与测试**:  新增技术文档和测试用例。

## 🚀 快速开始

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
docker-compose up -d --build  # 首次构建
docker-compose up -d          # 日常启动

# 4. 访问应用
# Web界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 安装依赖
pip install -e .

# 2. 启动应用
python start_web.py

# 3. 访问 http://localhost:8501
```

### 📊 开始分析

1.  **选择模型**: DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票**: `000001` (A股) / `AAPL` (美股) / `0700.HK` (港股)
3.  **开始分析**:  点击"🚀 开始分析"
4.  **查看报告**: 查看专业分析报告

## 🎯 核心功能

*   **📊 智能分析配置**: 多市场支持，5级研究深度，智能体选择。
*   **🚀 实时进度跟踪**: 可视化分析流程，准确时间预估。
*   **📈 专业结果展示**:  买入/持有/卖出建议，多维分析，专业报告。
*   **🤖 多LLM模型管理**: 支持 4 大提供商，60+ 模型。
*   **📰 智能新闻分析**:  AI新闻过滤，质量评估，相关性分析。

## 🖥️ Web 界面截图

![Web界面截图](images/README/1755003162925.png)
![Web界面截图](images/README/1755002901204.png)
*（更多截图请参考原README）*

## 📚 文档

*   **中文文档**:  [docs/](./docs/)  - 详细安装、使用教程、API文档。

## 🤝 贡献

欢迎提交 Bug 报告、新功能、文档改进等。 详见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 许可证

基于 [Apache 2.0 许可证](LICENSE) 开源。

---