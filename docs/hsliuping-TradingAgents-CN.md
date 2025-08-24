# 📈 TradingAgents-CN: 中文金融交易决策框架 🚀

> 💡 **利用多智能体大语言模型，为中国市场量身定制的金融交易决策框架，助您深入分析 A 股、港股和美股。**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## ✨ 主要特性

*   🤖 **原生OpenAI支持 & Google AI 全面集成**: 利用最新 LLM 模型，提升分析能力。
*   📰 **智能新闻分析**: 筛选关键新闻，洞悉市场情绪。
*   🇨🇳 **A 股 / 港股 / 美股全面支持**:  覆盖中国、香港和美国市场。
*   🐳 **Docker 容器化部署**:  简化安装和环境配置。
*   📊 **专业报告导出**: 生成 Markdown / Word / PDF 格式的专业分析报告。
*   📝 **完整中文文档**: 深入解析架构与使用方法。

## 🌟 核心功能

*   **多智能体协作**: 由基本面、技术面、新闻面和情绪分析师组成，做出综合决策。
*   **结构化分析**:  支持看涨/看跌研究员进行深度辩论。
*   **智能决策**: 基于全面的数据分析，生成买入、持有或卖出建议。
*   **风险管理**:  多维度风险评估，保护您的投资。

## 🚀 最新动态:  cn-0.1.13-preview

**全面集成 OpenAI 和 Google AI 生态系统，增强了 LLM 模型支持和灵活性**

*   **原生 OpenAI 支持**: 支持自定义 OpenAI 端点，使用任何 OpenAI 兼容模型。
*   **Google AI 集成**:  集成 Google AI，包含 Gemini 系列模型。
*   **LLM 适配器优化**: 统一接口，改进错误处理和性能监控。
*   **Web 界面优化**:  智能模型选择，提升用户体验。

## 🛠️ 快速上手

### 🐳 Docker 部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
docker-compose up -d --build

# 4. 访问应用
# Web界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 升级pip (重要!)
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -e .

# 3. 启动应用
python start_web.py

# 4. 访问 http://localhost:8501
```

### 📊 开始分析

1.  **选择模型**: DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票**: `000001` (A股) / `AAPL` (美股) / `0700.HK` (港股)
3.  **开始分析**: 点击"🚀 开始分析"
4.  **查看报告**: 浏览分析报告
5.  **导出报告**: 下载专业报告

## 📚 深入了解

*   **完整文档**:  [中文文档](./docs/) - 涵盖安装、使用、架构和 API。
*   **贡献指南**:  欢迎参与项目贡献： [CONTRIBUTING.md](CONTRIBUTING.md)
*   **问题反馈**:  提交问题和建议： [GitHub Issues](https://github.com/hsliuping/TradingAgents-CN/issues)

**基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 的代码，旨在为中国用户提供更完善、更本土化的金融交易决策工具。**