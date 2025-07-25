# 🚀 TradingAgents-CN: AI驱动的中文金融交易框架 | A股/港股/美股分析

利用多智能体大语言模型，赋能中文金融交易决策，提供 A股、港股、美股全市场分析，深度优化中文用户体验。 [访问原项目: TauricResearch/TradingAgents](https://github.com/hsliuping/TradingAgents-CN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.10-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**核心特点:**

*   🇨🇳 **中文优化**: 全面支持A股、港股市场，中文界面和分析结果。
*   🚀 **v0.1.10 新增**: 实时分析进度、智能会话管理，Web界面全面升级。
*   🐳 **一键部署**: Docker容器化部署，快速启动和环境隔离。
*   📄 **专业报告**: 支持多种格式导出，生成投资建议。
*   🧠 **LLM 支持**: 集成 DeepSeek V3、阿里百炼、OpenAI 等模型。

## 🌟 主要特性

*   **实时分析进度**：异步进度跟踪，告别黑盒等待。
*   **智能会话管理**：页面刷新不丢失分析结果。
*   **A股/港股支持**：提供完整的A股和港股数据支持。
*   **国产LLM**: 优化中文分析效果。
*   **专业报告导出**：Word、PDF、Markdown多种格式导出。

## 🆕 v0.1.10 更新亮点

*   **🚀 实时进度显示系统**: 异步进度跟踪、智能时间计算、多种显示模式。
*   **📊 智能会话管理**: 状态持久化、自动降级机制、一键查看报告。
*   **🎨 用户体验优化**: 界面简化、响应式设计、错误处理增强。

## 🎯 核心功能

### 🤖 多智能体协作架构

*   **专业分工**：基本面、技术面、新闻面、情绪面四大分析师。
*   **结构化辩论**：看涨/看跌研究员进行深度分析。
*   **智能决策**：交易员基于所有输入做出最终投资建议。
*   **风险管理**：多层次风险评估和管理机制。

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
docker-compose up -d --build

# 4. 访问应用
# Web界面: http://localhost:8501
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
6.  **导出报告**: 支持Word/PDF/Markdown格式

## 🎯 核心优势

*   **实时进度**：异步分析，告别等待。
*   **智能会话**：分析结果持久化。
*   **🇨🇳 中国优化**：A股/港股数据、国产LLM、中文界面。
*   **🐳 容器化**：Docker一键部署。
*   **📄 专业报告**：多格式导出。
*   **🛡️ 稳定可靠**：多层数据源，错误恢复。

## 🧠 LLM 模型支持

*   🇨🇳 阿里百炼: `qwen-turbo/plus/max`
*   🇨🇳 DeepSeek: `deepseek-chat`
*   🌍 Google AI: `gemini-2.0-flash/1.5-pro`
*   🤖 OpenAI: `GPT-4o/4o-mini/3.5-turbo`

## 📊 数据源与市场

| 市场类型 | 数据源        | 覆盖范围                          |
| ---------- | ------------- | --------------------------------- |
| 🇨🇳 A股      | Tushare, AkShare | 沪深两市，实时行情，财报数据       |
| 🇭🇰 港股      | AkShare, Yahoo Finance | 港交所，实时行情，基本面         |
| 🇺🇸 美股      | FinnHub, Yahoo Finance | NYSE, NASDAQ，实时数据           |
| 📰 新闻      | Google News   | 实时新闻，多语言支持              |

## 📚 文档和支持

*   📖 **完整文档**: [docs/](./docs/) - 安装指南、使用教程、API文档
*   🚨 **故障排除**: [troubleshooting/](./docs/troubleshooting/) - 常见问题解决方案
*   🔄 **更新日志**: [CHANGELOG.md](./docs/releases/CHANGELOG.md) - 详细版本历史
*   🚀 **快速开始**: [QUICKSTART.md](./QUICKSTART.md) - 5分钟快速部署指南

---

**💡 立即体验：[访问原项目](https://github.com/hsliuping/TradingAgents-CN)**