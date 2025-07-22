# 🚀 TradingAgents-CN: 中文金融交易决策框架

> 💡 **打造您的 AI 投资助手！** 基于 [Tauric Research](https://github.com/TauricResearch) 的 TradingAgents，专为中国市场优化，提供全面的 A 股、港股和美股分析能力，并支持国产大模型，让您轻松驾驭智能投资决策。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.10-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## ✨ 主要特性

*   **🇨🇳 中文本地化**: 针对中文用户优化，提供中文界面和A股、港股、美股市场支持。
*   **🤖 多智能体协作**:  市场分析师、基本面分析师、新闻分析师、情绪分析师、交易员等专业分工，进行深度分析和智能决策。
*   **🚀 实时进度显示**:  v0.1.10 引入异步进度跟踪，实时展示分析步骤和时间，告别黑盒等待。
*   **💾 智能会话管理**:  页面刷新不丢失分析结果，支持状态持久化和自动降级。
*   **🐳 Docker 部署**:  一键部署，环境隔离，快速扩展。
*   **📄 专业报告导出**:  一键导出 Markdown、Word、PDF 格式专业分析报告，生成投资建议。
*   **🧠 LLM 支持**:  支持 DeepSeek V3、阿里百炼、Google AI、OpenAI 等多种大语言模型。
*   **📊 数据源集成**: 支持 Tushare、AkShare、Yahoo Finance 等多个数据源，提供实时行情和财务数据。

## 🆕 v0.1.10 更新亮点

*   **🚀 实时进度显示**: 全新的 AsyncProgressTracker，智能时间计算，支持多种显示模式 (Streamlit, 静态, 统一)
*   **📊 智能会话管理**: 状态持久化，页面刷新恢复分析状态，Redis 不可用时自动切换到文件存储。
*   **🎨 用户体验优化**: 简化界面，响应式设计，增强错误处理。

## 💡 核心功能

*   **分析师团队**: 市场分析、基本面分析、新闻分析、情绪分析。
*   **研究团队**: 看涨研究员、看跌研究员。
*   **决策团队**: 交易决策员。
*   **风险管理**: 多层次风险评估和管理机制。
*   **Web 界面**: 现代化响应式界面，实时交互和数据可视化。
*   **配置管理**: Web端 API 密钥管理、模型选择、参数配置。
*   **命令行界面 (CLI)**: 界面与日志分离，智能进度显示，时间预估功能，Rich 彩色输出。

## 快速上手

### 🐳 Docker 部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置 .env 文件 (API 密钥)
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
6.  **导出报告**: 支持 Word/PDF/Markdown 格式

## 📚 深入了解

*   **📖 完整文档**:  [docs/](./docs/)  -  安装指南、使用教程、API 文档，提供超过 **50,000 字** 的详细中文文档。
*   **🚨 故障排除**:  [troubleshooting/](./docs/troubleshooting/)  -  常见问题解决方案
*   **🔄 更新日志**:  [CHANGELOG.md](./docs/releases/CHANGELOG.md)  -  详细版本历史

## 许可证

本项目基于 [Apache 2.0](LICENSE) 许可证开源。

---

<div align="center">
    **🔥 立即体验，开启您的 AI 投资之旅!  给个 Star 支持我们！**
    <br>
    [⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [📖 Read the docs](./docs/)
</div>
```
Key improvements and SEO optimizations:

*   **One-sentence hook:** The opening sentence is engaging and summarizes the project's core value.
*   **Clear Headings:**  Uses clear and concise headings for better readability and SEO.
*   **Bulleted Key Features:** Uses bullet points for easy scanning of key features, improving readability.
*   **Keyword Optimization:** Includes relevant keywords such as "中文", "金融", "交易", "A股", "港股", "AI", "决策", "大语言模型" etc.,  strategically placed in headings, and feature descriptions to improve search engine visibility.
*   **Concise and Direct Language:** The language is clear, concise, and directly conveys the project's purpose.
*   **Emphasis on Value Proposition:** Highlights the benefits for the user, such as "轻松驾驭智能投资决策."
*   **Call to Action:** Includes a clear call to action ("给个 Star 支持我们!") at the end with links to the repo and documentation.
*   **Internal Linking:**  Links to documentation and other relevant sections within the README.
*   **Structure and Organization:**  The README is well-structured, making it easy for users to understand the project at a glance.
*   **SEO-Friendly Headings:** Using `H1`, `H2` etc. headings in the markdown.
*   **Direct Links Back to Original:** The "Based on" section now directly links back to the original repo.
*   **Focus on User Benefits:** The features are described emphasizing the benefits to the user rather than just technical details.
*   **Improved descriptions:** Shortened descriptions and made them more descriptive using action verbs.
*   **Removed redundant info:** Only included key details in `v0.1.10`
*   **Added a more concise key features:** Included only core features to keep it from being too lengthy.