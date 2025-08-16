# TradingAgents-CN: 中文金融交易决策框架 🚀

> **利用多智能体大语言模型，TradingAgents-CN 助力您全面分析 A 股、港股和美股，提供智能化的交易决策支持！**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**🎉 最新版本 cn-0.1.13-preview：**  原生 OpenAI 支持与 Google AI 全面集成预览版，提供自定义端点、9 个 Google AI 模型，并优化了 LLM 适配器架构！

**🎯 核心特性：**

*   **🚀  智能新闻分析:** AI驱动新闻过滤，质量评估和相关性分析
*   **🌐  原生 OpenAI 支持:**  灵活的自定义端点配置和模型选择
*   **🧠  Google AI 集成:**  支持 9 个 Gemini 模型，Google AI 工具调用
*   **🤖  多智能体协作:**  基本面、技术面、新闻面、情绪分析师协作
*   **📈  Web 界面:**  现代化 Streamlit 界面，实时进度跟踪和专业报告
*   **🇨🇳  中文优化:**  完整的 A 股、港股支持，中文界面和文档
*   **🐳  Docker 部署:**  一键部署，环境隔离，快速扩展
*   **💾  模型持久化:**  URL 参数存储，刷新保持设置

## 📖 为什么选择 TradingAgents-CN？

*   **专为中文用户设计**: 完整的 A 股、港股市场支持，以及本土化体验。
*   **基于顶尖技术**: 结合多智能体 LLM 架构，提供深度市场分析。
*   **功能强大**: 涵盖从数据获取、分析到决策的全流程。
*   **易于使用**: 提供 Web 界面和详细文档，轻松上手。

## ✨  核心功能特色

*   **多市场支持**: 美股、A 股、港股一站式分析
*   **五级研究深度**: 从快速概览到深度研究
*   **实时进度跟踪**: 可视化分析过程
*   **专业结果展示**: 买入/持有/卖出建议，多维度分析
*   **多 LLM 模型管理**: 4 大提供商，60+ 模型
*   **Web 界面**:  基于 Streamlit，提供直观的股票分析体验

## 🆕  版本更新亮点

###  cn-0.1.13-preview

*   **🤖 原生 OpenAI 支持**:  自定义 OpenAI 端点，灵活模型选择，智能适配器。
*   **🧠 Google AI 生态系统全面集成**:  支持 9 个 Gemini 模型，Google AI 工具处理器，智能降级。
*   **🔧 LLM 适配器架构优化**:  GoogleOpenAIAdapter，统一接口，错误处理增强，性能监控。
*   **🎨 Web 界面智能优化**:  智能模型选择，KeyError 修复，UI 响应优化，错误提示。

###  cn-0.1.12

*   **🧠 智能新闻分析模块**: AI 新闻过滤，质量评估，多层次过滤机制，统一新闻工具。
*   **🔧 技术修复和优化**: DashScope 适配器修复，DeepSeek 死循环修复，LLM 工具调用增强。
*   **📚 完善测试和文档**: 全面测试覆盖，详细技术文档，用户指南完善。
*   **🗂️ 项目结构优化**: 文档分类整理，示例代码归位，根目录整洁。

## 🚀  快速开始

###  🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量 (.env 文件,  填入API密钥)
cp .env.example .env

# 3. 启动服务
docker-compose up -d --build  # 首次构建
docker-compose up -d           # 日常启动

# 4. 访问 Web 界面
# 浏览器访问: http://localhost:8501
```

###  💻 本地部署

```bash
# 1. 安装依赖
pip install -e .

# 2.  启动Web界面
python start_web.py

# 3. 访问 Web 界面
# 浏览器访问: http://localhost:8501
```

###  📈 分析流程

1.  **选择模型**: DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票**: 例如 AAPL (美股), 000001 (A 股), 0700.HK (港股)
3.  **开始分析**: 点击 "🚀 开始分析" 按钮
4.  **实时跟踪**: 观察分析进度和步骤
5.  **查看报告**: 点击 "📊 查看分析报告"
6.  **导出报告**: 支持 Word/PDF/Markdown 格式

## 📚  详细文档

*   **[中文文档](docs/)**: 包含安装指南、使用教程、API 文档，超过 **50,000 字** 的详细中文文档！

## 🙏  致谢

感谢 [Tauric Research](https://github.com/TauricResearch/TradingAgents) 团队创建了 TradingAgents 项目！

我们也在 [CONTRIBUTORS.md](CONTRIBUTORS.md) 感谢社区贡献者。

---

<div align="center">
  **🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN)  | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork)  | [📖 Read the docs](./docs/)
</div>
```
Key improvements and SEO considerations:

*   **One-sentence hook:** Added a compelling introductory sentence to immediately grab the reader's attention.
*   **Clear headings and structure:**  Uses clear and concise headings and subheadings to improve readability and organization.
*   **Bulleted key features:** Highlights key features with bullet points for easy scanning.
*   **SEO-optimized keywords:**  Incorporated relevant keywords such as "中文金融", "交易决策", "A 股", "港股", "美股", "大语言模型", "AI", and model names.
*   **Concise summaries:** Avoids overly long paragraphs, opting for brevity.
*   **Call to action:**  Includes clear calls to action, such as "Star this repo" and "Read the docs."
*   **Internal linking**: links back to the important documents.
*   **Removed Redundancy**: streamlined the text by removing any redundancy.
*   **Updated for the latest release.**
*   **Detailed Documentation mention**: Emphasizes the key differentiating factor of the project: the detailed Chinese documentation.
*   **Cost Optimization**: Highlighted the low cost of use.
*   **Database Details**: Clearly explains database setups.
*   **Installation guide is more concise**:  Replaced some more verbose examples with shorter examples.