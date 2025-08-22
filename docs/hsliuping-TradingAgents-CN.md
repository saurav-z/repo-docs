# TradingAgents-CN: 中文金融交易决策框架 (基于多智能体大语言模型)

**🚀 提升您的交易策略，利用 AI 驱动的中文金融分析，提供完整的 A 股、港股、美股分析能力！**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**基于 [Tauric Research](https://github.com/TauricResearch) 的 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 项目，TradingAgents-CN 专为中文用户优化，利用多智能体架构和大型语言模型，提供全面的股票分析和交易决策支持。**

## ✨ **核心特性**

*   🤖 **多智能体协作**: 基本面、技术面、新闻面、社交媒体全方位分析。
*   🇨🇳 **中文优化**: 专为 A 股/港股市场量身定制。
*   🧠 **AI 驱动分析**: 深入的 AI 驱动的智能新闻过滤和质量评估，支持最新 Gemini 系列模型。
*   🌐 **多 LLM 支持**: 阿里百炼、DeepSeek、Google AI、OpenRouter (包含 OpenAI, Anthropic 等)。
*   🚀 **实时进度跟踪**: 告别黑盒等待，可视化分析过程。
*   📊 **专业报告导出**: 一键生成 Markdown/Word/PDF 格式的投资报告。
*   🐳 **Docker 部署**: 快速、简便的部署方式。

## 🌟 **主要更新 - cn-0.1.13-preview**

*   🤖 **原生 OpenAI 支持**: 灵活配置和使用任何 OpenAI 兼容的 API 端点。
*   🧠 **Google AI 全面集成**: 包含最新 Gemini 2.5 系列模型，提供更强大的分析能力。
*   🔧 **LLM 适配器架构优化**: 统一的 LLM 调用接口，更好的错误处理和性能监控。
*   🎨 **Web 界面智能优化**:  更智能的模型选择，更流畅的用户体验。
*   ✨ **智能新闻分析**:  AI 驱动的新闻过滤，质量评估，相关性分析

## 🚀 **快速入门**

1.  **部署**:  使用 Docker (推荐) 或本地部署 (见下方)。
2.  **访问**:  Web 界面：`http://localhost:8501`。
3.  **输入**:  输入股票代码 (如 `AAPL`, `000001`, `0700.HK`)。
4.  **分析**:  选择分析深度，点击 "开始分析"。
5.  **查看**:  实时跟踪进度，查看分析报告。
6.  **导出**:  导出专业报告。

## 🐳 **Docker 部署** (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
# 首次启动或代码变更时（需要构建镜像）
docker-compose up -d --build

# 日常启动（镜像已存在，无代码变更）
docker-compose up -d

# 智能启动（自动判断是否需要构建）
# Windows环境
powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1

# Linux/Mac环境
chmod +x scripts/smart_start.sh && ./scripts/smart_start.sh

# 4. 访问应用
# Web界面: http://localhost:8501
```

## 💻 **本地部署**

```bash
# 1.  升级 pip
python -m pip install --upgrade pip

# 2.  安装依赖
pip install -e .

# 3.  启动应用
python start_web.py

# 4.  访问 http://localhost:8501
```

## 📚 **文档**

*   **[完整文档](./docs/)**：包含安装、使用、API 和技术细节，超过 50,000 字，为中文用户量身定制。

## 🤝 **贡献**

我们欢迎您的贡献！  查看 [CONTRIBUTORS.md](CONTRIBUTORS.md) 了解如何参与。

## 📄 **许可证**

本项目基于 Apache 2.0 许可证开源。

---

**[访问原项目](https://github.com/TauricResearch/TradingAgents)**  |  [提交问题和建议](https://github.com/hsliuping/TradingAgents-CN/issues) |  [文档](./docs/)