# TradingAgents-CN: 中文金融交易决策框架，解锁 A 股投资新视野!

> 🚀 **基于多智能体大语言模型的中文金融交易决策框架，为中国用户提供全面的 A 股、港股、美股分析能力，赋能智能投资决策！**  [访问原始仓库](https://github.com/hsliuping/TradingAgents-CN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**TradingAgents-CN** 是一个基于先进多智能体架构，并针对中文环境优化的金融交易决策框架。它利用大语言模型 (LLMs) 分析股票市场，并提供投资建议。 该项目针对中国市场进行了定制，支持 A 股、港股和美股，并且特别注重本土化，例如，添加了对中文的支持和集成国产大模型，提供更出色的用户体验。

## ✨ 主要特点

*   **🇨🇳 中文本地化**: 全面支持中文，包括界面、分析结果和文档。
*   **🌐 多LLM支持**: 整合 DashScope、DeepSeek、Google AI、OpenRouter 及 OpenAI 兼容模型。
*   **📈 股票市场覆盖**: 支持 A 股、港股、美股市场的分析。
*   **🤖 多智能体架构**: 采用专业的分析师团队 (基本面、技术面、新闻面、情绪面)，进行深度分析。
*   **📊 专业报告**: 提供 Markdown/Word/PDF 格式的专业投资报告导出功能。
*   **🐳 一键部署**: 通过 Docker 容器化部署，快速搭建和扩展。
*   **🧠 智能新闻分析**: 新闻过滤、质量评估、相关性分析 (v0.1.12 新增)。
*   **🚀 实时进度跟踪**: 可视化分析过程，方便用户监控 (v0.1.10 新增)。
*   **💾 配置持久化**: 模型选择和设置将得到保存，便于分享和复用 (v0.1.11 新增)。
*   **🎉 快速切换**: 提供热门模型一键切换按钮，简化操作 (v0.1.11 新增)。

## 🌟 核心更新 (cn-0.1.13-preview)

*   **🤖 原生 OpenAI 支持**:
    *   自定义 OpenAI 端点。
    *   灵活的模型选择。
    *   新增原生 OpenAI 适配器。
    *   统一的配置管理系统。
*   **🧠 Google AI 生态系统集成**:
    *   全面支持 langchain-google-genai、google-generativeai、google-genai。
    *   支持包括 gemini-2.5-pro、gemini-2.5-flash 等 9 个验证模型。
    *   新增 Google AI 工具调用处理器。
    *   智能降级机制。
*   **🔧 LLM 适配器架构优化**:
    *   新增 Google AI 的 OpenAI 兼容适配器。
    *   统一的调用接口。
    *   增强的错误处理和自动重试。
    *   LLM 调用性能监控。
*   **🎨 Web 界面智能优化**:
    *   智能模型选择。
    *   KeyError 修复。
    *   UI 响应优化。
    *   改进的错误提示。

## 快速上手

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
docker-compose up -d --build # 首次启动或代码变更
docker-compose up -d         # 日常启动

# 4. 访问应用
# Web 界面: http://localhost:8501
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

## 深入了解

*   **📖 文档**:  [docs/](docs/)  - 包含安装指南、使用教程和 API 文档。
*   **🚨 故障排除**:  [docs/troubleshooting/](docs/troubleshooting/)  - 常见问题解决方案。
*   **🚀 快速开始**:  [QUICKSTART.md](./QUICKSTART.md)  - 5 分钟快速部署指南。
*   **📚 详细文档目录**:  请查看完整的文档体系，帮助你深入了解框架的各个方面，包括架构设计、智能体设计和数据处理流程。

## 🤝 贡献指南

我们非常欢迎社区贡献！  请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献流程。

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。

---