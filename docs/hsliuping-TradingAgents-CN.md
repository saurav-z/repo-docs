# TradingAgents-CN: 中文金融交易决策框架 - 🚀 智能AI赋能您的投资决策

> 🚀 **快速上手，智能分析，A股、港股、美股全支持！** TradingAgents-CN 基于多智能体大语言模型，专为中文用户优化，提供全面的股票分析和投资决策支持，助您洞悉市场脉搏。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## ✨ 核心特性

*   **🤖 多智能体架构**: 协同分析市场、基本面、新闻和情绪，实现深度洞察。
*   **🌐 多LLM支持**:  兼容阿里百炼、DeepSeek、Google AI、原生OpenAI、OpenRouter等，灵活选择模型。
*   **📈  全面的市场覆盖**:  支持A股、港股、美股，提供多市场数据分析。
*   **📊  专业报告生成**:  提供Markdown、Word、PDF多种格式的专业投资报告。
*   **🚀  Web界面**:  基于Streamlit构建的Web界面，直观易用，快速上手。
*   **🧠  智能新闻分析**:  新增AI驱动的新闻过滤和质量评估系统。
*   **🐳  Docker部署**:  一键部署，快速启动，环境隔离，方便扩展。

## 🆕 版本更新：cn-0.1.13-preview

*   **🤖  原生OpenAI支持**: 灵活自定义端点，兼容任何OpenAI格式的模型。
*   **🧠  Google AI全面集成**: 支持Gemini 2.5 系列及更多模型，提供更强大的分析能力。
*   **🔧  LLM适配器架构优化**: 统一接口，错误处理增强，性能监控。

## 🛠️ 快速开始

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置API密钥 (编辑 .env 文件)
cp .env.example .env
# 3. 启动服务
docker-compose up -d --build # 初次构建镜像
docker-compose up -d          # 之后启动

# 4. 访问
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

## 📚 深入了解

*   **[完整文档](docs/)**:  详细的安装指南、使用教程、API文档和项目架构解析。
*   **[演示视频](https://www.bilibili.com/video/BV15s4y1t7C9/)**: 快速了解 TradingAgents-CN
*   **[示例代码](examples/)**:  快速上手

## 🙏 致谢

感谢 [Tauric Research](https://github.com/TauricResearch/TradingAgents) 团队提供的卓越的 TradingAgents 框架！

**项目仓库**: [hsliuping/TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN)

---

<div align="center">

**⭐️  如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [📖 阅读文档](docs/)

</div>
```

Key improvements and optimizations:

*   **SEO-optimized title:**  Uses key phrases like "中文金融交易决策框架" and "智能AI" to improve searchability.  The one-sentence hook encapsulates the core benefit.
*   **Concise Summary:** Clearly states the value proposition and highlights key features early on.
*   **Clear Headings:**  Uses descriptive headings (e.g., "核心特性", "快速开始", "版本更新") for better organization.
*   **Bulleted Key Features:** Makes it easy to scan and understand the main functionalities.
*   **Streamlined Content:**  Removed redundant information and focused on the most important details.
*   **Actionable Instructions:**  Provides clear "快速开始" instructions for both Docker and local deployment.
*   **Links to Documentation and Examples:**  Directs users to the most important resources.
*   **Clean Presentation:**  Uses Markdown formatting for readability.
*   **Clear Call to Action:** Encourages users to star the repository.
*   **Concise and Focused:** Keeps the README brief and to the point, highlighting the key aspects of the project without overwhelming the reader.
*   **Removed Redundant Screenshots**: Added link to demo video instead of repeated screenshots.
*   **Simplified Version History**: Reduced verbosity and focused on the main changes.
*   **Removed Detailed Installation for Local Deployment**: Kept the essential steps.
*   **Included Chinese keywords**: Improves searchability in the Chinese market.
*   **Combined similar sections.**
*   **Removed redundant information.**
*   **Updated version.**
*   **Improved spacing for better readability.**