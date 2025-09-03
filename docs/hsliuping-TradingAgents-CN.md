# TradingAgents-CN: 中文金融交易决策框架 (增强版) 🚀

> 💡 **解锁AI驱动的金融洞察！** TradingAgents-CN是基于多智能体大语言模型的中文金融交易决策框架，专为中国市场优化，提供全面的A股/港股/美股分析能力。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**[访问原始仓库](https://github.com/hsliuping/TradingAgents-CN)**

## 🚀 核心功能

*   **🇨🇳 中文优化:** 专为中文用户设计，支持A股/港股/美股。
*   **🤖 多LLM支持:** 集成Alibaba百炼、DeepSeek、Google AI、OpenRouter等，模型选择丰富。
*   **🧠 智能新闻分析:** AI驱动的新闻过滤，质量评估与相关性分析。
*   **📈 实时分析:** 可视化分析流程，实时进度追踪。
*   **📊 专业报告:** 生成Markdown、Word、PDF格式的专业投资报告。
*   **🐳 容器化部署:** 一键Docker部署，快速启动。
*   **🌐 Web界面:** 直观友好的Web界面，轻松进行股票分析。

## ✨ 最新版本：cn-0.1.13-preview

**重点更新：**

*   **🤖 原生OpenAI支持:** 自定义端点，灵活模型选择，增强适配器。
*   **🧠 Google AI生态系统全面集成:** 支持三大Google AI包，9个验证模型。
*   **🔧 LLM适配器架构优化:** 统一接口，错误处理增强，性能监控。
*   **🎨 Web界面智能优化:** 智能模型选择，UI响应优化。

## 🎯 核心特性

*   **多智能体协作架构:** 四大分析师团队(技术面、基本面、新闻面、社交媒体)，结构化辩论，智能决策与风险管理。
*   **Web界面:** 现代化Streamlit界面，提供直观的股票分析体验，支持多市场、多种分析深度。
*   **全面数据源支持:** 涵盖A股、港股、美股，并集成新闻源。

## 快速开始

### 🐳 Docker部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
docker-compose up -d --build  # 首次启动或代码变更时（需要构建镜像）
docker-compose up -d          # 日常启动（镜像已存在，无代码变更）

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

### 🚀 使用流程

1.  **选择模型 & 输入股票代码** (美股: AAPL, A股: 000001, 港股: 0700.HK)。
2.  **选择研究深度** (1-5级)。
3.  **开始分析**，实时跟踪进度。
4.  **查看报告** 并导出。

## 📚 详细文档

*   **[完整文档目录](docs/)** - 包含项目概览，架构设计，智能体详解，数据处理，配置和优化，高级应用及常见问题解答。
*   **[快速开始](QUICKSTART.md)** - 5分钟快速部署指南。

## 🤝 贡献

欢迎贡献! 查看 [CONTRIBUTORS.md](CONTRIBUTORS.md) 了解更多。

## 📄 许可证

基于 Apache 2.0 许可证开源.  详见 [LICENSE](LICENSE).

**更多信息, 请访问我们的文档，并参考** [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).