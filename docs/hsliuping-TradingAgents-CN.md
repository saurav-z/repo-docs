# TradingAgents-CN: 中文金融交易决策框架 (增强版)

> 🚀 开启您的金融交易之旅！ TradingAgents-CN 是一款基于多智能体大语言模型的中文金融交易决策框架，专为中国市场量身打造，提供 Web 界面、Docker 容器化部署、专业报告导出、国产大模型集成等核心功能，帮助您在金融市场中做出更明智的决策。 访问 [hsliuping/TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) 获取更多信息！

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.8-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## Key Features

*   **🇨🇳 中国市场支持**: 完整的A股、港股、新三板等市场数据和交易支持
*   **🤖 多智能体架构**: 模拟专业交易团队，协同分析市场
*   **🧠 大语言模型集成**: 支持阿里百炼、DeepSeek、Google AI、OpenAI 等模型
*   **🌐 Web 界面**: 现代化 Streamlit Web 界面，实时交互和可视化
*   **🐳 Docker 容器化**: 快速部署，环境隔离，易于扩展
*   **📄 专业报告导出**: 支持 Markdown、Word、PDF 多种格式
*   **📊 实时监控**: Token 使用统计、缓存状态监控
*   **🔑 安全**: API 密钥加密，数据安全保护
*   **📚 完整文档**: 50,000+ 字中文文档，从入门到精通
*   **⚙️ 智能配置**: 自动检测、智能降级、零配置启动

## 项目概述

TradingAgents-CN 是一个基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 开发的中文增强版金融交易框架。 旨在为中国用户提供更便捷、更强大的 AI 驱动的金融交易决策支持。

## 主要优势

*   **开箱即用**: 完整的 Web 界面，无需命令行操作
*   **中国优化**: A股数据 + 国产 LLM + 中文界面
*   **智能配置**: 自动检测、智能降级，零配置启动
*   **实时监控**: Token 使用统计，缓存状态，系统监控
*   **稳定可靠**: 多层数据源，错误恢复，生产就绪
*   **容器化**: Docker 部署，环境隔离，快速扩展
*   **专业报告**: 多格式导出，自动生成

## 核心特性

### 多智能体协作架构

*   基本面分析师
*   技术面分析师
*   新闻分析师
*   情绪分析师
*   看涨研究员
*   看跌研究员
*   交易决策员
*   风险管理员
*   研究主管

### 多 LLM 模型支持

*   阿里百炼 (qwen-turbo, qwen-plus, qwen-max)
*   DeepSeek (deepseek-chat)
*   Google AI (gemini-2.0-flash, gemini-1.5-pro)
*   OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
*   Anthropic (Claude-3-Opus, Claude-3-Sonnet)
*   智能混合 (Google AI + 阿里百炼)

### 全面数据集成

*   A股实时数据 (通达信API, AkShare)
*   美股数据 (FinnHub, Yahoo Finance)
*   新闻数据 (Google News)
*   社交数据 (Reddit, Twitter API)
*   数据库支持 (MongoDB, Redis)
*   智能降级 (MongoDB → 通达信API → 本地缓存)
*   统一配置 (.env 文件)

### 高性能特性

*   并行处理
*   智能缓存
*   实时分析
*   灵活配置
*   数据目录配置
*   数据库加速
*   高可用架构

### Web 管理界面

*   直观操作
*   实时进度
*   智能配置
*   结果展示
*   中文界面
*   配置管理
*   Token 统计
*   缓存管理

## 与原版的主要区别

*   🇨🇳 A股数据和市场支持
*   中文文档体系
*   🐳 Docker 容器化部署
*   多 LLM 模型集成 (DeepSeek, 阿里百炼, Google AI)
*   📄 专业报告导出
*   🌐 现代化 Web 界面
*   🚀 优化后的配置管理

## 快速开始

1.  **克隆项目:**
    ```bash
    git clone https://github.com/hsliuping/TradingAgents-CN.git
    cd TradingAgents-CN
    ```

2.  **配置 .env 文件:**
    ```bash
    cp .env.example .env
    # 编辑 .env 文件，填入API密钥
    ```

3.  **选择部署方式**:

    *   🐳 **Docker (推荐):**
        ```bash
        docker-compose up -d --build
        # Web界面: http://localhost:8501
        # 数据库管理: http://localhost:8081
        # 缓存管理: http://localhost:8082
        ```

    *   💻 **本地:**
        ```bash
        python -m venv env
        # Windows: env\Scripts\activate
        # Linux/macOS: source env/bin/activate
        pip install -r requirements.txt
        streamlit run web/app.py
        # 浏览器访问 http://localhost:8501
        ```

    更多详细安装和使用指南，请参考 [文档目录](docs/)

## 贡献指南

我们欢迎您的贡献！ 请参考 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多信息。

## 许可证

本项目基于 Apache 2.0 许可证开源。 详见 [LICENSE](LICENSE)。

## 联系方式

*   GitHub Issues: [hsliuping/TradingAgents-CN/issues](https://github.com/hsliuping/TradingAgents-CN/issues)
*   邮箱: hsliup@163.com

---

<div align="center">

**🌟 感谢您的支持！ 如果项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [📖 Read the docs](./docs/)

</div>