# TradingAgents-CN: 中文金融交易决策框架 - AI 赋能您的投资决策 🚀

**通过多智能体大语言模型，革新您的中文金融市场分析与交易策略！** 基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)，TradingAgents-CN 专为中国市场优化，提供全面的 A 股/港股/美股分析能力，助您做出更明智的投资决策。

## ✨ 核心特性

*   🤖 **多智能体架构:** 模拟专业分析师团队，提供全方位的市场视角。
*   🌐 **多 LLM 模型支持:**  支持阿里云百炼、DeepSeek、Google AI、OpenRouter 等多种大语言模型。
*   📈 **A 股/港股/美股 全面支持:**  涵盖中国及全球主要股票市场。
*   📰 **智能新闻分析:**  AI 驱动的新闻过滤和质量评估，捕捉市场动态。
*   💻 **Web 界面:** 简洁直观的界面，支持多市场分析，轻松导出专业报告。
*   🐳 **Docker 部署:** 一键部署，环境隔离，快速上手。

## 🚀 最新动态:  v0.1.13 - OpenAI & Google AI 集成预览版  🔥

*   🤖 **原生 OpenAI 支持:**  支持自定义 OpenAI 端点，灵活选择模型。
*   🧠 **Google AI 生态系统集成:**  全面支持 Gemini 2.5 系列模型。
*   🔧 **LLM 适配器架构优化:**  统一接口，增强错误处理和性能监控。
*   🎨 **Web 界面智能优化:**  智能模型选择，更好的用户体验。

## 📖 快速上手

1.  **🐳  Docker 部署 (推荐)**
    ```bash
    git clone https://github.com/hsliuping/TradingAgents-CN.git
    cd TradingAgents-CN
    cp .env.example .env  #  编辑 .env 文件配置 API 密钥
    docker-compose up -d --build  # 首次构建 或代码有更改
    docker-compose up -d  #  日常启动 (镜像已存在)
    # Web 界面访问: http://localhost:8501
    ```
2.  **💻  本地部署**
    ```bash
    pip install -e .  # 升级 pip 并安装依赖
    python start_web.py  #  启动Web界面
    #  Web 界面访问: http://localhost:8501
    ```

## 🎯 核心功能特色

*   **智能分析配置:** 多市场支持，5级研究深度，灵活时间设置。
*   **实时进度跟踪:** 可视化分析过程，智能时间预估。
*   **专业结果展示:** 投资建议、多维分析、量化指标、专业报告导出。
*   **多 LLM 模型管理:** 支持多种 LLM 提供商，快速切换。

## 📚 详细文档 (必读!)

[访问完整文档](./docs/)  了解框架的架构、智能体、数据处理、配置、部署和高级应用，从入门到专家，应有尽有。

## 🤝 贡献指南

我们欢迎您的贡献!  请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 开源.

---

<div align="center">

**🌟  点亮星星支持我们！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN)  |  [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork)  |  [📖 阅读文档](./docs/)

</div>