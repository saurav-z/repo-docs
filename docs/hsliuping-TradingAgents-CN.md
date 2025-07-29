# TradingAgents-CN: 中文金融交易决策框架

基于多智能体大语言模型的中文金融交易决策框架，专为中文用户优化，提供全面的A股/港股/美股分析能力，助力您在金融市场中做出更明智的决策。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.12-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**[原项目地址：TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)**

## 🚀 核心特性

*   **🤖 多智能体协作**: 专业的分析师团队，协同分析市场、基本面、新闻和情绪。
*   **📰 智能新闻分析**:  AI驱动的新闻过滤、质量评估和相关性分析。
*   **🇨🇳 中文优化**:  全面支持A股、港股市场，以及中文LLM模型。
*   **🌐 多LLM支持**: 集成多家LLM提供商，包括阿里云百炼、DeepSeek，OpenRouter等,  提供60+模型选择。
*   **💾 模型选择持久化**:  通过URL参数实现模型配置持久化，方便分享。
*   **🐳 Docker 部署**: 一键部署，环境隔离，快速启动。
*   **📊 专业报告导出**: 支持Word/PDF/Markdown格式，生成专业的投资建议报告。

## ✨ 主要更新 (v0.1.12)

*   **🧠 智能新闻分析模块**: 全面升级，包括：
    *   智能新闻过滤器
    *   多层次过滤机制
    *   新闻质量评估
    *   统一新闻工具

## 🎯 主要功能

*   **🤖 多智能体协作**: 市场分析、基本面分析、新闻分析、情绪分析，看涨/看跌辩论，交易决策和风险管理。
*   **🧠 智能新闻分析**: AI驱动的新闻过滤，质量评估，相关性分析，多层次过滤机制，统一新闻工具。
*   **📊 Web界面体验**:  现代化、响应式界面，实时交互和数据可视化。
*   **🎨 CLI 用户体验**:  界面与日志分离、智能进度显示、时间预估功能、Rich彩色输出。
*   **🇨🇳 中文支持**:  A股、港股、美股数据，中文界面，中文LLM模型。
*   **🐳 Docker 部署**:  一键部署，环境隔离，快速扩展。
*   **💾 模型选择持久化**:  基于URL参数的存储，刷新配置保持。
*   **🚀 快速切换**:  一键切换不同AI模型。
*   **📄 专业报告**:  多格式导出，自动生成投资建议。
*   **🛡️ 稳定可靠**:  多层数据源，智能降级，错误恢复。

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

## 📚 文档

全面的中文文档，提供详细的安装指南、使用教程、API文档和常见问题解答：

*   **[完整文档目录](docs/)**

## 🤝 贡献

欢迎通过以下方式为项目做贡献：

*   提交Bug修复
*   添加新功能
*   完善文档
*   提供代码优化

详细的贡献流程请参考 CONTRIBUTING.md。

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。

## 🙏 致谢

感谢 [Tauric Research](https://github.com/TauricResearch) 团队创建的 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 项目!

**[查看原项目](https://github.com/TauricResearch/TradingAgents)**

---

**[在GitHub上给个Star吧!](https://github.com/hsliuping/TradingAgents-CN)**