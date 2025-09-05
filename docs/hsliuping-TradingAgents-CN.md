# TradingAgents-CN: 中文金融交易决策框架 (基于 TradingAgents) 🚀

> **利用多智能体大语言模型，全面分析A股、港股和美股，提供专业的投资建议和报告。**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

基于 [Tauric Research](https://github.com/TauricResearch) 团队的 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 项目，TradingAgents-CN 为中文用户量身定制，提供**强大的中文金融交易决策框架**，支持 A股、港股和美股的全面分析。

## 🚀 **核心特性**

*   ✅ **中文支持**: 全面支持中文界面和分析，专为中国市场优化。
*   ✅ **多市场分析**: A股、港股、美股全覆盖，一站式分析体验。
*   ✅ **多智能体架构**: 模拟专业分析师团队，深度分析股票。
*   ✅ **专业报告**: 生成详细的投资报告，支持多种格式导出。
*   ✅ **LLM 支持**:  支持阿里百炼、DeepSeek、Google AI、OpenRouter (60+ 模型) 和原生 OpenAI 端点。
*   ✅ **智能新闻分析**:  AI驱动的新闻过滤和相关性分析（v0.1.12 新增）。
*   ✅ **Docker 部署**: 一键部署，方便快捷。

## ✨ **最新版本：cn-0.1.13-preview**

*   🤖 **原生 OpenAI 支持**:  全面集成 OpenAI，支持自定义端点和灵活模型选择。
*   🧠 **Google AI 集成**:  支持三大 Google AI 包和最新 Gemini 模型。
*   🔧 **LLM 适配器优化**: 提升 LLM 调用的兼容性和性能。

## 🛠️ **快速开始**

### 🐳 **Docker 部署 (推荐)**

```bash
# 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 配置API密钥 (编辑 .env 文件)
cp .env.example .env
# ... 填写API密钥

# 启动服务 (构建或启动)
docker-compose up -d --build # 构建 (首次或代码变更)
docker-compose up -d           # 启动 (镜像已存在)

# 访问 Web 界面
# 访问Web界面: http://localhost:8501
```

### 💻 **本地部署**

```bash
# 1. 升级 pip
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -e .

# 3. 启动应用
python start_web.py

# 4. 访问 Web 界面
# 访问: http://localhost:8501
```

### 🚀 **分析股票**

1.  **选择模型**： 在 Web 界面或命令行中选择 LLM。
2.  **输入股票代码**： (例如: AAPL, 000001, 0700.HK)。
3.  **选择深度**： 选择分析深度 (从快速到全面)。
4.  **开始分析**： 点击按钮开始分析。
5.  **查看报告**： 实时跟踪进度并查看生成的报告。

## 📚 **详细文档**

*   [**完整中文文档**](docs/)： 包含安装、使用、架构、常见问题解答等全面信息。

## 🤝 **贡献**

欢迎贡献代码、文档、反馈和建议。 更多信息，请参阅 [贡献指南](CONTRIBUTING.md)。

## 📄 **许可证**

本项目基于 [Apache 2.0 许可证](LICENSE) 开源。

## 🔗 **项目来源**

*   [**Tauric Research/TradingAgents**](https://github.com/TauricResearch/TradingAgents) (原项目)

---