# TradingAgents-CN: 中文金融交易决策框架 - AI 赋能您的投资策略

**打造基于多智能体大语言模型的中文金融交易决策框架，助力您全面分析 A 股、港股和美股！**

[**查看原始仓库 (TauricResearch/TradingAgents)**](https://github.com/TauricResearch/TradingAgents)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)

> **🚀  cn-0.1.13-preview (最新版):**  原生 OpenAI 支持、Google AI 深度集成！ 体验自定义 OpenAI 端点、9 个 Google AI 模型以及更优 LLM 适配器！

## 🔑 核心特性

*   🤖 **多智能体协作**:  技术面、基本面、新闻面、情绪面等多维度分析师团队。
*   🧠 **智能新闻分析**:  AI 驱动的新闻过滤、质量评估与相关性分析 (v0.1.12)。
*   🌐 **多 LLM 模型支持**: 阿里百炼、DeepSeek、Google AI、OpenRouter，超过 60+ 模型可选。
*   🇨🇳 **中文 & 多市场支持**:  全面支持A股、港股、美股，提供本地化体验。
*   📊 **Web 界面**:  直观的股票分析体验，实时进度跟踪，专业报告导出。
*   🐳 **Docker 部署**: 一键部署，环境隔离，方便快速启动和扩展。

## 🌟 主要更新 - cn-0.1.13-preview

### 🤖 原生 OpenAI 支持

*   **自定义 OpenAI 端点**: 支持任何 OpenAI 兼容 API 端点。
*   **灵活模型选择**: 使用任意 OpenAI 格式的模型。
*   **智能适配器**: 新增 OpenAI 适配器，提升兼容性和性能。
*   **统一配置管理**: 统一端点和模型配置。

### 🧠 Google AI 生态系统全面集成

*   **三大 Google AI 包支持**:  langchain-google-genai, google-generativeai, google-genai。
*   **9 个验证模型**: Gemini 2.5 系列等最新模型。
*   **Google 工具处理器**: 专用 Google AI 工具调用处理器。
*   **智能降级机制**: 失败时自动降级到基础功能。

### 🔧 LLM 适配器架构优化

*   **GoogleOpenAIAdapter**: 新增 Google AI OpenAI 兼容适配器。
*   **统一接口**: 所有 LLM 提供商统一调用接口。
*   **错误处理增强**:  改进的异常处理和自动重试机制。
*   **性能监控**: 添加 LLM 调用性能监控。

### 🎨 Web 界面智能优化

*   **智能模型选择**: 根据可用性自动选择最佳模型。
*   **KeyError 修复**: 解决模型选择中的 KeyError 问题。
*   **UI 响应优化**:  提升模型切换的响应速度。
*   **错误提示**:  更友好的错误提示和解决建议。

## 🎯 核心功能详解

### 多智能体协作架构

*   **专业分工**:  四大分析师 (技术、基本面、新闻、情绪)。
*   **结构化辩论**:  多角度深度分析，看涨/看跌研究员。
*   **智能决策**:  交易员基于全面分析给出投资建议。
*   **风险管理**:  多层次风险评估和管理机制。

### Web 界面展示

<details>
<summary><strong>Web 界面截图</strong></summary>

*  [主界面](images/README/1755003162925.png) - 分析配置，支持多市场股票分析
*  [实时分析进度](images/README/1755002731483.png) - 可视化分析过程
*  [分析结果展示](images/README/1755002901204.png) - 专业投资报告，多维度分析结果，一键导出
</details>

### 核心功能特色

*   **📋 智能分析配置**: 多市场、多深度、灵活时间设置。
*   **🚀 实时进度跟踪**: 可视化进度、智能步骤识别、时间预估。
*   **📈 专业结果展示**: 投资决策、多维分析、量化指标、专业报告。
*   **🤖 多 LLM 模型管理**:  4 大提供商，60+ 模型可选，快速切换，配置持久化。

### Web 界面操作指南

1.  **启动应用**: `python start_web.py` 或 `docker-compose up -d`
2.  **访问界面**: 浏览器打开 `http://localhost:8501`
3.  **配置模型**: 选择 LLM 提供商和模型。
4.  **输入股票**: 输入股票代码。
5.  **选择深度**: 选择研究深度 (1-5级)。
6.  **开始分析**: 点击 "🚀 开始分析"。
7.  **查看结果**:  实时跟踪进度，查看分析报告。
8.  **导出报告**:  一键导出专业格式报告。

### 📈 支持的股票代码格式

*   **🇺🇸 美股**: `AAPL`, `TSLA`, `MSFT`...
*   **🇨🇳 A股**: `000001`, `600519`...
*   **🇭🇰 港股**: `0700.HK`, `9988.HK`...

### 🎯 研究深度说明

*   **1级**: 快速概览，基础指标。
*   **2级**: 标准分析，技术+基本面。
*   **3级**: 深度分析，加入新闻情绪 (推荐)。
*   **4级**: 全面分析，多轮辩论。
*   **5级**: 最深度分析，完整研究报告。

## 🚀 快速开始

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填写 API 密钥

# 3. 启动服务
docker-compose up -d --build  # 首次启动或代码变更时
docker-compose up -d           # 日常启动

# 4. 访问应用
# Web 界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 升级 pip
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -e .

# 3. 启动应用
python start_web.py

# 4. 访问 http://localhost:8501
```

## 💡 核心优势

*   **🧠 智能新闻分析**:  AI 驱动的新闻过滤和质量评估 (v0.1.12)。
*   **🌐 多 LLM 集成**:  4 大提供商，60+ 模型，一站式 AI 体验。
*   **💾 配置持久化**: 模型选择持久化，刷新不丢失。
*   **🇨🇳 中国优化**:  A股/港股数据 + 国产 LLM + 中文界面。
*   🐳 **容器化**: Docker 一键部署，环境隔离。
*   📄 **专业报告**:  多格式导出，自动生成投资建议。

## 📚 详细文档与支持

*   **📖 完整文档**:  [docs/](./docs/) - 快速上手、架构详解、 API 文档。
*   **🚨 故障排除**: [troubleshooting/](./docs/troubleshooting/) - 常见问题解决方案。
*   **🔄 更新日志**: [CHANGELOG.md](./docs/releases/CHANGELOG.md) - 详细版本历史。
*   **🚀 快速开始**: [QUICKSTART.md](./QUICKSTART.md) - 5分钟快速部署指南。

[**获取更多信息 & 参与贡献!**](https://github.com/hsliuping/TradingAgents-CN)