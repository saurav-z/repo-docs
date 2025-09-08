# TradingAgents-CN: 中文金融AI交易决策框架

🚀 **利用多智能体AI，全面分析A股/港股/美股，助您做出更明智的投资决策！**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.13--preview-orange.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 的中文增强版，提供更全面的 A 股、港股市场支持和 Native OpenAI / Google AI 集成。**

## 主要特点

*   **全面中文支持**: 专为中文用户优化，提供 A 股、港股、美股分析能力。
*   **多智能体协作**: 基本面、技术面、新闻面、情绪面分析师协同工作。
*   **Native OpenAI / Google AI 集成**:  支持自定义端点，提供多个模型选择。
*   **Web 界面**:  基于 Streamlit 构建的响应式 Web 应用，提供直观的股票分析体验。
*   **专业报告**:  支持 Markdown/Word/PDF 格式报告导出，提供投资建议。

## 核心功能

*   **多市场支持**: 美股、A股、港股一站式分析。
*   **AI 驱动的新闻分析**: 智能新闻过滤，质量评估，相关性分析。
*   **多 LLM 模型**:  支持 DashScope、DeepSeek、Google AI、OpenRouter 等多LLM 提供商，包含超过 60 个模型。
*   **实时进度跟踪**:  可视化分析过程，智能时间预估。
*   **配置持久化**:  模型选择，URL 参数存储，刷新保持。
*   **专业结果展示**:  明确的买入/持有/卖出建议，多维分析，专业报告。
*   **Docker 部署**:  一键部署，环境隔离，快速扩展。

## 最新更新

### v0.1.13 - 🚀  原生OpenAI & Google AI 集成 (预览版)

*   **原生 OpenAI 端点支持**: 自定义 OpenAI 兼容 API 端点，灵活选择 OpenAI 格式模型。
*   **Google AI 生态系统集成**:  集成 langchain-google-genai、google-generativeai、google-genai，提供 9 个验证模型 (gemini-2.5-pro, gemini-2.5-flash 等)。
*   **LLM 适配器架构优化**: Google AI 的 OpenAI 兼容适配器、统一调用接口，增强错误处理和性能监控。
*   **Web 界面智能优化**:  智能模型选择，KeyError 修复，UI 响应优化，更友好的错误提示。

### 核心特性详细介绍

*   **多智能体协作架构**
    *   专业分工：基本面、技术面、新闻面、社交媒体四大分析师。
    *   结构化辩论：看涨/看跌研究员进行深度分析。
    *   智能决策：交易员基于所有输入做出最终投资建议。
    *   风险管理：多层次风险评估和管理机制。

*   **Web 界面展示**
    *   [ 🏠 主界面 - 分析配置](images/README/1755003162925.png) - 智能配置面板，支持多市场股票分析，5 级研究深度选择。
    *   [ 📊 实时分析进度](images/README/1755002731483.png) - 实时进度跟踪，可视化分析过程，智能时间预估。
    *   [ 📈 分析结果展示](images/README/1755002901204.png) - 专业投资报告，多维度分析结果，一键导出功能。

    *   **核心功能特色**
        *   **智能分析配置**
            *   🌍 多市场支持：美股、A股、港股一站式分析。
            *   🎯 5 级研究深度：从 2 分钟快速分析到 25 分钟全面研究。
            *   🤖 智能体选择：市场技术、基本面、新闻、社交媒体分析师。
            *   📅 灵活时间设置：支持历史任意时间点分析。
        *   **实时进度跟踪**
            *   📊 可视化进度：实时显示分析进展和剩余时间。
            *   🔄 智能步骤识别：自动识别当前分析阶段。
            *   ⏱️ 准确时间预估：基于历史数据的智能时间计算。
            *   💾 状态持久化：页面刷新不丢失分析进度。
        *   **专业结果展示**
            *   🎯 投资决策：明确的买入/持有/卖出建议。
            *   📊 多维分析：技术面、基本面、新闻面综合评估。
            *   🔢 量化指标：置信度、风险评分、目标价位。
            *   📄 专业报告：支持 Markdown/Word/PDF 格式导出。
        *   **多 LLM 模型管理**
            *   🌐 4 大提供商：DashScope、DeepSeek、Google AI、OpenRouter
            *   🎯 60+ 模型选择：从经济型到旗舰级模型全覆盖
            *   💾 配置持久化：URL 参数存储，刷新保持设置
            *   ⚡ 快速切换：5 个热门模型一键选择按钮

##  快速开始

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API 密钥

# 3. 启动服务
docker-compose up -d --build  # 首次启动或代码变更
# or
docker-compose up -d  # 日常启动
# or
# Windows
powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1
# Linux/Mac
chmod +x scripts/smart_start.sh && ./scripts/smart_start.sh

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

## 🎯 功能特性

### 🚀  智能新闻分析 ✨ v0.1.12 重大升级

| 功能特性               | 状态        | 详细说明                                 |
| ---------------------- | ----------- | ---------------------------------------- |
| **🧠 智能新闻分析**    | 🆕 v0.1.12  | AI 新闻过滤，质量评估，相关性分析         |
| **🔧 新闻过滤器**      | 🆕 v0.1.12  | 多层次过滤，基础/增强/集成三级处理       |
| **📰 统一新闻工具**    | 🆕 v0.1.12  | 整合多源新闻，统一接口，智能检索         |

### 🧠 LLM 模型支持 ✨ v0.1.13 全面升级

| 模型提供商        | 支持模型                     | 特色功能                | 新增功能 |
| ----------------- | ---------------------------- | ----------------------- | -------- |
| **🇨🇳 阿里百炼** | qwen-turbo/plus/max          | 中文优化，成本效益高    | ✅ 集成  |
| **🇨🇳 DeepSeek** | deepseek-chat                | 工具调用，性价比极高    | ✅ 集成  |
| **🌍 Google AI**  | **9 个验证模型**              | 最新 Gemini 2.5 系列      | 🆕 升级  |
| ├─**最新旗舰**  | gemini-2.5-pro/flash         | 最新旗舰，超快响应      | 🆕 新增  |
| ├─**稳定推荐**  | gemini-2.0-flash             | 推荐使用，平衡性能      | 🆕 新增  |
| ├─**经典强大**  | gemini-1.5-pro/flash         | 经典稳定，高质量分析    | ✅ 集成  |
| └─**轻量快速**  | gemini-2.5-flash-lite        | 轻量级任务，快速响应    | 🆕 新增  |
| **🌐 原生 OpenAI** | **自定义端点支持**           | 任意 OpenAI 兼容端点      | 🆕 新增  |
| **🌐 OpenRouter** | **60+ 模型聚合平台**          | 一个 API 访问所有主流模型 | ✅ 集成  |
| ├─**OpenAI**    | o4-mini-high, o3-pro, GPT-4o | 最新 o 系列，推理专业版   | ✅ 集成  |
| ├─**Anthropic** | Claude 4 Opus/Sonnet/Haiku   | 顶级性能，平衡版本      | ✅ 集成  |
| ├─**Meta**      | Llama 4 Maverick/Scout       | 最新 Llama 4 系列         | ✅ 集成  |
| └─**自定义**    | 任意 OpenRouter 模型 ID         | 无限扩展，个性化选择    | ✅ 集成  |

## 📚 详细文档

*   **快速开始**:  [QUICKSTART.md](./QUICKSTART.md)
*   **文档**:  [docs/](./docs/) - 包含安装指南、使用教程、API 文档
*   **更新日志**:  [CHANGELOG.md](./docs/releases/CHANGELOG.md)
*   **Web 界面详细使用指南**: [🖥️ Web界面详细使用指南](docs/usage/web-interface-detailed-guide.md)
*   **导出功能指南**: [导出功能指南](docs/EXPORT_GUIDE.md)
*   **数据目录配置指南**: [📁 数据目录配置指南](docs/configuration/data-directory-configuration.md)
*   **数据库架构文档**: [数据库架构文档](docs/architecture/database-architecture.md)

## 🤝 贡献

我们欢迎各种形式的贡献！  查看 [CONTRIBUTORS.md](CONTRIBUTORS.md)  了解贡献者名单。

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">
   **🌟 如果这个项目对您有帮助，请给我们一个 Star！**
   [⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [📖 Read the docs](./docs/)
</div>