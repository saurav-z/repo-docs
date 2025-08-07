# TradingAgents-CN: 🇨🇳 中文金融交易决策框架

> 🚀 基于多智能体大语言模型的中文金融交易决策框架，专为中国市场优化，提供全面的 A 股/港股/美股分析能力！

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.12-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## ✨ 主要特性

*   **🧠 智能新闻分析:** AI驱动的新闻过滤、质量评估和相关性分析 (v0.1.12)
*   **📰 多层次新闻过滤:** 基础、增强、集成三级过滤 (v0.1.12)
*   **🤖 多LLM提供商集成:** 支持 4 大提供商，60+ 模型，智能分类管理 (v0.1.11)
*   **💾 模型选择持久化:** 通过 URL 参数存储，刷新后保留配置 (v0.1.11)
*   **🇨🇳 中国市场优化:** A 股、港股数据支持，国产 LLM 集成，中文界面
*   **🐳 Docker 部署:** 一键部署，环境隔离，快速扩展
*   **📄 专业报告导出:** 导出 Markdown、Word 和 PDF 格式的分析报告

## 🆕 v0.1.12 更新亮点

### 🧠 智能新闻分析 🚀

*   **智能新闻过滤器:** 基于 AI 的新闻相关性评分和质量评估
*   **多层次过滤机制:** 基础过滤、增强过滤、集成过滤三级处理
*   **新闻质量评估:** 自动识别和过滤低质量、重复、无关新闻
*   **统一新闻工具:** 整合多个新闻源，提供统一的新闻获取接口

### 🔧 技术修复和优化

*   **DashScope 适配器修复:** 解决工具调用兼容性问题
*   **DeepSeek 死循环修复:** 修复新闻分析师的无限循环问题
*   **LLM 工具调用增强:** 提升工具调用的可靠性和稳定性
*   **新闻检索器优化:** 增强新闻数据获取和处理能力

### 📚 完善测试和文档

*   **全面测试覆盖:** 新增 15+ 个测试文件，覆盖所有新功能
*   **详细技术文档:** 新增 8 个技术分析报告和修复文档
*   **用户指南完善:** 新增新闻过滤使用指南和最佳实践
*   **演示脚本:** 提供完整的新闻过滤功能演示

### 🗂️ 项目结构优化

*   **文档分类整理:** 按功能将文档分类到 docs 子目录
*   **示例代码归位:** 演示脚本统一到 examples 目录
*   **根目录整洁:** 保持根目录简洁，提升项目专业度

## 核心特性

### 🤖 多智能体协作架构

*   **专业分工:** 基本面、技术面、新闻面、社交媒体四大分析师
*   **结构化辩论:** 看涨/看跌研究员进行深度分析
*   **智能决策:** 交易员基于所有输入做出最终投资建议
*   **风险管理:** 多层次风险评估和管理机制

### 🎯 功能特性

| 功能特性               | 状态        | 详细说明                                 |
| ---------------------- | ----------- | ---------------------------------------- |
| **🧠 智能新闻分析**    | 🆕 v0.1.12  | AI 新闻过滤，质量评估，相关性分析         |
| **🔧 新闻过滤器**      | 🆕 v0.1.12  | 多层次过滤，基础/增强/集成三级处理       |
| **📰 统一新闻工具**    | 🆕 v0.1.12  | 整合多源新闻，统一接口，智能检索         |
| **🤖 多 LLM 提供商**     | 🆕 v0.1.11  | 4 大提供商，60+ 模型，智能分类管理         |
| **💾 模型选择持久化**  | 🆕 v0.1.11  | URL 参数存储，刷新保持，配置分享          |
| **🎯 快速选择按钮**    | 🆕 v0.1.11  | 一键切换热门模型，提升操作效率           |
| **📊 实时进度显示**    | ✅ v0.1.10  | 异步进度跟踪，智能步骤识别，准确时间计算 |
| **💾 智能会话管理**    | ✅ v0.1.10  | 状态持久化，自动降级，跨页面恢复         |
| **🎯 一键查看报告**    | ✅ v0.1.10  | 分析完成后一键查看，智能结果恢复         |
| **🖥️ Streamlit 界面** | ✅ 完整支持 | 现代化响应式界面，实时交互和数据可视化   |
| **⚙️ 配置管理**      | ✅ 完整支持 | Web 端 API 密钥管理，模型选择，参数配置     |

### 🧠 LLM 模型支持 ✨ **v0.1.11 全面升级**

| 模型提供商        | 支持模型                     | 特色功能                | 新增功能 |
| ----------------- | ---------------------------- | ----------------------- | -------- |
| **🇨🇳 阿里百炼** | qwen-turbo/plus/max          | 中文优化，成本效益高    | ✅ 集成  |
| **🇨🇳 DeepSeek** | deepseek-chat                | 工具调用，性价比极高    | ✅ 集成  |
| **🌍 Google AI**  | gemini-2.0-flash/1.5-pro     | 多模态支持，推理能力强  | ✅ 集成  |
| **🌐 OpenRouter** | **60+模型聚合平台**          | 一个 API 访问所有主流模型 | 🆕 新增  |
| ├─**OpenAI**    | o4-mini-high, o3-pro, GPT-4o | 最新 o 系列，推理专业版   | 🆕 新增  |
| ├─**Anthropic** | Claude 4 Opus/Sonnet/Haiku   | 顶级性能，平衡版本      | 🆕 新增  |
| ├─**Meta**      | Llama 4 Maverick/Scout       | 最新 Llama 4 系列         | 🆕 新增  |
| ├─**Google**    | Gemini 2.5 Pro/Flash         | 多模态专业版            | 🆕 新增  |
| └─**自定义**    | 任意 OpenRouter 模型 ID         | 无限扩展，个性化选择    | 🆕 新增  |

**🎯 快速选择**: 5 个热门模型快速按钮 | **💾 持久化**: URL 参数存储，刷新保持 | **🔄 智能切换**: 一键切换不同提供商

### 📊 数据源与市场

| 市场类型      | 数据源                   | 覆盖范围                     |
| ------------- | ------------------------ | ---------------------------- |
| **🇨🇳 A 股**  | Tushare, AkShare, 通达信 | 沪深两市，实时行情，财报数据 |
| **🇭🇰 港股** | AkShare, Yahoo Finance   | 港交所，实时行情，基本面     |
| **🇺🇸 美股** | FinnHub, Yahoo Finance   | NYSE, NASDAQ，实时数据       |
| **📰 新闻**   | Google News              | 实时新闻，多语言支持         |

### 🤖 智能体团队

**分析师团队**: 📈市场分析 | 💰基本面分析 | 📰新闻分析 | 💬情绪分析
**研究团队**: 🐂看涨研究员 | 🐻看跌研究员 | 🎯交易决策员
**管理层**: 🛡️风险管理员 | 👔研究主管

## 🚀 快速开始

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API 密钥

# 3. 启动服务
# 首次启动或代码变更时（需要构建镜像）
docker-compose up -d --build

# 日常启动（镜像已存在，无代码变更）
docker-compose up -d

# 智能启动（自动判断是否需要构建）
# Windows 环境
powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1

# Linux/Mac 环境
chmod +x scripts/smart_start.sh && ./scripts/smart_start.sh

# 4. 访问应用
# Web 界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 升级 pip (重要！避免安装错误)
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -e .

# 3. 启动应用
python start_web.py

# 4. 访问 http://localhost:8501
```

### 📊 开始分析

1.  **选择模型**: DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票**: `000001` (A 股) / `AAPL` (美股) / `0700.HK` (港股)
3.  **开始分析**: 点击"🚀 开始分析"按钮
4.  **实时跟踪**: 观察实时进度和分析步骤
5.  **查看报告**: 点击"📊 查看分析报告"按钮
6.  **导出报告**: 支持 Word/PDF/Markdown 格式

## 🎯 核心优势

*   🧠 **智能新闻分析:** v0.1.12 新增 AI 驱动的新闻过滤和质量评估系统
*   🔧 **多层次过滤:** 基础、增强、集成三级新闻过滤机制
*   📰 **统一新闻工具:** 整合多源新闻，提供统一的智能检索接口
*   🆕 **多 LLM 集成:** v0.1.11 新增 4 大提供商，60+ 模型，一站式 AI 体验
*   💾 **配置持久化:** 模型选择真正持久化，URL 参数存储，刷新保持
*   🎯 **快速切换:** 5 个热门模型快速按钮，一键切换不同 AI
*   📐 **界面优化:** 320px 侧边栏，响应式设计，空间利用更高效
*   🆕 **实时进度:** v0.1.10 异步进度跟踪，告别黑盒等待
*   💾 **智能会话:** 状态持久化，页面刷新不丢失分析结果
*   🇨🇳 **中国优化:** A 股/港股数据 + 国产 LLM + 中文界面
*   🐳 **容器化:** Docker 一键部署，环境隔离，快速扩展
*   📄 **专业报告:** 多格式导出，自动生成投资建议
*   🛡️ **稳定可靠:** 多层数据源，智能降级，错误恢复

## 🔧 技术架构

**核心技术**: Python 3.10+ | LangChain | Streamlit | MongoDB | Redis
**AI 模型**: DeepSeek V3 | 阿里百炼 | Google AI | OpenRouter(60+ 模型) | OpenAI
**数据源**: Tushare | AkShare | FinnHub | Yahoo Finance
**部署**: Docker | Docker Compose | 本地部署

## 📚 文档和支持

*   **📖 完整文档**: [docs/](./docs/) - 安装指南、使用教程、API 文档
*   **🚨 故障排除**: [troubleshooting/](./docs/troubleshooting/) - 常见问题解决方案
*   **🔄 更新日志**: [CHANGELOG.md](./docs/releases/CHANGELOG.md) - 详细版本历史
*   **🚀 快速开始**: [QUICKSTART.md](./QUICKSTART.md) - 5 分钟快速部署指南

## 🙏 致敬原项目

感谢 [Tauric Research](https://github.com/TauricResearch) 团队创造的革命性多智能体交易框架 [TradingAgents](https://github.com/TauricResearch/TradingAgents)！

## 🆚 中文增强特色

**相比原版新增**: 智能新闻分析 | 多层次新闻过滤 | 新闻质量评估 | 统一新闻工具 | 多 LLM 提供商集成 | 模型选择持久化 | 快速切换按钮 | 实时进度显示 | 智能会话管理 | 中文界面 | A 股数据 | 国产 LLM | Docker 部署 | 专业报告导出 | 统一日志管理 | Web 配置界面 | 成本优化

## 📞 联系方式

-   **GitHub Issues**: [提交问题和建议](https://github.com/hsliuping/TradingAgents-CN/issues)
-   **邮箱**: hsliup@163.com
-   项目 Q 群：782124367
-   **原项目**: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
-   **文档**: [完整文档目录](docs/)

**更多信息和文档请参考：[TradingAgents-CN 原始仓库](https://github.com/hsliuping/TradingAgents-CN)**