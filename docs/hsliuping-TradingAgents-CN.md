# TradingAgents-CN: 中文金融交易决策框架 - AI 赋能，中文优化 (基于 TauricResearch/TradingAgents)

[![](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/Version-cn--0.1.12-green.svg)](./VERSION)
[![](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

🚀 **开启 AI 金融新时代：** TradingAgents-CN 是一个基于多智能体大语言模型的中文金融交易决策框架，专为中文用户优化，提供完整的 A股/港股/美股分析能力。 **[查看原始项目](https://github.com/hsliuping/TradingAgents-CN) 以获取更多信息!**

## 🎯 核心特性

*   **智能新闻分析 (v0.1.12 新增)**：AI 驱动的新闻过滤、质量评估和相关性分析。
*   **多 LLM 提供商集成 (v0.1.11)**：支持 阿里百炼, DeepSeek V3, Google AI, OpenRouter (60+ 模型)。
*   **模型选择持久化 (v0.1.11)**：通过 URL 参数保存模型配置，刷新不丢失。
*   **A股/港股/美股支持**：全面覆盖中国股票市场，提供实时行情和分析。
*   **多智能体协作**: 基本面、技术面、新闻面、社交媒体四大分析师 + 看涨/看跌研究员 + 交易员。
*   **Web 界面 & 命令行界面**: 提供友好的用户体验，方便操作。
*   **专业报告导出**: 支持 Word/PDF/Markdown 格式，一键生成投资建议。
*   **Docker 部署**: 一键部署，环境隔离，快速扩展。

## 🆕 最新更新 - v0.1.12 亮点

### 🧠 智能新闻分析模块

*   **智能新闻过滤器**:  基于 AI 的新闻相关性评分和质量评估。
*   **多层次过滤机制**:  基础、增强、集成三级过滤。
*   **新闻质量评估**: 自动识别和过滤低质量、重复、无关新闻。
*   **统一新闻工具**:  整合多个新闻源，提供统一的新闻获取接口。

### 🔧 技术修复和优化

*   **DashScope 适配器修复**: 解决工具调用兼容性问题。
*   **DeepSeek 死循环修复**: 修复新闻分析师的无限循环问题。
*   **LLM 工具调用增强**: 提升工具调用的可靠性和稳定性。
*   **新闻检索器优化**:  增强新闻数据获取和处理能力。

### 📚 完善测试和文档

*   **全面测试覆盖**: 新增 15+ 个测试文件，覆盖所有新功能。
*   **详细技术文档**: 新增 8 个技术分析报告和修复文档。
*   **用户指南完善**: 新增新闻过滤使用指南和最佳实践。
*   **演示脚本**:  提供完整的新闻过滤功能演示。

### 🗂️ 项目结构优化

*   **文档分类整理**: 按功能将文档分类到 `docs` 子目录。
*   **示例代码归位**: 演示脚本统一到 `examples` 目录。
*   **根目录整洁**: 保持根目录简洁，提升项目专业度。

## 快速开始

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
# 1. 升级 pip
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -e .

# 3. 启动应用
python start_web.py

# 4. 访问 http://localhost:8501
```

### 📊 开始分析

1.  **选择模型**:  DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票**: `000001` (A股) / `AAPL` (美股) / `0700.HK` (港股)
3.  **开始分析**: 点击"🚀 开始分析"按钮
4.  **实时跟踪**:  观察实时进度和分析步骤
5.  **查看报告**: 点击"📊 查看分析报告"按钮
6.  **导出报告**:  支持 Word/PDF/Markdown 格式

## 🎯 核心优势

*   **🇨🇳 中文优化**: A股/港股数据 + 国产LLM + 中文界面
*   **🐳 容器化**: Docker 一键部署，环境隔离，快速扩展
*   **📄 专业报告**: 多格式导出，自动生成投资建议
*   **🛡️ 稳定可靠**: 多层数据源，智能降级，错误恢复

## 📚 详细文档

*   [**📖 快速开始**](docs/overview/quick-start.md) -  快速上手指南
*   [**🏛️ 系统架构**](docs/architecture/system-architecture.md) -  深入了解系统设计
*   [**🤖 智能体详解**](docs/agents/analysts.md) - 了解每个分析师的职责
*   [**❓ 常见问题**](docs/faq/faq.md) -  解决您的问题

## 🤝 贡献

欢迎贡献代码、文档和建议！  查阅 [CONTRIBUTORS.md](CONTRIBUTORS.md) 了解更多信息。

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢 [Tauric Research](https://github.com/TauricResearch) 团队创建的原始项目 [TradingAgents](https://github.com/TauricResearch/TradingAgents)！

---

**请为我们点亮 ⭐ Star， 如果本项目对您有所帮助！**