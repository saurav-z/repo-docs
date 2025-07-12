# TradingAgents-CN: 中文金融交易决策框架 (基于 TradingAgents)

🚀 **革新金融交易！** TradingAgents-CN 是一个基于多智能体大语言模型的中文金融交易决策框架，旨在为中国用户提供强大且易于使用的 AI 驱动的交易工具。 访问原项目: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).

**Key Features:**

*   🇨🇳 **中文支持**: 完整的中文文档、界面和分析结果。
*   🤖 **多智能体架构**: 模拟专业交易团队的决策流程。
*   🧠 **国产 LLM 集成**: 支持阿里云百炼等国产大语言模型。
*   📊 **A 股数据支持**:  集成通达信API等，支持A股实时行情和历史数据。
*   🌐 **Web 管理界面**: 直观易用的 Web 界面，方便配置和监控。
*   🗄️ **数据库支持**:  MongoDB 和 Redis 数据库，提升性能和数据持久化。
*   🚀 **高性能特性**:  并行处理、智能缓存、多层数据源，确保稳定可靠。
*   📖 **详尽文档**: 超过 50,000 字的中文技术文档，包括架构设计和使用指南。

---

## 核心功能与优势

TradingAgents-CN 旨在为金融交易领域带来革命性的变革。它基于 [Tauric Research](https://github.com/TauricResearch) 开发的 TradingAgents 框架，并针对中国市场进行了深度优化。

*   **多智能体协作**：该框架的核心在于模拟真实的交易公司结构。通过将任务分解为多个专业智能体（如分析师、研究员、交易员），实现更精准的决策和风险管理。

    *   **分析师团队**：基于基本面、技术面、新闻面和社交媒体四大维度，为交易决策提供全面分析。
    *   **研究员团队**：通过看涨/看跌研究员的辩论，对市场进行深入评估。
    *   **交易员智能体**：基于所有输入做出最终交易决策。
    *   **风险管理**：提供多层次的风险评估和管理机制。
    *   **管理层**：负责协调团队工作，确保决策的质量。
*   **多 LLM 支持**:  灵活选择合适的语言模型。

    *   **🇨🇳 阿里百炼**: 完整的阿里云百炼模型支持， 包括 qwen-turbo, qwen-plus-latest, qwen-max 等。
    *   **Google AI**: 支持 gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash 等 Gemini 模型。
    *   **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo (通过配置使用)。
    *   **Anthropic**: Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku (通过配置使用)。
    *   **智能混合**:  Google AI 推理 + 阿里百炼嵌入，实现更优效果。
*   **全面数据集成**:  整合广泛的金融数据源。

    *   **🇨🇳 A 股数据**： 通过通达信API，提供A股实时行情和历史数据。
    *   **美股数据**： 通过 FinnHub、Yahoo Finance 等，支持美股实时行情。
    *   **新闻数据**： 集成 Google News、财经新闻等实时新闻 API。
    *   **社交数据**： 通过 Reddit 情绪分析，了解市场情绪。
    *   **🗄️ 数据库支持**： 通过 MongoDB 持久化数据，并使用 Redis 进行高速缓存，提供卓越的性能。
    *   **🔄 智能降级**：MongoDB -> 通达信API -> 本地缓存，确保服务高可用性。
    *   **⚙️ 统一配置**：通过 .env 文件进行统一管理，启用和禁用功能，方便配置。

*   **高性能特性**：

    *   **并行处理**：通过多智能体的并行分析，显著提高效率。
    *   **智能缓存**：采用多层缓存策略，降低 API 调用成本，加快数据访问速度。
    *   **实时分析**：支持实时市场数据分析，为交易决策提供及时信息。
    *   **灵活配置**：智能体的行为和模型选择高度可定制。
    *   **数据目录配置**:  灵活配置数据存储路径。
    *   **⚡ 数据库加速**:  Redis 毫秒级缓存，MongoDB 持久化存储。
    *   **🔄 高可用架构**:  多层数据源降级，确保服务稳定性。

*   **🌐 Web 管理界面**:  直观的 Web 界面。

    *   **直观操作**： 基于 Streamlit 的现代化 Web 界面。
    *   **实时进度**：可视化分析过程，实时显示分析进度。
    *   **智能配置**： 提供 5 级研究深度，满足不同分析需求。
    *   **结果展示**： 结构化显示投资建议、目标价位、风险评估等。
    *   **🇨🇳 中文界面**： 用户界面和分析结果完全中文化。
    *   **🎛️ 配置管理**： API 密钥管理、模型选择、系统配置。
    *   **💰 Token 统计**： 实时 Token 使用统计和成本追踪。
    *   **💾 缓存管理**： 数据缓存状态监控和管理。

---

##  快速开始

要开始使用 TradingAgents-CN，请按照以下步骤进行操作：

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 创建虚拟环境
python -m venv env
# Windows
env\Scripts\activate
# Linux/macOS
source env/bin/activate

# 3. 安装基础依赖
pip install -r requirements.txt

# 4. 安装A股数据支持（可选）
pip install pytdx  # 通达信API，用于A股实时数据

# 5. 安装数据库支持（可选，推荐）
pip install -r requirements_db.txt  # MongoDB + Redis 支持
```

###  配置 API 密钥

在项目中使用 .env 文件管理 API 密钥，为了保证安全，请参考项目提供的安全指南。

####  🇨🇳 推荐：使用阿里百炼（国产大模型）

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，配置以下必需的API密钥：
DASHSCOPE_API_KEY=your_dashscope_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here

# 可选：Google AI API（支持Gemini模型）
GOOGLE_API_KEY=your_google_api_key_here

# 可选：数据库配置（提升性能，默认禁用）
MONGODB_ENABLED=false  # 设为true启用MongoDB
REDIS_ENABLED=false    # 设为true启用Redis
MONGODB_HOST=localhost
MONGODB_PORT=27018     # 使用非标准端口避免冲突
REDIS_HOST=localhost
REDIS_PORT=6380        # 使用非标准端口避免冲突
```

####  🌍 可选：使用国外模型

```bash
# OpenAI (需要科学上网)
OPENAI_API_KEY=your_openai_api_key

# Anthropic (需要科学上网)
ANTHROPIC_API_KEY=your_anthropic_api_key
```

###  数据库配置 (MongoDB + Redis)

为了获得最佳性能，建议启用数据库支持。

**方式一：Docker Compose (推荐)**

```bash
# 启动 MongoDB + Redis 服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 停止服务
docker-compose down
```

**方式二：手动安装**

```bash
# 安装数据库依赖
pip install -r requirements_db.txt

# 启动 MongoDB (默认端口 27017)
mongod --dbpath ./data/mongodb

# 启动 Redis (默认端口 6379)
redis-server
```

### 启动应用

####  🌐 Web 界面 (推荐)

```bash
# 激活虚拟环境
# Windows
.\env\Scripts\activate
# Linux/macOS
source env/bin/activate

# 启动Web管理界面
streamlit run web/app.py
```

在浏览器中访问 `http://localhost:8501` 。

####  💻 代码调用 (适合开发者)

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# 配置阿里百炼
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "dashscope"
config["deep_think_llm"] = "qwen-plus"      # 深度分析
config["quick_think_llm"] = "qwen-turbo"    # 快速任务

# 创建交易智能体
ta = TradingAgentsGraph(debug=True, config=config)

# 分析股票 (以苹果公司为例)
state, decision = ta.propagate("AAPL", "2024-01-15")

# 输出分析结果
print(f"推荐动作: {decision['action']}")
print(f"置信度: {decision['confidence']:.1%}")
print(f"风险评分: {decision['risk_score']:.1%}")
print(f"推理过程: {decision['reasoning']}")
```

---

## 详细文档

TradingAgents-CN 提供了全面的中文文档，帮助您深入了解框架的各个方面。

*   **项目概述**:  了解项目背景、核心价值和技术特色。
*   **架构设计**:  深入解析多智能体协作机制。
*   **智能体详解**:  了解四类专业分析师、交易员和风险管理。
*   **数据处理**:  掌握数据获取、处理和缓存的技术。
*   **配置指南**:  详细说明配置选项，进行性能调优。
*   **示例教程**:  从基础到高级，实战应用指南。
*   **常见问题**:  解决常见问题，提供故障排除方案。

详细文档目录，请访问 docs/ 目录下的 markdown 文件。

---

##  贡献指南

我们欢迎社区贡献，包括 Bug 修复、新功能、文档改进等。请参考贡献指南。

---

## 许可证

本项目基于 Apache 2.0 许可证开源。

---

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)