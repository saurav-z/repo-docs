# TradingAgents-CN: 中文金融交易决策框架 🚀

**利用 AI 驱动的中文金融交易分析，轻松掌握 A 股、港股和美股市场！** [查看原始项目](https://github.com/hsliuping/TradingAgents-CN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.12-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Based on](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

> **v0.1.12 (最新更新)**: 智能新闻分析模块全面升级，提供更精准的中文金融市场分析，包括 A 股、港股和美股。

## 主要特性

*   **智能新闻分析**: 🆕 v0.1.12 智能新闻过滤，质量评估，相关性分析
*   **多LLM 提供商支持**: 🆕 v0.1.11 4 大提供商，60+ 模型，智能模型分类
*   **模型选择持久化**: 🆕 v0.1.11 URL 参数存储，刷新保持
*   **A股/港股/美股 全面支持**: 分析全球金融市场
*   **多智能体协作架构**: 模拟专业分析师团队
*   **Docker 一键部署**: 快速、便捷的环境搭建
*   **专业报告导出**: 生成 Markdown、Word 和 PDF 报告
*   **中文界面**: 全中文界面，更适合中国用户
*   **智能会话管理**: 状态持久化，页面刷新不丢失分析结果

## 核心优势

*   **易于使用**: 友好的 Web 界面和 CLI 界面。
*   **高度可定制**:  选择不同的 AI 模型，配置参数。
*   **高性能**: 利用缓存技术，加快分析速度。
*   **可扩展性**: 灵活添加新的 LLM 提供商和数据源。
*   **专业分析**: 提供专业的投资建议和风险评估。

## 核心功能

*   **多智能体协作**:
    *   市场分析师、基本面分析师、新闻分析师、情绪分析师
    *   看涨/看跌研究员进行深入分析
    *   交易员根据所有输入做出投资建议
    *   多层次风险评估和管理机制
*   **智能新闻分析**:
    *   AI驱动的新闻过滤和质量评估系统
    *   多层次过滤机制：基础、增强、集成
    *   统一新闻工具：整合多源新闻，提供智能检索接口
*   **LLM 模型支持**:
    *   支持 4 大 LLM 提供商: 阿里百炼、DeepSeek V3、Google AI、OpenRouter（60+ 模型）
    *   支持 OpenAI，Anthropic，Meta，Google 等主流模型
    *   自定义模型，满足个性化需求
*   **数据源与市场**:
    *   **A股**: Tushare, AkShare, 通达信
    *   **港股**: AkShare, Yahoo Finance
    *   **美股**: FinnHub, Yahoo Finance
    *   **新闻**: Google News
*   **Web 界面**:
    *   模型选择持久化
    *   5 个热门模型快速按钮，一键切换
    *   实时进度显示和分析步骤跟踪
    *   支持 Markdown、Word、PDF 报告导出
*   **CLI 界面**:
    *   用户界面清爽美观，技术日志独立管理
    *   多阶段进度跟踪，防止重复提示
    *   智能分析阶段显示预计耗时
    *   Rich 彩色输出，增强视觉效果
*   **其他**:
    *   Docker 一键部署
    *   专业报告导出：支持 Word, PDF, Markdown 格式

## 快速开始

### Docker 部署 (推荐)

1.  克隆项目：`git clone https://github.com/hsliuping/TradingAgents-CN.git`
2.  进入项目目录: `cd TradingAgents-CN`
3.  配置环境变量：`cp .env.example .env` (编辑 `.env` 文件，填入 API 密钥)
4.  启动服务: `docker-compose up -d --build` 或 `docker-compose up -d` (如果镜像已存在)
5.  访问应用: Web 界面: `http://localhost:8501`

### 本地部署

1.  安装依赖: `pip install -e .`
2.  启动应用: `python start_web.py`
3.  访问应用: `http://localhost:8501`

## 贡献

我们欢迎任何形式的贡献，包括代码、文档、测试和反馈。请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 获取更多信息。

## 许可证

本项目基于 Apache 2.0 许可证开源。