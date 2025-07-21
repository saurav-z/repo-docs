# 🚀 TradingAgents-CN: 中文金融交易决策框架 - AI驱动的投资分析 (基于 TradingAgents)

基于多智能体大语言模型，**TradingAgents-CN** 致力于为中文用户提供强大的金融交易决策支持。它以 **开源框架 TradingAgents** 为基础，针对中国市场进行了深度优化，全面支持A股、港股和美股分析，助您在金融市场中做出更明智的投资决策。

**[访问原始项目](https://github.com/hsliuping/TradingAgents-CN)** | **[查看文档](./docs/)**

## ✨ 主要特性

*   **🇨🇳 中文支持**: 专为中文用户优化，提供全面的中文界面和分析。
*   **📈 市场覆盖**: 全面支持A股、港股和美股市场。
*   **🤖 多智能体架构**: 模拟专业分析师团队，进行全方位的分析。
*   **🐳 Docker部署**: 一键部署，快速搭建和扩展。
*   **📊 报告导出**: 支持Word, PDF和Markdown格式的专业报告。
*   **🧠 LLM集成**: 兼容多种LLM模型，包括DeepSeek, 阿里百炼, OpenAI等。
*   **🆕 实时进度**: v0.1.10版本新增异步进度跟踪，告别黑盒等待。
*   **💾 智能会话**: 状态持久化，页面刷新不丢失分析结果。

## 🚀 v0.1.10 最新更新

*   **🚀 实时进度显示**: 全新异步进度跟踪，准确显示分析进度和步骤。
*   **📊 智能会话管理**: 页面刷新后恢复分析状态和历史报告。
*   **🎨 用户体验优化**: 界面简化，响应式设计，错误处理增强。

## 💡 核心功能

*   **多智能体协作**: 基本面、技术面、新闻面、情绪面等多维度分析。
*   **深度分析**: 看涨/看跌研究员进行深入的辩论分析。
*   **智能决策**: 交易员基于所有输入做出最终投资建议。
*   **风险管理**: 多层次风险评估和管理机制。

## 快速上手

### 🐳 Docker部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量 (编辑 .env 文件并填入API密钥)
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

## 🎯 核心优势

*   **中国优化**: 针对中国市场的数据和用户体验进行优化。
*   **快速部署**: Docker一键部署，快速启动。
*   **专业报告**: 生成专业投资建议。

## 📚 文档

*   [完整文档目录](./docs/) - 详细的安装指南、使用教程和API文档。

## 许可证

本项目基于 [Apache 2.0](LICENSE) 许可证开源。

## 🙏 致谢

感谢 [Tauric Research](https://github.com/TauricResearch) 团队创造的 [TradingAgents](https://github.com/TauricResearch/TradingAgents)！

---

🌟 **如果您喜欢本项目，请给个 Star！**