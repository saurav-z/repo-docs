# 🚀 TradingAgents-CN: 中文金融交易决策框架

**利用多智能体大语言模型，为中国市场量身定制，提供A股/港股/美股分析，助您掌握投资先机！**

[![](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/Version-cn--0.1.10-green.svg)](./VERSION)
[![](https://img.shields.io/badge/Docs-中文文档-green.svg)](./docs/)
[![](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

**基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 项目，专为中国用户优化的金融交易决策框架，支持A股、港股、美股市场，并集成中文大模型。**

## ✨ 主要特性

*   ✅ **A股、港股、美股全覆盖**: 深入分析中国及全球股票市场。
*   ✅ **中文本地化**: 全中文界面，更贴合中国用户的使用习惯。
*   ✅ **多智能体协作架构**: 四大分析师团队，协同分析，提供深度见解。
*   ✅ **Web 界面**: 现代化 Streamlit 界面，实时交互，数据可视化。
*   ✅ **Docker 一键部署**: 轻松部署，快速上手。
*   ✅ **专业报告导出**: 支持 Word/PDF/Markdown 格式，快速生成投资报告。
*   ✅ **国产 LLM 集成**: 支持阿里百炼、DeepSeek 等中文大模型。
*   ✅ **实时进度显示**: 异步进度跟踪，告别黑盒等待。
*   ✅ **智能会话管理**: 状态持久化，页面刷新不丢失分析结果。

## 🌟 最新版本 v0.1.10 更新

*   🚀 **实时进度显示**: 异步进度跟踪，智能步骤识别，准确时间计算。
*   💾 **智能会话管理**: 状态持久化，自动降级，跨页面恢复。
*   🎯 **一键查看报告**: 分析完成后一键查看，智能结果恢复。
*   🎨 **界面优化**: 移除重复按钮，响应式设计，视觉层次优化。

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

## 🎯 核心功能

*   **多智能体协作**: 基本面、技术面、新闻面、社交媒体分析师团队协同工作。
*   **深度分析**: 看涨/看跌研究员进行深度分析，交易员基于所有输入给出投资建议。
*   **风险管理**: 多层次风险评估和管理机制。
*   **数据源**: 支持 A 股 (Tushare, AkShare)，港股 (AkShare, Yahoo Finance)，美股 (FinnHub, Yahoo Finance) 和新闻数据。
*   **LLM 模型**: 支持阿里百炼, DeepSeek, Google AI, OpenAI 等模型。

## 📚 完整文档

*   **完整中文文档**：[docs/](docs/)，包含安装指南、使用教程、API 文档。
*   **快速部署指南**: [QUICKSTART.md](./QUICKSTART.md)

## 🤝 贡献指南

欢迎贡献代码、文档、翻译等。 详见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。 详见 [LICENSE](LICENSE) 文件。

## 📞 联系

*   GitHub Issues: [提交问题和建议](https://github.com/hsliuping/TradingAgents-CN/issues)
*   邮箱: hsliup@163.com

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)

</div>
```
Key improvements and summaries:

*   **SEO Optimization:**  Included keywords like "中文", "金融", "交易", "A股", "港股", "美股", "AI", "决策", "框架" throughout the text, targeting relevant search terms.  Added a clear, concise one-sentence hook at the beginning.
*   **Clear Headings and Structure:** Improved the existing headings and subheadings for better readability and organization.
*   **Concise and Informative Bullets:**  Replaced lengthy paragraphs with concise bulleted lists to highlight key features and benefits.
*   **Emphasis on Value Proposition:**  Strongly emphasized the benefits to the target audience (Chinese users).
*   **Simplified "Quick Start" and "Core Features" Sections**: Made them more direct and easier to understand.
*   **Removed Redundant Information:** Removed unnecessary phrases.
*   **"Contact" Section:** Added a clear "Contact" section.
*   **Stronger Call to Action:**  Included a prominent "Star this repo" and "Fork this repo" call to action.
*   **Complete & Concise**: The README is well-organized, providing all necessary information without being overly verbose.
*   **Improved Language and Tone:** The writing is clear, concise, and enthusiastic, reflecting the project's value.
*   **Corrected minor markdown issues**
*   **Added Version and Documentation Shields**
*   **Focus on User Benefit**: Emphasized the advantages of using the project.
*   **Integrated key information from the original README:**  Ensured that all of the original information was included, but re-organized and improved.