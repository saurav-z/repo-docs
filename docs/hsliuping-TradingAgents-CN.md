# 🇨🇳 TradingAgents-CN:  中文金融交易决策框架 - 提升您的投资分析能力

**🚀  基于多智能体大语言模型的中文金融交易框架，专为中国市场优化，提供A股/港股/美股分析能力。** 
[访问原项目](https://github.com/hsliuping/TradingAgents-CN)

## ✨ 主要特性

*   📊  **实时进度显示：**  告别黑盒等待，异步进度跟踪，分析过程一目了然！
*   💾  **智能会话管理：**  状态持久化，页面刷新不丢失分析结果！
*   🇨🇳  **中国优化：**  A股/港股数据、国产LLM支持、中文界面，更懂中国市场！
*   🐳  **Docker 一键部署：**  环境隔离，快速部署，方便快捷！
*   📄  **专业报告导出：**  多种格式 (Word/PDF/Markdown)，自动生成投资建议，轻松分享！

## 核心功能

*   **🤖 多智能体架构：**  专业分工，涵盖基本面、技术面、新闻面、社交媒体分析师，研究员深度分析，交易员决策。
*   **📈 市场覆盖：**  支持 A股、港股、美股数据，覆盖沪深、港交所、纽交所、纳斯达克。
*   **🧠 多LLM支持：**  支持 OpenAI、阿里百炼、DeepSeek、Google AI 等模型。
*   **🐳 Docker部署：**  快速启动，环境隔离，简化部署流程。
*   **📊 专业报告：**  多种格式 (Word/PDF/Markdown) 导出分析报告，自动生成投资建议。

## ✨  v0.1.10 最新更新

*   🚀  **实时进度显示:** 异步进度跟踪，准确显示分析耗时。
*   📊  **智能会话管理:**  状态持久化，页面刷新后恢复分析状态。
*   🎨  **用户体验优化:**  界面简化，响应式设计。

## 快速上手

### 🐳 Docker 部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2.  配置环境变量
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

## 核心优势

*   🇨🇳  **中国市场优化:** 深度支持 A股/港股市场。
*   🐳  **Docker 部署:** 简化部署流程，快速上手。
*   📄  **专业报告:** 多格式导出，方便分享和分析。
*   🧠  **国产 LLM 集成:** 支持阿里百炼等模型，优化成本。
*   🚀  **实时进度显示:**  告别等待，可视化分析过程。
*   💾  **智能会话管理:** 页面刷新不丢失分析结果。

## 示例分析

1.  **选择模型:** DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票代码:** `000001` (A股) / `AAPL` (美股) / `0700.HK` (港股)
3.  **开始分析:** 点击 “🚀 开始分析” 按钮
4.  **实时跟踪:** 观察分析进度。
5.  **查看报告:** 点击 "📊 查看分析报告" 按钮。
6.  **导出报告:**  选择报告格式导出 (Word/PDF/Markdown)。

## 架构与技术栈

*   **核心技术:** Python, LangChain, Streamlit, MongoDB, Redis
*   **AI 模型:** DeepSeek V3, 阿里百炼, Google AI, OpenAI
*   **数据源:** Tushare, AkShare, FinnHub, Yahoo Finance
*   **部署:** Docker, Docker Compose, 本地部署

## 🤝 贡献

欢迎贡献代码、文档或提出建议！  查看 [贡献指南](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目基于 Apache 2.0 许可证开源。  查看 [LICENSE](LICENSE) 了解详情。

## 📚 更多信息

*   **文档**: [docs/](docs/) - 完整文档，包括安装、使用、API 和故障排除。
*   **更新日志**: [CHANGELOG.md](./docs/releases/CHANGELOG.md)
*   **原项目**:  [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
*   **联系我们**:  通过 GitHub Issues 提出问题和建议。

**🌟  如果您觉得这个项目对您有帮助，请给我们一个 Star!  感谢您的支持！**