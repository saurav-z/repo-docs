# TradingAgents-CN: 中文金融AI交易决策框架 🚀

**利用多智能体大语言模型，专为中文市场优化的AI金融交易决策框架，助您深入分析A股、港股和美股，做出更明智的投资决策！** ([原项目地址](https://github.com/hsliuping/TradingAgents-CN))

## ✨ 核心特性

*   **🤖 多智能体架构:** 由基本面、技术面、新闻面和情绪分析师组成的专业团队，共同决策。
*   **🇨🇳 中文优化:** 深度支持A股、港股市场，以及中文新闻与数据。
*   **🧠 智能新闻分析 (v0.1.12):** AI驱动的新闻过滤、质量评估和相关性分析，提升信息筛选效率。
*   **💻 灵活部署:** 支持 Docker 一键部署和本地部署，方便快捷。
*   **🌐 多LLM支持 (v0.1.11):** 集成4大提供商，60+模型，包括国产大模型，满足个性化需求。
*   **📊 专业报告导出:** 支持 Markdown, Word, 和 PDF 格式的分析报告，方便分享和归档。
*   **🚀 实时进度显示 (v0.1.10):** 可视化分析过程，告别黑盒等待。
*   **💾 模型选择持久化 (v0.1.11):** 基于URL参数存储，刷新后配置不丢失。

## 🆕 最新版本更新 (v0.1.12) 🚀

### 🧠 智能新闻分析模块全面升级

*   **智能新闻过滤器:** 基于AI的新闻相关性评分和质量评估
*   **多层次过滤机制:** 基础过滤、增强过滤、集成过滤三级处理
*   **新闻质量评估:** 自动识别和过滤低质量、重复、无关新闻
*   **统一新闻工具:** 整合多个新闻源，提供统一的新闻获取接口

### 🔧 技术修复和优化

*   **DashScope适配器修复**: 解决工具调用兼容性问题
*   **DeepSeek死循环修复**: 修复新闻分析师的无限循环问题
*   **LLM工具调用增强**: 提升工具调用的可靠性和稳定性
*   **新闻检索器优化**: 增强新闻数据获取和处理能力

### 📚 完善测试和文档

*   **全面测试覆盖**: 新增15+个测试文件，覆盖所有新功能
*   **详细技术文档**: 新增8个技术分析报告和修复文档
*   **用户指南完善**: 新增新闻过滤使用指南和最佳实践
*   **演示脚本**: 提供完整的新闻过滤功能演示

### 🗂️ 项目结构优化

*   **文档分类整理**: 按功能将文档分类到docs子目录
*   **示例代码归位**: 演示脚本统一到examples目录
*   **根目录整洁**: 保持根目录简洁，提升项目专业度

## 🚀 快速开始

### 🐳 Docker部署 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/hsliuping/TradingAgents-CN.git
cd TradingAgents-CN

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 3. 启动服务
# 首次启动或代码变更时（需要构建镜像）
docker-compose up -d --build

# 日常启动（镜像已存在，无代码变更）
docker-compose up -d

# 智能启动（自动判断是否需要构建）
# Windows环境
powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1

# Linux/Mac环境
chmod +x scripts/smart_start.sh && ./scripts/smart_start.sh

# 4. 访问应用
# Web界面: http://localhost:8501
```

### 💻 本地部署

```bash
# 1. 升级pip (重要！避免安装错误)
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -e .

# 3. 启动应用
python start_web.py

# 4. 访问 http://localhost:8501
```

## 📚 详细文档与支持

*   **📖 完整文档:** [docs/](./docs/) - 包含安装指南、使用教程、API文档等，**深度剖析，超过 50,000 字！**
*   **🚨 故障排除:** [docs/troubleshooting/](./docs/troubleshooting/) - 常见问题解决方案
*   **🔄 更新日志:** [CHANGELOG.md](./docs/releases/CHANGELOG.md) - 详细版本历史

## 🤝 贡献指南

欢迎贡献代码、文档、改进建议等！  详见 [CONTRIBUTING.md](CONTRIBUTORS.md) 。

## 📄 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 开源。

## 🙏 感谢

特别感谢 [Tauric Research](https://github.com/TauricResearch) 团队的 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 项目。

<div align="center">

**🌟 如果本项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)

</div>
```

**Key improvements and SEO considerations:**

*   **Concise Hook:** Starts with a strong one-sentence hook to grab attention.
*   **Keyword Optimization:** Includes relevant keywords like "中文金融," "AI交易," "A股," "港股," "美股," "交易决策," and "多智能体." These keywords appear naturally throughout the document.
*   **Clear Headings:**  Uses clear, descriptive headings (e.g., "核心特性," "最新版本更新") to improve readability and organization.
*   **Bulleted Lists:**  Employs bulleted lists to highlight key features, making the information easy to scan.
*   **Feature-Rich Description:** Expands on the key features with detailed descriptions.
*   **SEO-friendly title**: Optimized the title for searchability.
*   **Call to Actions:** Includes calls to action such as "Star this repo" and "Read the docs."
*   **Concise and Informative:**  The summary is more concise and focuses on the most important information for potential users.
*   **Clear Structure:** Provides a logical flow from introduction to quick start to contribution guidelines.
*   **Markdown Formatting:** Maintains proper Markdown formatting for easy rendering on GitHub.
*   **Focus on User Benefits:** Highlights the benefits for the user, such as "make more informed investment decisions."
*   **Links Back to Original Repo:**  Maintains the important attribution and provides a link back to the original project.
*   **Updated Descriptions:** Includes a detailed and improved description of the key features.
*   **Key Updates:** Features the most important upgrades at the top of the description.