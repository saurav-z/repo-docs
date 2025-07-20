# 🚀 TradingAgents-CN: 中文金融交易决策框架

**使用多智能体大语言模型，为中文用户提供强大的A股、港股、美股分析能力，助您做出更明智的投资决策！**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.10-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-中文文档-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/基于-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

## ✨ 核心亮点

*   **🇨🇳 中文优化**:  全面支持A股/港股数据、中文界面、国产LLM模型。
*   **🚀 实时进度**: v0.1.10新增异步进度跟踪，告别黑盒等待。
*   **💾 智能会话**:  状态持久化，页面刷新不丢失分析结果。
*   **🐳 容器化部署**:  Docker一键部署，环境隔离，快速扩展。
*   **📄 专业报告**:  多格式导出，自动生成投资建议。

## 🎯 主要特点

*   **多智能体架构**:  由基本面、技术面、新闻、情绪分析师、看涨/看跌研究员、交易员、风险管理员协作完成分析。
*   **支持市场**:  A股、港股、美股实时行情和数据支持。
*   **LLM 模型**:  支持DeepSeek V3、阿里百炼、Google AI、OpenAI等。
*   **Web界面**: Streamlit界面，配置管理。
*   **CLI体验**:  界面与日志分离、智能进度显示、Rich彩色输出。

## 🆕 v0.1.10 最新更新

*   **🚀 实时进度显示**: 异步进度跟踪，智能步骤识别，准确时间计算。
*   **💾 智能会话管理**: 状态持久化，页面刷新不丢失分析结果。
*   **🎯 一键查看报告**: 分析完成后一键查看报告。
*   **🎨 界面优化**: 移除重复按钮，响应式设计，视觉层次优化。

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
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python start_web.py

# 3. 访问 http://localhost:8501
```

### 📊 开始分析

1.  **选择模型**: DeepSeek V3 / 通义千问 / Gemini
2.  **输入股票**: `000001` (A股) / `AAPL` (美股) / `0700.HK` (港股)
3.  **开始分析**: 点击"🚀 开始分析"按钮
4.  **实时跟踪**: 观察实时进度和分析步骤
5.  **查看报告**: 点击"📊 查看分析报告"按钮
6.  **导出报告**: 支持Word/PDF/Markdown格式

## 📚 深入了解

*   **📖 完整文档**: [docs/](./docs/)  - 详细的安装、使用教程、API文档，**50,000+字中文文档**，从入门到精通!
*   **🚨 故障排除**: [troubleshooting/](./docs/troubleshooting/) - 常见问题解决方案。

## 🚀 项目基于

感谢 [Tauric Research](https://github.com/TauricResearch) 团队开发的 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 项目！

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](https://github.com/hsliuping/TradingAgents-CN) | [🍴 Fork this repo](https://github.com/hsliuping/TradingAgents-CN/fork) | [📖 Read the docs](./docs/)

</div>