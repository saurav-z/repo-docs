<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot"/>
</a>
</p>

<div align="center">

<a href="https://hellogithub.com/repository/langbot-app/LangBot" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=5ce8ae2aa4f74316bf393b57b952433c&claim_uid=gtmc6YWjMZkT21R" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

[English](README_EN.md) / 简体中文 / [繁體中文](README_TW.md) / [日本語](README_JP.md) / (PR for your language)

[![Discord](https://img.shields.io/discord/1335141740050649118?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb)](https://discord.gg/wdNEHETs87)
[![QQ Group](https://img.shields.io/badge/%E7%A4%BE%E5%8C%BAQQ%E7%BE%A4-966235608-blue)](https://qm.qq.com/q/JLi38whHum)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/langbot-app/LangBot)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/langbot-app/LangBot)](https://github.com/langbot-app/LangBot/releases/latest)
<img src="https://img.shields.io/badge/python-3.10 ~ 3.13 -blue.svg" alt="python">
[![star](https://gitcode.com/RockChinQ/LangBot/star/badge.svg)](https://gitcode.com/RockChinQ/LangBot)

<a href="https://langbot.app">项目主页</a> ｜
<a href="https://docs.langbot.app/zh/insight/guide.html">部署文档</a> ｜
<a href="https://docs.langbot.app/zh/plugin/plugin-intro.html">插件介绍</a> ｜
<a href="https://github.com/langbot-app/LangBot/issues/new?assignees=&labels=%E7%8B%AC%E7%AB%8B%E6%8F%92%E4%BB%B6&projects=&template=submit-plugin.yml&title=%5BPlugin%5D%3A+%E8%AF%B7%E6%B1%82%E7%99%BB%E8%AE%B0%E6%96%B0%E6%8F%92%E4%BB%B6">提交插件</a>

</div>

# LangBot: Build Your Own AI Chatbot Platform

**LangBot is an open-source platform for developing native instant messaging (IM) chatbots, enabling you to easily create and deploy AI-powered bots across various messaging platforms.**

Key Features:

*   🤖 **Multi-Platform Support:** Works with QQ, QQ Channel, Enterprise WeChat, WeChat, Feishu, Discord, Telegram, and more.
*   💬 **Advanced AI Capabilities:** Supports Large Language Models (LLMs) for dialog, Agents, RAG (Retrieval-Augmented Generation), and MCP (Model Context Protocol) integration, including support for Dify.
*   🧩 **Extensible with Plugins:** Offers a robust plugin system for custom features, event-driven architecture, and component extensions.  Includes over hundreds of existing plugins.
*   🌐 **Web UI & Management:** Easy management via web browser, no manual config file editing needed.
*   🛠️ **High Stability & Functionality:** Includes access control, rate limiting, profanity filters, and multiple deployment options for robust performance.

  See the [LangBot GitHub Repository](https://github.com/langbot-app/LangBot) for more information and to get started.

## 🚀 Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access at http://localhost:5300.

Detailed instructions: [Docker Deployment](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options:

*   **BaoTa Panel:** Available on BaoTa Panel.  Follow the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html) for setup.
*   **Zeabur:** Community-contributed Zeabur template.  [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway:** Deploy with Railway. [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Use the releases.  See [Manual Deployment](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## 🌟 Stay Updated

Star and watch the repository to get the latest updates!

![star gif](https://docs.langbot.app/star.gif)

## ✨ Features in Detail

*   **Conversational AI & Agents:** Support for multiple LLMs.  Includes multi-turn conversations, tool use, multimodal capabilities, streaming output, and built-in RAG functionality, and Dify integration.
*   **Cross-Platform Compatibility:** Supports a wide range of messaging platforms including QQ (personal and official bots), Enterprise WeChat, WeChat, Feishu, Discord, Telegram, and more.
*   **Robust & Feature-Rich:** Native support for access control, rate limiting, and profanity filtering. Simple configuration and various deployment options. Supports multi-pipeline configurations for different chatbot use cases.
*   **Plugin Ecosystem & Active Community:** Plugin system with event-driven architecture and component extensions. Compatible with Anthropic's [MCP Protocol](https://modelcontextprotocol.io/). A large number of plugins are available.
*   **Web-Based Management Panel:** Provides a user-friendly interface for managing your LangBot instance, eliminating the need to manually configure settings.

For detailed specifications, visit the [documentation](https://docs.langbot.app/zh/insight/features.html).

Or visit the demo environment: https://demo.langbot.dev/

  -   Login: Email: `demo@langbot.app`  Password: `langbot123456`
  -   Note: WebUI demo only. Do not enter sensitive information.

### Supported Messaging Platforms:

| Platform          | Status | Notes                                     |
| ----------------- | ------ | ----------------------------------------- |
| QQ (Personal)     | ✅     | Private & Group Chat                      |
| QQ (Official Bot) | ✅     | Supports channels, private, and group chat |
| Enterprise WeChat | ✅     |                                           |
| WeChat            | ✅     |                                           |
| WeChat Official Account | ✅     |                                           |
| Feishu            | ✅     |                                           |
| DingTalk          | ✅     |                                           |
| Discord           | ✅     |                                           |
| Telegram          | ✅     |                                           |
| Slack             | ✅     |                                           |

### Supported LLMs

| Model                                                                | Status | Notes                                                                |
| -------------------------------------------------------------------- | ------ | -------------------------------------------------------------------- |
| [OpenAI](https://platform.openai.com/)                             | ✅     | Supports all OpenAI API format models                                  |
| [DeepSeek](https://www.deepseek.com/)                              | ✅     |                                                                      |
| [Moonshot](https://www.moonshot.cn/)                              | ✅     |                                                                      |
| [Anthropic](https://www.anthropic.com/)                            | ✅     |                                                                      |
| [xAI](https://x.ai/)                                             | ✅     |                                                                      |
| [ZhipuAI](https://open.bigmodel.cn/)                                | ✅     |                                                                      |
| [YouCloud](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)         | ✅     | Large model and GPU resources platform                              |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | Large model and GPU resources platform                              |
| [ShengSuanYun](https://www.shengsuanyun.com/?from=CH_KYIPP758)        | ✅     | Large model and GPU resources platform                              |
| [302.AI](https://share.302.ai/SuTG99)                                 | ✅     | Large model aggregation platform                                     |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat)          | ✅     |                                                                      |
| [Dify](https://dify.ai)                                            | ✅     | LLMOps platform                                                     |
| [Ollama](https://ollama.com/)                                       | ✅     | Local large model running platform                                  |
| [LMStudio](https://lmstudio.ai/)                                    | ✅     | Local large model running platform                                  |
| [GiteeAI](https://ai.gitee.com/)                                    | ✅     | Large model interface aggregation platform                          |
| [SiliconFlow](https://siliconflow.cn/)                              | ✅     | Large model aggregation platform                                     |
| [Aliyun Baichuan](https://bailian.console.aliyun.com/)               | ✅     | Large model aggregation platform, LLMOps platform                    |
| [VolcEngine](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | Large model aggregation platform, LLMOps platform                    |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅     | Large model aggregation platform                                     |
| [MCP](https://modelcontextprotocol.io/)                            | ✅     | Supports obtaining tools through the MCP protocol                   |

### TTS (Text-to-Speech)

| Platform/Model                               | Notes                                  |
| -------------------------------------------- | -------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [HaiTing AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)          | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image

| Platform/Model        | Notes                                            |
| --------------------- | ------------------------------------------------ |
| Aliyun Baichuan       | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## 💖 Community Contribution

Thanks to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and other community members for their contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and explanations:

*   **SEO Optimization:**  Uses keywords like "AI chatbot," "open-source," "LLM," "instant messaging," etc.  Includes platform and LLM names.
*   **One-Sentence Hook:** Starts with a strong, concise sentence to immediately grab the reader's attention.
*   **Clear Headings:** Uses headings to organize information logically (Features, Getting Started, Supported Platforms, etc.) and uses emojis for visual appeal.
*   **Bulleted Key Features:** Uses bullet points for readability and easy skimming.  Focuses on the most important features.
*   **Concise Language:**  Avoids unnecessary jargon and focuses on the benefits.
*   **Comprehensive Platform & LLM Lists:**  Provides detailed tables with status updates. This is valuable information for users.
*   **Clear Links:**  Provides links to all relevant resources (documentation, demo, etc.).
*   **Call to Action:** Includes a clear call to action (Star and Watch the repo).
*   **Complete README:** Includes everything from the original but improves readability and organization.
*   **Community Acknowledgement:** The "Community Contribution" section is important for attracting and retaining contributors.
*   **Updated for modern usage and standards.**
*   **More thorough and complete.**