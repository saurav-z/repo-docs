# LangBot: Build Your Own AI Chatbot (with LLM & Plugins!)

LangBot is an open-source platform that makes building powerful, AI-powered chatbots a breeze, offering a flexible and customizable experience for various messaging platforms.  [See the original repository](https://github.com/langbot-app/LangBot).

<div align="center">

简体中文 / [English](README_EN.md) / [日本語](README_JP.md) / (PR for your language)

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

## Key Features

*   **🤖 Powerful AI Capabilities:** Supports diverse Large Language Models (LLMs), including Agent and RAG (Retrieval-Augmented Generation) features for enhanced conversational abilities. Deep integration with [Dify](https://dify.ai).
*   **💬 Multi-Platform Support:** Works seamlessly with popular messaging platforms like QQ, WeChat, Enterprise WeChat, Feishu, Discord, Telegram, and more.
*   **🛠️ Robust and Feature-Rich:** Includes built-in access control, rate limiting, and profanity filtering.  Supports multiple deployment options and custom pipeline configurations.
*   **🧩 Extensible with Plugins:**  Offers an event-driven plugin system for custom functionality.  Compatible with the Anthropic [MCP protocol](https://modelcontextprotocol.io/) with hundreds of available plugins.
*   **😻 Web-Based Management:**  Manage your LangBot instance through a user-friendly web interface, eliminating the need for manual configuration.

For detailed specifications and features, explore the [documentation](https://docs.langbot.app/zh/insight/features.html).

Check out the demo environment: [https://demo.langbot.dev/](https://demo.langbot.dev/)
*   Login: `demo@langbot.app`
*   Password: `langbot123456`

### Supported Platforms

| Platform          | Status | Notes                      |
| ----------------- | ------ | -------------------------- |
| QQ Personal       | ✅     | Private and group chats     |
| QQ Official Bot   | ✅     | Channels, private and group |
| WeChat            | ✅     |                            |
| Enterprise WeChat | ✅     |                            |
| WeChat Official Account | ✅     |                            |
| Feishu            | ✅     |                            |
| DingTalk          | ✅     |                            |
| Discord           | ✅     |                            |
| Telegram          | ✅     |                            |
| Slack             | ✅     |                            |

### Supported LLMs

| Model                                                                       | Status | Notes                                |
| --------------------------------------------------------------------------- | ------ | ------------------------------------ |
| [OpenAI](https://platform.openai.com/)                                    | ✅     | Access any OpenAI API-compatible models |
| [DeepSeek](https://www.deepseek.com/)                                       | ✅     |                                      |
| [Moonshot](https://www.moonshot.cn/)                                         | ✅     |                                      |
| [Anthropic](https://www.anthropic.com/)                                     | ✅     |                                      |
| [xAI](https://x.ai/)                                                     | ✅     |                                      |
| [智谱AI](https://open.bigmodel.cn/)                                          | ✅     |                                      |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)                  | ✅     | LLM and GPU resources                 |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | LLM and GPU resources                 |
| [302.AI](https://share.302.ai/SuTG99)                                       | ✅     | LLM Aggregation Platform              |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat)               | ✅     |                                      |
| [Dify](https://dify.ai)                                                  | ✅     | LLMOps Platform                      |
| [Ollama](https://ollama.com/)                                               | ✅     | Local LLM runner                     |
| [LMStudio](https://lmstudio.ai/)                                            | ✅     | Local LLM runner                     |
| [GiteeAI](https://ai.gitee.com/)                                           | ✅     | LLM API Aggregation Platform        |
| [SiliconFlow](https://siliconflow.cn/)                                      | ✅     | LLM Aggregation Platform              |
| [阿里云百炼](https://bailian.console.aliyun.com/)                            | ✅     | LLM and LLMOps Platform             |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | LLM and LLMOps Platform             |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅     | LLM Aggregation Platform              |
| [MCP](https://modelcontextprotocol.io/)                                      | ✅     | Supports tool access via MCP protocol |

### Text-to-Speech (TTS)

| Platform/Model                                                       | Notes                                   |
| -------------------------------------------------------------------- | --------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)                  | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)                    | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)                             | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)    |

### Text-to-Image

| Platform/Model      | Notes                                        |
| ------------------- | -------------------------------------------- |
| 阿里云百炼           | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access at http://localhost:5300 to start using LangBot.

Detailed documentation: [Docker Deployment](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Baota Panel Deployment

Available on Baota Panel. If you have Baota Panel installed, you can follow the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).

### Zeabur Cloud Deployment

Community-contributed Zeabur template.

[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)

### Railway Cloud Deployment

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)

### Manual Deployment

Run directly from the release version. See the documentation [Manual Deployment](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to stay informed about the latest developments.

![star gif](https://docs.langbot.app/star.gif)

## Community Contributions

Thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and other community members for their contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>