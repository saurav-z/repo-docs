# LangBot: Open-Source LLM-Powered IM Robot Platform

**Create powerful and customizable AI-powered chatbots for various messaging platforms with LangBot, the open-source platform that makes it easy.** ([Original Repository](https://github.com/langbot-app/LangBot))

<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot" width="400"/>
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

LangBot is an open-source platform designed to simplify the development of AI-powered IM (Instant Messaging) bots. It offers out-of-the-box features, including Agent, RAG (Retrieval-Augmented Generation), and MCP (Model Context Protocol) functionalities.  LangBot seamlessly integrates with popular messaging platforms worldwide and provides a rich set of APIs for custom development.

## Key Features

*   **Versatile LLM Capabilities**: Supports a variety of large language models, enabling multi-turn conversations, tool usage, and multimodal capabilities. Includes built-in RAG for enhanced knowledge retrieval and seamless integration with [Dify](https://dify.ai).
*   **Multi-Platform Support**: Works with popular messaging platforms like QQ, QQ Channels, WeChat for Business, Personal WeChat, Feishu, Discord, and Telegram.
*   **Robust & Feature-Rich**: Provides built-in access control, rate limiting, and profanity filtering. Configuration is simple, and multiple deployment methods are available. Offers multi-pipeline configurations for diverse bot applications.
*   **Extensible with Plugins**: Supports event-driven and component-based plugin architecture, with compatibility for the Anthropic [MCP protocol](https://modelcontextprotocol.io/). Hundreds of plugins are already available.
*   **Web-Based Management**: Manage your LangBot instance via a user-friendly web interface, eliminating the need for manual configuration file editing.

Explore the comprehensive feature set in our [detailed documentation](https://docs.langbot.app/zh/insight/features.html).

Alternatively, try out the demo environment at https://demo.langbot.dev/:

*   **Login:** Email: `demo@langbot.app` Password: `langbot123456`
*   **Note:** This is a public demo. Please avoid entering sensitive information.

### Supported Messaging Platforms

| Platform           | Status | Notes                                 |
| ------------------ | ------ | ------------------------------------- |
| QQ Personal       | ✅     | Private and group chats                |
| QQ Official Bot   | ✅     | Supports Channels, private & group chats |
| WeChat for Business | ✅     |                                       |
| WeChat for Business | ✅     |                                       |
| Personal WeChat    | ✅     |                                       |
| WeChat Official Account | ✅     |                                       |
| Feishu             | ✅     |                                       |
| DingTalk           | ✅     |                                       |
| Discord            | ✅     |                                       |
| Telegram           | ✅     |                                       |
| Slack              | ✅     |                                       |

### Supported LLMs

| Model                                  | Status | Notes                                    |
| -------------------------------------- | ------ | ---------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Compatible with any OpenAI-compatible model |
| [DeepSeek](https://www.deepseek.com/) | ✅     |                                          |
| [Moonshot](https://www.moonshot.cn/)   | ✅     |                                          |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                          |
| [xAI](https://x.ai/)                    | ✅     |                                          |
| [智谱AI](https://open.bigmodel.cn/)    | ✅     |                                          |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     | GPU and LLM resources                 |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | GPU and LLM resources                 |
| [302.AI](https://share.302.ai/SuTG99)  | ✅     | LLM Aggregation Platform                    |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     |                                          |
| [Dify](https://dify.ai)                 | ✅     | LLMOps Platform                          |
| [Ollama](https://ollama.com/)           | ✅     | Local LLM Platform                       |
| [LMStudio](https://lmstudio.ai/)        | ✅     | Local LLM Platform                       |
| [GiteeAI](https://ai.gitee.com/)       | ✅     | LLM API Aggregation Platform            |
| [SiliconFlow](https://siliconflow.cn/) | ✅     | LLM Aggregation Platform                  |
| [阿里云百炼](https://bailian.console.aliyun.com/) | ✅     | LLM Aggregation Platform, LLMOps Platform    |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | LLM Aggregation Platform, LLMOps Platform    |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅     | LLM Aggregation Platform                  |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports tools via MCP protocol         |

### Text-to-Speech (TTS)

| Platform/Model                          | Notes                                       |
| --------------------------------------- | ------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)   | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image

| Platform/Model      | Notes                                                |
| ------------------- | ---------------------------------------------------- |
| 阿里云百炼         | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access the application at http://localhost:5300.

For detailed instructions, see the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options

*   **BaoTa Panel:** Available in BaoTa Panel, follow the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Community-contributed Zeabur template. [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:** [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** See the [manual deployment guide](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to receive the latest updates and announcements.

![star gif](https://docs.langbot.app/star.gif)

## Community Contributions

We appreciate the contributions from our [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and the broader community.

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>