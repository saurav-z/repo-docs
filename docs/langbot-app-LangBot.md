# LangBot: The Open-Source IM Robot Development Platform

**[LangBot](https://github.com/langbot-app/LangBot) empowers you to build intelligent, multi-platform chatbots with ease.**

<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot"/>
</a>

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

</p>


## Key Features

*   **Versatile LLM Integration:** Support for a wide range of Large Language Models (LLMs) including OpenAI, DeepSeek, Moonshot, Anthropic, xAI, Google Gemini, and more.
*   **Multi-Platform Support:** Compatible with popular messaging platforms such as QQ, WeChat, Enterprise WeChat, Feishu, Discord, Telegram, and Slack.
*   **Agent & RAG Capabilities:**  Offers Agent functionality, Retrieval-Augmented Generation (RAG) for knowledge base integration, and Model Context Protocol (MCP) support.
*   **Extensible Architecture:** Provides a plugin system for easy customization and expansion, including support for Anthropic's MCP protocol, with hundreds of available plugins.
*   **Web-Based Management:**  Includes a web UI for easy management and configuration of your LangBot instances.
*   **High Stability and Functionality:** Built-in features for access control, rate limiting, and content filtering. Multiple deployment options.

For detailed features, see the [documentation](https://docs.langbot.app/zh/insight/features.html).

## Getting Started

### Deploy with Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access the running instance at http://localhost:5300.

See the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html) for more information.

### Other Deployment Options

*   **Baota Panel Deployment:** If you have Baota Panel installed, follow the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html) for one-click deployment.
*   **Zeabur Cloud Deployment:** Deploy using a Zeabur template: [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud Deployment:** Deploy using a Railway template: [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Refer to the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and Watch this repository to receive the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Demo

Explore the web UI at https://demo.langbot.dev/
  - Login: Email: `demo@langbot.app` Password: `langbot123456`
  - Please do not enter sensitive information in the demo.

## Supported Platforms

| Platform           | Status | Notes                                 |
| ------------------ | ------ | ------------------------------------- |
| QQ (Personal)      | ✅      | Private and group chats                |
| QQ (Official Bot)  | ✅      | Supports channels, private and group chats |
| WeChat             | ✅      |                                       |
| Enterprise WeChat  | ✅      |                                       |
| WeChat Official Account | ✅      |                                       |
| Feishu             | ✅      |                                       |
| DingTalk           | ✅      |                                       |
| Discord            | ✅      |                                       |
| Telegram           | ✅      |                                       |
| Slack            | ✅      |                                       |

## Supported LLMs

| Model                                  | Status | Notes                                                      |
| -------------------------------------- | ------ | ---------------------------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅      | Supports any OpenAI API-compatible models                 |
| [DeepSeek](https://www.deepseek.com/)  | ✅      |                                                            |
| [Moonshot](https://www.moonshot.cn/)  | ✅      |                                                            |
| [Anthropic](https://www.anthropic.com/) | ✅      |                                                            |
| [xAI](https://x.ai/)                  | ✅      |                                                            |
| [智谱AI](https://open.bigmodel.cn/)   | ✅      |                                                            |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅      | LLM and GPU resource platform                          |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅      | LLM and GPU resource platform                          |
| [302.AI](https://share.302.ai/SuTG99) | ✅      | LLM Aggregation Platform                               |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅      |                                                            |
| [Dify](https://dify.ai)                | ✅      | LLMOps platform                                           |
| [Ollama](https://ollama.com/)           | ✅      | Local LLM platform                                        |
| [LMStudio](https://lmstudio.ai/)        | ✅      | Local LLM platform                                        |
| [GiteeAI](https://ai.gitee.com/)     | ✅      | LLM Interface Aggregation Platform                      |
| [SiliconFlow](https://siliconflow.cn/) | ✅      | LLM Aggregation Platform                               |
| [阿里云百炼](https://bailian.console.aliyun.com/)   | ✅      | LLM Aggregation Platform, LLMOps platform |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅      | LLM Aggregation Platform, LLMOps platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅      | LLM Aggregation Platform |
| [MCP](https://modelcontextprotocol.io/) | ✅      | Supports tool access via the MCP protocol             |

## TTS (Text-to-Speech) Integrations

| Platform/Model                      | Notes                     |
| ----------------------------------- | ------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)   | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)  | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image Integrations

| Platform/Model                      | Notes                     |
| ----------------------------------- | ------------------------- |
| 阿里云百炼                     | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

We are grateful to the [contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and the community for their valuable contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>