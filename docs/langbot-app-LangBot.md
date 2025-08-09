# LangBot: Open-Source LLM-Powered IM Robot Development Platform

**Unleash the power of Large Language Models in your favorite messaging platforms with LangBot, a versatile and open-source IM robot development platform.**  [Explore the original repository](https://github.com/langbot-app/LangBot)

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

## Key Features

*   **Versatile LLM Integration:** Supports a wide range of large language models, including OpenAI, DeepSeek, Anthropic, and many more.
*   **Cross-Platform Compatibility:** Works seamlessly with popular messaging platforms such as QQ, QQ Channels, WeChat, Enterprise WeChat, Feishu, Discord, Telegram, and more.
*   **Agent & RAG Capabilities:** Includes Agent functionality, Retrieval-Augmented Generation (RAG) for enhanced knowledge retrieval, and Model Context Protocol (MCP) compatibility.
*   **Extensible Plugin Architecture:**  Offers a robust plugin system based on event-driven and component-based architecture, supporting custom extensions and integrations.
*   **Web-Based Management:** Provides a user-friendly web interface for easy configuration and management of your LangBot instances, eliminating the need for manual configuration file edits.
*   **Multiple Deployment Options:** Supports Docker Compose, Baota Panel, Zeabur Cloud, Railway Cloud, and manual deployment methods.

## Getting Started

### Quick Deployment (Docker Compose)

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```
Access the web interface at `http://localhost:5300`.

### Other Deployment Options:
*   **Baota Panel:**  Available on the Baota Panel, follow the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html) for setup.
*   **Zeabur Cloud:**  Deploy using the community-contributed [Zeabur template](https://zeabur.com/zh-CN/templates/ZKTBDH).
*   **Railway Cloud:**  Deploy via the [Railway template](https://railway.app/template/yRrAyL?referralCode=vogKPF).
*   **Manual Deployment:** Refer to the [manual deployment guide](https://docs.langbot.app/zh/deploy/langbot/manual.html) for detailed instructions.

## Stay Updated

Star and Watch the repository to receive the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Features in Detail

-   💬 **Advanced Conversational AI:**  Supports multi-turn conversations, tool use, and multimodal capabilities for enriched interactions. Includes built-in RAG for knowledge retrieval and deep integration with Dify.ai.
-   🤖 **Broad Platform Support:** Compatible with QQ, QQ Channels, Enterprise WeChat, WeChat Official Accounts, Feishu, DingTalk, Discord, Telegram, and Slack.
-   🛠️ **Robust and Feature-Rich:**  Offers access controls, rate limiting, and sensitive word filtering. Supports multiple pipeline configurations for different use cases.
-   🧩 **Plugin Ecosystem and Active Community:** Supports event-driven and component-based plugins, compatible with Anthropic [MCP](https://modelcontextprotocol.io/).  Hundreds of plugins available.
-   😻 **Web-Based Management Panel:** Manage LangBot instances via a browser-based UI.

For detailed specifications, consult the [Features documentation](https://docs.langbot.app/zh/insight/features.html).

## Demo
Explore a demo environment at: https://demo.langbot.dev/
-   Login: demo@langbot.app, password: langbot123456
    -   **Note:** This is a public demo, so please do not enter any sensitive information.

## Supported Platforms

| Platform         | Status | Notes                                 |
| :--------------- | :----- | :------------------------------------ |
| QQ Personal      | ✅     | QQ personal private and group chats  |
| QQ Official Bot  | ✅     | QQ official bot, supports channels, private and group chats |
| Enterprise WeChat | ✅     |                                       |
| WeChat for business     | ✅     |                                       |
| WeChat Official Account    | ✅     |                                       |
| Feishu           | ✅     |                                       |
| DingTalk         | ✅     |                                       |
| Discord          | ✅     |                                       |
| Telegram         | ✅     |                                       |
| Slack            | ✅     |                                       |

## Supported LLMs
| Model                                  | Status | Notes                                                     |
| :------------------------------------- | :----- | :-------------------------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Supports any OpenAI API-compatible models                 |
| [DeepSeek](https://www.deepseek.com/) | ✅     |                                                           |
| [Moonshot](https://www.moonshot.cn/)  | ✅     |                                                           |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                                           |
| [xAI](https://x.ai/) | ✅     |                                                                                              |
| [智谱AI](https://open.bigmodel.cn/) | ✅     |                                                           |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     | 大模型和 GPU 资源平台                                           |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | 大模型和 GPU 资源平台                                           |
| [302.AI](https://share.302.ai/SuTG99) | ✅     | 大模型聚合平台                                                              |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat)   | ✅     |                                                           |
| [Dify](https://dify.ai)                | ✅     | LLMOps Platform                                          |
| [Ollama](https://ollama.com/)           | ✅     | Local LLM platform                                      |
| [LMStudio](https://lmstudio.ai/)       | ✅     | Local LLM platform                                      |
| [GiteeAI](https://ai.gitee.com/)       | ✅     | LLM API Aggregation platform                               |
| [SiliconFlow](https://siliconflow.cn/) | ✅     | LLM API Aggregation platform                               |
| [阿里云百炼](https://bailian.console.aliyun.com/)      | ✅     | LLM API Aggregation and LLMOps Platform                           |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW)       | ✅     | LLM API Aggregation and LLMOps Platform                              |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro)      | ✅     | LLM API Aggregation Platform                           |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports tool retrieval via MCP protocol                 |

## Text-to-Speech (TTS) Integrations
| Platform/Model                   | Notes                                                   |
| :------------------------------- | :------------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)       | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image (TTI) Integrations
| Platform/Model              | Notes                                                        |
| :-------------------------- | :----------------------------------------------------------- |
| 阿里云百炼 | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin)

## Community Contributions

A huge thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and all community members for their invaluable contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>