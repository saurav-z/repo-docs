# LangBot: Your Open-Source IM Robot Development Platform (Powered by LLMs)

**Unleash the power of Large Language Models (LLMs) to create intelligent chatbots for various messaging platforms with LangBot – the open-source platform designed for easy and versatile IM robot development.**  Learn more at the [original LangBot repository](https://github.com/langbot-app/LangBot).

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

## Key Features:

*   **Versatile LLM Capabilities:** Supports diverse LLMs, enabling multi-turn conversations, tool utilization, and multimodal interactions.  Includes built-in RAG (Retrieval-Augmented Generation) capabilities, with strong integration with Dify.
*   **Multi-Platform Support:**  Works seamlessly with popular messaging platforms including QQ, QQ Channels, WeChat Work, Personal WeChat, Feishu, Discord, Telegram, and Slack.
*   **Robust and Feature-Rich:** Offers built-in access control, rate limiting, and profanity filtering.  Simple configuration and multiple deployment options are provided, along with support for multi-pipeline configurations to serve different bot applications.
*   **Extensible with Plugins & Community Driven:**  Features a plugin system based on event-driven and component extensions, including support for the Anthropic [MCP Protocol](https://modelcontextprotocol.io/), with hundreds of existing plugins available.
*   **Web-Based Management:** Manage your LangBot instances through a user-friendly web interface, eliminating the need for manual configuration file edits.

**Explore the live demo:** https://demo.langbot.dev/
*   Demo login:  email: `demo@langbot.app`, password: `langbot123456`

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access the bot at http://localhost:5300.

For detailed information, consult the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options:

*   **Baota Panel Deployment:** Available on the Baota Panel; follow the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html) for setup.
*   **Zeabur Cloud Deployment:**  Utilize the community-contributed Zeabur template: [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud Deployment:**  Deploy using Railway: [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:**  Run the release version directly; see the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to receive the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Supported Features

### Message Platforms

| Platform         | Status | Notes                                  |
| ---------------- | ------ | -------------------------------------- |
| QQ (Personal)    | ✅     | Private and group chats               |
| QQ Official Bot  | ✅     | Supports Channels, private, and group chats |
| WeChat Work      | ✅     |                                        |
| WeChat Work External Customer Service | ✅     |                                        |
| Personal WeChat  | ✅     |                                        |
| WeChat Official Account | ✅     |                                        |
| Feishu           | ✅     |                                        |
| DingTalk         | ✅     |                                        |
| Discord          | ✅     |                                        |
| Telegram         | ✅     |                                        |
| Slack            | ✅     |                                        |

### Large Language Model (LLM) Support

| Model                                                                                              | Status | Notes                                                                        |
| -------------------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------- |
| [OpenAI](https://platform.openai.com/)                                                              | ✅     | Compatible with any OpenAI API format model                                    |
| [DeepSeek](https://www.deepseek.com/)                                                              | ✅     |                                                                              |
| [Moonshot](https://www.moonshot.cn/)                                                              | ✅     |                                                                              |
| [Anthropic](https://www.anthropic.com/)                                                            | ✅     |                                                                              |
| [xAI](https://x.ai/)                                                            | ✅     |                                                                              |
| [ZhipuAI](https://open.bigmodel.cn/)                                                              | ✅     |                                                                              |
| [YouCloud Compute](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)                                | ✅     | LLM and GPU resource platform                                                  |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot)                | ✅     | LLM and GPU resource platform                                                  |
| [302.AI](https://share.302.ai/SuTG99)                                                              | ✅     | LLM Aggregation Platform                                                       |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅ | |
| [Dify](https://dify.ai)                                                                | ✅     | LLMOps Platform                                                             |
| [Ollama](https://ollama.com/)                                                                 | ✅     | Local LLM platform                                                           |
| [LMStudio](https://lmstudio.ai/)                                                               | ✅     | Local LLM platform                                                           |
| [GiteeAI](https://ai.gitee.com/)                                                               | ✅     | LLM Interface Aggregation Platform                                                |
| [SiliconFlow](https://siliconflow.cn/)                                                               | ✅     | LLM Aggregation Platform                                                        |
| [Alibaba Cloud Baichuan](https://bailian.console.aliyun.com/) | ✅     | LLM and LLMOps Platform |
| [Volcano Engine Ark](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅ | LLM and LLMOps Platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅ | LLM Aggregation Platform |
| [MCP](https://modelcontextprotocol.io/)                                                               | ✅     | Supports tool access through the MCP protocol                                  |

### Text-to-Speech (TTS)

| Platform/Model                         | Notes                                               |
| -------------------------------------- | --------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)        | [Plugin](https://github.com/the-lazy-me/NewChatVoice)         |
| [HaiTun AI](https://www.ttson.cn/?source=thelazy)    | [Plugin](https://github.com/the-lazy-me/NewChatVoice)         |
| [AzureTTS](https://portal.azure.com/) | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image (TTI)

| Platform/Model       | Notes                                                         |
| -------------------- | ------------------------------------------------------------- |
| Alibaba Cloud Baichuan | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

A big thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and all community members for their contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO optimizations:

*   **Clear Hook:** The one-sentence hook at the beginning immediately grabs attention and highlights the core value proposition.
*   **Keyword Optimization:**  The summary uses relevant keywords like "Large Language Models (LLMs)," "chatbots," "open-source," "IM robot development," and platform names.
*   **Structured Headings:**  Uses clear headings and subheadings to organize information, improving readability and SEO.
*   **Bulleted Lists:**  Employs bulleted lists for key features and platform/model support, making information easily scannable.
*   **Concise Language:** The text is more concise and direct.
*   **Internal Linking:** Includes links to key sections, documentation, and the demo environment.
*   **Call to Action:** Encourages users to star and watch the repository to stay updated.
*   **Community Appreciation:** Acknowledges and thanks community contributors.
*   **Comprehensive Platform/Model Lists:** Provides detailed tables with the status and notes for message platforms, LLMs, TTS, and TTI.
*   **Multiple Language Support:** Keeps language links, promoting broader reach.