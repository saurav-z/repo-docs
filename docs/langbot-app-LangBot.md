<div align="center">
  <a href="https://langbot.app">
    <img src="https://docs.langbot.app/social_zh.png" alt="LangBot" width="300"/>
  </a>
</div>

# LangBot: Open-Source LLM-Powered IM Bot Development Platform

**Supercharge your instant messaging with AI!** LangBot is a powerful, open-source platform designed to help you build and deploy intelligent IM bots with ease.  Access the original repo [here](https://github.com/langbot-app/LangBot).

<div align="center">
  <a href="https://hellogithub.com/repository/langbot-app/LangBot" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=5ce8ae2aa4f74316bf393b57b952433c&claim_uid=gtmc6YWjMZkT21R" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

  [English](README_EN.md) / 简体中文 / [繁體中文](README_TW.md) / [日本語](README_JP.md) / (PR for your language)

  [![Discord](https://img.shields.io/discord/1335141740050649118?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb)](https://discord.gg/wdNEHETs87)
  [![QQ Group](https://img.shields.io/badge/%E7%A4%BE%E5%8C%BAQQ%E7%BE%A4-966235608-blue)](https://qm.qq.com/q/JLi38whHum)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/langbot-app/LangBot)
  [![GitHub release (latest by date)](https://img.shields.io/github/v/release/langbot-app/LangBot)](https://github.com/langbot-app/LangBot/releases/latest)
  <img src="https://img.shields.io/badge/python-3.10 ~ 3.13 -blue.svg" alt="python">
  [![star](https://gitcode.com/RockChinQ/LangBot/star/badge.svg)](https://gitcode.com/RockChinQ/LangBot)

  <a href="https://langbot.app">Project Homepage</a> |
  <a href="https://docs.langbot.app/zh/insight/guide.html">Deployment Docs</a> |
  <a href="https://docs.langbot.app/zh/plugin/plugin-intro.html">Plugins Introduction</a> |
  <a href="https://github.com/langbot-app/LangBot/issues/new?assignees=&labels=%E7%8B%AC%E7%AB%8B%E6%8F%92%E4%BB%B6&projects=&template=submit-plugin.yml&title=%5BPlugin%5D%3A+%E8%AF%B7%E6%B1%82%E7%99%BB%E8%AE%B0%E6%96%B0%E6%8F%92%E4%BB%B6">Submit Plugin</a>
</div>

## Key Features

*   **AI-Powered Conversations & Agents:** Leverage the power of large language models for engaging interactions.  Supports multi-turn conversations, tool use, and multimodal capabilities. Includes built-in RAG (Retrieval-Augmented Generation) and deep integration with [Dify](https://dify.ai).
*   **Multi-Platform Support:** Deploy your bot on popular messaging platforms.  Current supported platforms include QQ, QQ Channel, WeChat Work, Personal WeChat, Feishu, Discord, Telegram, and more!
*   **Robust & Feature-Rich:** Built for stability with features like access control, rate limiting, and profanity filtering. Easy configuration and multiple deployment options. Supports multi-pipeline configuration for diverse bot applications.
*   **Extensible with Plugins & a Vibrant Community:**  Enhance your bot's functionality with a modular plugin system, including event-driven and component extensions.  Supports the Anthropic [MCP Protocol](https://modelcontextprotocol.io/). Hundreds of plugins are already available!
*   **Web-Based Management:**  Manage your LangBot instance through a convenient web interface, eliminating the need for manual configuration file editing.

For detailed specifications, visit the [documentation](https://docs.langbot.app/zh/insight/features.html).

You can explore the demo environment: https://demo.langbot.dev/
*   Login: Email: `demo@langbot.app`, Password: `langbot123456`
*   Note: This is a public environment, so please do not enter any sensitive information.

## Quick Start: Deployment Options

### Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access at http://localhost:5300.
Detailed documentation: [Docker Deployment](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Baota Panel

Available on Baota Panel. Follow instructions in the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).

### Zeabur Cloud

Community contributed Zeabur template.
[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)

### Railway Cloud

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)

### Manual Deployment

Use the release versions. See documentation: [Manual Deployment](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and Watch the repository to receive the latest updates!

![star gif](https://docs.langbot.app/star.gif)

## Supported Messaging Platforms

| Platform         | Status | Notes                         |
| ---------------- | ------ | ----------------------------- |
| QQ Personal      | ✅     | Private and group chats      |
| QQ Official Bot  | ✅     | Supports channels, private chats, and group chats |
| WeChat Work      | ✅     |                               |
| WeChat Customer Service  | ✅     |                               |
| WeChat Official Account | ✅     |                               |
| Feishu           | ✅     |                               |
| DingTalk         | ✅     |                               |
| Discord          | ✅     |                               |
| Telegram         | ✅     |                               |
| Slack            | ✅     |                               |

## Supported Large Language Models

| Model                             | Status | Notes                                  |
| --------------------------------- | ------ | -------------------------------------- |
| [OpenAI](https://platform.openai.com/)  | ✅     | Supports any OpenAI API format model   |
| [DeepSeek](https://www.deepseek.com/)    | ✅     |                                        |
| [Moonshot](https://www.moonshot.cn/)   | ✅     |                                        |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                        |
| [xAI](https://x.ai/)                 | ✅     |                                        |
| [智谱AI](https://open.bigmodel.cn/)   | ✅     |                                        |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)  | ✅     |  Large model and GPU resource platform |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot)    | ✅     | Large model and GPU resource platform  |
| [302.AI](https://share.302.ai/SuTG99)  | ✅     | Large model aggregation platform       |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     |                                        |
| [Dify](https://dify.ai)              | ✅     | LLMOps Platform                        |
| [Ollama](https://ollama.com/)            | ✅     | Local large model platform             |
| [LMStudio](https://lmstudio.ai/)           | ✅     | Local large model platform             |
| [GiteeAI](https://ai.gitee.com/)       | ✅     | Large model API aggregation platform   |
| [SiliconFlow](https://siliconflow.cn/)     | ✅     | Large model aggregation platform       |
| [阿里云百炼](https://bailian.console.aliyun.com/)    | ✅     | Large model aggregation, LLMOps        |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW)  | ✅     | Large model aggregation, LLMOps        |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro)  | ✅     | Large model aggregation                 |
| [MCP](https://modelcontextprotocol.io/)    | ✅     | Supports tool retrieval via MCP     |

## Text-to-Speech (TTS)

| Platform/Model                       | Notes                                             |
| ----------------------------------- | ------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)       | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image

| Platform/Model              | Notes                                 |
| --------------------------- | ------------------------------------- |
| 阿里云百炼               | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

Thank you to all [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and other community members for their contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>