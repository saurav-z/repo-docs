# LangBot: The Open-Source LLM-Powered IM Bot Platform

**LangBot** is an open-source platform designed to simplify the creation of large language model (LLM)-powered instant messaging (IM) bots, offering a seamless, out-of-the-box experience.  ([Visit the original repo](https://github.com/langbot-app/LangBot))

<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot"/>
</a>

<div align="center">

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

</p>

## Key Features

*   **Versatile LLM Applications:** Supports Agent, RAG (Retrieval-Augmented Generation), and MCP (Model Context Protocol) functionalities, offering diverse LLM applications.
*   **Cross-Platform Compatibility:** Works seamlessly with popular messaging platforms, including QQ, QQ Channels, WeChat, Enterprise WeChat, Feishu, Discord, Telegram, Slack and more.
*   **Robust and Feature-Rich:** Includes built-in access control, rate limiting, and profanity filtering. Supports multiple deployment options and allows for multi-pipeline configuration for various use cases.
*   **Extensible Plugin Architecture:** Features a plugin system with event-driven and component-based extension capabilities.  Adapts to the Anthropic MCP protocol and has hundreds of available plugins.
*   **User-Friendly Web Interface:** Provides a web management panel, allowing you to configure and manage your LangBot instances through a browser, eliminating the need for manual configuration file editing.
*   **Model Agnostic**: LangBot works with many different LLMs, including OpenAI, DeepSeek, Moonshot, Anthropic, xAI, and others.

  For a detailed feature list, visit the [documentation](https://docs.langbot.app/zh/insight/features.html).
  Or, check out the demo environment: https://demo.langbot.dev/
    *   Login Information: Email: `demo@langbot.app` Password: `langbot123456`
    *   Note: This is a public environment. Please refrain from entering sensitive information.

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access the bot at http://localhost:5300.

For detailed instructions, see the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options

*   **BaoTa Panel:** Available on the BaoTa Panel. Instructions can be found [here](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Community-contributed Zeabur template. [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:** [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Run the release version. See the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to receive the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Messaging Platform Support

| Platform          | Status | Notes                                                                                             |
| ----------------- | ------ | ------------------------------------------------------------------------------------------------- |
| QQ (Personal)     | ✅     | Private and group chats                                                                             |
| QQ (Official Bot) | ✅     | Supports Channels, private and group chats                                                          |
| WeChat            | ✅     |                                                                                                   |
| Enterprise WeChat | ✅     |                                                                                                   |
| WeChat Official Account   | ✅     |                                                                                                   |
| Feishu            | ✅     |                                                                                                   |
| DingTalk          | ✅     |                                                                                                   |
| Discord           | ✅     |                                                                                                   |
| Telegram          | ✅     |                                                                                                   |
| Slack             | ✅     |                                                                                                   |

## Large Language Model Support

| Model                                    | Status | Notes                                                            |
| ---------------------------------------- | ------ | ---------------------------------------------------------------- |
| [OpenAI](https://platform.openai.com/)   | ✅     | Compatible with any OpenAI API-compatible models.                |
| [DeepSeek](https://www.deepseek.com/)    | ✅     |                                                                  |
| [Moonshot](https://www.moonshot.cn/)    | ✅     |                                                                  |
| [Anthropic](https://www.anthropic.com/)  | ✅     |                                                                  |
| [xAI](https://x.ai/)    | ✅     |                                                                  |
| [智谱AI](https://open.bigmodel.cn/)   | ✅     |                                                                  |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     | Large model and GPU resource platform                                                |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | Large model and GPU resource platform                                                |
| [302.AI](https://share.302.ai/SuTG99)  | ✅     | Large model aggregation platform                                                  |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat)  | ✅     |                                              |
| [Dify](https://dify.ai)  | ✅     | LLMOps platform                                               |
| [Ollama](https://ollama.com/)  | ✅     | Local large model running platform                                               |
| [LMStudio](https://lmstudio.ai/)  | ✅     | Local large model running platform                                               |
| [GiteeAI](https://ai.gitee.com/)  | ✅     | Large model interface aggregation platform                                               |
| [SiliconFlow](https://siliconflow.cn/)  | ✅     | Large model aggregation platform                                               |
| [阿里云百炼](https://bailian.console.aliyun.com/)  | ✅     | Large model aggregation platform, LLMOps platform                                            |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW)  | ✅     | Large model aggregation platform, LLMOps platform                                           |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro)  | ✅     | Large model aggregation platform                                           |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports obtaining tools via the MCP protocol.                                      |

## TTS Support

| Platform/Model              | Notes                                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)      | [Plugin](https://github.com/the-lazy-me/NewChatVoice)                                              |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)        | [Plugin](https://github.com/the-lazy-me/NewChatVoice)                                              |
| [AzureTTS](https://portal.azure.com/) | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)                                              |

## Text-to-Image Support

| Platform/Model    | Notes                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------ |
| 阿里云百炼          | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

We appreciate the contributions from our [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and the wider community!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>