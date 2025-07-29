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

# LangBot: The Open-Source LLM-Powered IM Robot Platform

**LangBot is an open-source platform designed to simplify the development of instant messaging robots, offering a seamless experience for integrating large language model (LLM) capabilities into your favorite IM platforms.**

## Key Features

*   **Versatile LLM Applications:**
    *   LLM Conversations and Agent capabilities: Supports various large language models.
    *   Adapts to group and private chats.
    *   Includes multi-turn conversations, tool usage, and multimodal capabilities.
    *   Built-in RAG (Retrieval-Augmented Generation) for knowledge base integration.
    *   Deep integration with [Dify](https://dify.ai).
*   **Cross-Platform Compatibility:** Supports multiple platforms:
    *   QQ (personal and official bots)
    *   QQ Channels
    *   WeChat (including public accounts and enterprise WeChat)
    *   Feishu
    *   Discord
    *   Telegram
    *   Slack
*   **Robust and Feature-Rich:**
    *   Built-in access control, rate limiting, and profanity filters.
    *   Easy configuration and multiple deployment options.
    *   Supports multi-pipeline configuration for different use cases.
*   **Extensible with Plugins:**
    *   Plugin system based on event-driven and component extension mechanisms.
    *   Adapts to the Anthropic [MCP Protocol](https://modelcontextprotocol.io/).
    *   Hundreds of available plugins.
*   **Web-Based Management:**
    *   Web UI for managing your LangBot instances.
    *   Eliminates the need for manual configuration file editing.

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access at http://localhost:5300.

For detailed information, refer to the [Docker Deployment Guide](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options

*   **Baota Panel:** Available on Baota Panel (refer to the [Baota Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html)).
*   **Zeabur Cloud:** Community-contributed Zeabur template ([Deploy on Zeabur](https://zeabur.com/zh-CN/templates/ZKTBDH)).
*   **Railway Cloud:** ([Deploy on Railway](https://railway.app/template/yRrAyL?referralCode=vogKPF)).
*   **Manual Deployment:** See the [Manual Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to get the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Message Platform Support

| Platform          | Status | Notes                                    |
| ----------------- | ------ | ---------------------------------------- |
| QQ (Personal)     | ✅     | Private and group chats                  |
| QQ (Official Bot) | ✅     | Supports channels, private, and group chat |
| WeChat            | ✅     |                                          |
| Enterprise WeChat | ✅     |                                          |
| WeChat Official Account | ✅     |                                          |
| Feishu            | ✅     |                                          |
| DingTalk          | ✅     |                                          |
| Discord           | ✅     |                                          |
| Telegram          | ✅     |                                          |
| Slack             | ✅     |                                          |

## Supported LLMs

| Model                     | Status | Notes                                                                       |
| ------------------------- | ------ | --------------------------------------------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Supports any OpenAI API-compatible models                          |
| [DeepSeek](https://www.deepseek.com/)  | ✅     |                                                                             |
| [Moonshot](https://www.moonshot.cn/)  | ✅     |                                                                             |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                                                             |
| [xAI](https://x.ai/) | ✅     |                                                                             |
| [智谱AI](https://open.bigmodel.cn/) | ✅     |                                                                             |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)  | ✅     | LLM and GPU resource platform                                          |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot)  | ✅     | LLM and GPU resource platform                                          |
| [302.AI](https://share.302.ai/SuTG99)  | ✅     | LLM aggregation platform                                                                             |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat)    | ✅     |                                                                             |
| [Dify](https://dify.ai)        | ✅     | LLMOps platform                                                             |
| [Ollama](https://ollama.com/)       | ✅     | Local LLM platform                                                          |
| [LMStudio](https://lmstudio.ai/)       | ✅     | Local LLM platform                                                          |
| [GiteeAI](https://ai.gitee.com/)       | ✅     | LLM interface aggregation platform                                                          |
| [SiliconFlow](https://siliconflow.cn/)       | ✅     | LLM interface aggregation platform                                                          |
| [阿里云百炼](https://bailian.console.aliyun.com/)       | ✅     | LLM aggregation platform, LLMOps platform                                                         |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | LLM aggregation platform, LLMOps platform                                                        |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅     | LLM aggregation platform                                                       |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports tool use via the MCP protocol                                  |

## TTS Support

| Platform/Model                  | Notes                       |
| ------------------------------- | --------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)    | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)    | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image Support

| Platform/Model           | Notes                                             |
| ------------------------- | ------------------------------------------------- |
| 阿里云百炼 | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

Thanks to all the [contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>

[**Get Started with LangBot!**](https://github.com/langbot-app/LangBot)