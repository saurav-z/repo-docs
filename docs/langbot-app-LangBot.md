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

# LangBot: Build Your Own AI-Powered Chatbot Platform

**LangBot is an open-source platform that simplifies building and deploying AI-powered chatbots across multiple messaging platforms.**  This README provides a concise overview, highlights key features, and guides you to get started. For more details, visit the [LangBot GitHub Repository](https://github.com/langbot-app/LangBot).

## Key Features

*   **Multi-Platform Support:** Connect with users across popular messaging apps, including QQ, Discord, Telegram, and more.
*   **AI-Powered Chatbot Capabilities:** Leverage Large Language Models (LLMs) for engaging conversations, including multi-turn dialog, tool usage, and multimodal input/output.
*   **Extensive Model Support:** Compatible with a wide range of LLMs, including OpenAI, Google Gemini, DeepSeek, and others.
*   **Robust Features:** Includes built-in access control, rate limiting, and sensitive word filtering for enhanced security and moderation.
*   **Plugin Ecosystem:** Extend functionality with a growing library of plugins, including support for Anthropic's MCP protocol.
*   **Web Management Panel:** Easily configure and manage your LangBot instance through a user-friendly web interface.

## Getting Started

### Docker Compose Deployment

1.  Clone the repository:
    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    ```
2.  Run using Docker Compose:
    ```bash
    docker compose up -d
    ```
3.  Access the chatbot at http://localhost:5300.

   Refer to the [Docker Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html) for detailed instructions.

### Alternative Deployment Options

*   **Baota Panel:** Deploy with ease via Baota Panel. Check out the [Baota Panel Documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur:** Deploy using a community-contributed Zeabur template.  [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway:** Deploy on Railway.  [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** For other deployment options, see the [Manual Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Keep up-to-date with the latest developments by starring and watching the repository.

![star gif](https://docs.langbot.app/star.gif)

## Features in Detail

*   **💬 LLM Chat and Agent Capabilities:** Supports various LLMs, and adapts to group and private chats; features multi-turn conversations, tool calls, and multimodal capabilities, along with built-in RAG (knowledge base) and deep integration with [Dify](https://dify.ai).
*   **🤖 Cross-Platform Support:** Currently supports QQ, QQ Channels, WeChat Enterprise, WeChat Personal, Feishu, Discord, Telegram, Slack, and more.
*   **🛠️ High Stability and Comprehensive Features:** Native support for access control, rate limiting, and sensitive word filtering mechanisms; simple configuration, supports various deployment methods. Supports multi-pipeline configuration for different chatbot use cases.
*   **🧩 Plugin Extensions and Active Community:** Supports plugin mechanisms driven by events and component extensions; complies with Anthropic's [MCP protocol](https://modelcontextprotocol.io/); currently, there are hundreds of plugins.
*   **😻 Web Management Panel:** Supports managing LangBot instances through a browser, eliminating the need to manually write configuration files.

Detailed specifications and features can be found in the [Documentation](https://docs.langbot.app/zh/insight/features.html).

Alternatively, explore the demo environment: https://demo.langbot.dev/
    -   Login Information: Email: `demo@langbot.app` Password: `langbot123456`
    -   Note: Only showcases WebUI effects; please do not enter sensitive information in this public environment.

### Messaging Platforms

| Platform            | Status | Notes                  |
| ------------------- | ------ | ---------------------- |
| QQ Personal Account | ✅     | Private and Group Chats |
| QQ Official Bot     | ✅     | Channels, Private, Group |
| WeChat Enterprise   | ✅     |                        |
| WeChat Customer Service | ✅     |                        |
| WeChat Personal     | ✅     |                        |
| WeChat Official Account | ✅     |                        |
| Feishu              | ✅     |                        |
| DingTalk            | ✅     |                        |
| Discord             | ✅     |                        |
| Telegram            | ✅     |                        |
| Slack               | ✅     |                        |

### Large Language Model (LLM) Support

| Model                      | Status | Notes                      |
| -------------------------- | ------ | -------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Any OpenAI API model  |
| [DeepSeek](https://www.deepseek.com/) | ✅     | |
| [Moonshot](https://www.moonshot.cn/) | ✅     | |
| [Anthropic](https://www.anthropic.com/) | ✅     | |
| [xAI](https://x.ai/) | ✅     | |
| [智谱AI](https://open.bigmodel.cn/) | ✅     | |
| [胜算云](https://www.shengsuanyun.com/?from=CH_KYIPP758) | ✅     | Global models available |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     | LLM and GPU platform |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | LLM and GPU platform |
| [302.AI](https://share.302.ai/SuTG99) | ✅     | LLM Aggregation Platform |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     | |
| [Dify](https://dify.ai) | ✅     | LLMOps Platform |
| [Ollama](https://ollama.com/) | ✅     | Local LLM Platform |
| [LMStudio](https://lmstudio.ai/) | ✅     | Local LLM Platform |
| [GiteeAI](https://ai.gitee.com/) | ✅     | LLM API Aggregation Platform |
| [SiliconFlow](https://siliconflow.cn/) | ✅     | LLM Aggregation Platform |
| [阿里云百炼](https://bailian.console.aliyun.com/) | ✅     | LLM Aggregation and LLMOps Platform |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | LLM Aggregation and LLMOps Platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅     | LLM Aggregation Platform |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports tool access via MCP protocol |

### Text-to-Speech (TTS)

| Platform/Model            | Notes                                       |
| -------------------------- | ------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/) | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image (TTI)

| Platform/Model            | Notes                                           |
| -------------------------- | ----------------------------------------------- |
| 阿里云百炼 | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## 🤝 Community Contributions

A big thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and all community members for their valuable contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>