# LangBot: Your Open-Source AI Chatbot Development Platform

LangBot empowers you to build powerful, AI-driven chatbots with ease, supporting various LLM applications and integrations across multiple platforms.  [Visit the original repository](https://github.com/langbot-app/LangBot) for more information.

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

*   **Versatile LLM Applications:**  LangBot supports Agent, RAG (Retrieval-Augmented Generation), and MCP (Model Context Protocol) functionalities for diverse LLM applications.
*   **Multi-Platform Compatibility:** Works seamlessly with popular messaging platforms like QQ, QQ Channels, WeChat Work, WeChat, Feishu, Discord, and Telegram.
*   **Robust & Customizable:**  Offers built-in access control, rate limiting, and profanity filters, along with flexible configuration options and support for various deployment methods.
*   **Extensible Plugin Ecosystem:**  Leverages an event-driven and component-based plugin architecture, including support for Anthropic's MCP protocol with hundreds of available plugins.
*   **Web-Based Management:** Includes a user-friendly web interface for easy management of your LangBot instances, eliminating the need for manual configuration file editing.

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access the application at http://localhost:5300.

For detailed instructions, see the [Docker Deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Alternative Deployment Options

*   **BaoTa Panel:** Available on BaoTa Panel - refer to the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Deploy with a community-contributed Zeabur template: [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:**  Deploy with Railway: [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:**  Run directly from the release version - see the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to receive the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Features Overview

*   **Large Language Model (LLM) Chat & Agent:** Supports multiple LLMs, designed for group and individual chats.  Features multi-turn conversations, tool usage, multimodal capabilities, and a built-in RAG system with integration with [Dify](https://dify.ai).
*   **Broad Platform Support:**  Currently supports QQ (personal and official), WeChat Work, WeChat, Feishu, Discord, Telegram, and more.
*   **High Stability & Feature-Rich:** Provides essential features like access control, rate limiting, and profanity filtering.  Offers easy configuration and multiple deployment options with support for multiple pipeline configurations to support different use cases.
*   **Plugin-Driven Extensibility & Active Community:**  Supports event-driven and component-based plugins with  Anthropic's [MCP Protocol](https://modelcontextprotocol.io/).
*   **Web Management Panel:**  Manage LangBot instances through a web browser, removing the need for manual configuration editing.

Explore detailed specifications in the [documentation](https://docs.langbot.app/zh/insight/features.html).

You can also explore a demo environment at: https://demo.langbot.dev/
  - Login:  Email: `demo@langbot.app`, Password: `langbot123456`
  - Note: This is a public demo; please avoid entering any sensitive information.

### Messaging Platform Support

| Platform           | Status | Notes                               |
| ------------------ | ------ | ----------------------------------- |
| QQ (Personal)      | ✅     | Private and Group Chat              |
| QQ (Official Bot)  | ✅     | Supports Channels, private and group chat |
| WeChat Work        | ✅     |                                     |
| WeChat Work (External) | ✅ |                                     |
| WeChat             | ✅     |                                     |
| WeChat Official Account | ✅     |                                     |
| Feishu             | ✅     |                                     |
| DingTalk           | ✅     |                                     |
| Discord            | ✅     |                                     |
| Telegram           | ✅     |                                     |
| Slack              | ✅     |                                     |

### Large Language Model (LLM) Support

| Model                                                                                                                                                   | Status | Notes                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------- |
| [OpenAI](https://platform.openai.com/)                                                                                                                | ✅     | Compatible with any OpenAI API format model            |
| [DeepSeek](https://www.deepseek.com/)                                                                                                                 | ✅     |                                                          |
| [Moonshot](https://www.moonshot.cn/)                                                                                                                | ✅     |                                                          |
| [Anthropic](https://www.anthropic.com/)                                                                                                                | ✅     |                                                          |
| [xAI](https://x.ai/)                                                                                                                                   | ✅     |                                                          |
| [智谱AI](https://open.bigmodel.cn/)                                                                                                                 | ✅     |                                                          |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)                                                                                          | ✅     | LLMs and GPU resource platform                          |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot)                                                                  | ✅     | LLMs and GPU resource platform                          |
| [302.AI](https://share.302.ai/SuTG99)                                                                                                                 | ✅     | LLM Aggregation Platform                               |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat)                                                                                          | ✅     |                                                          |
| [Dify](https://dify.ai)                                                                                                                                   | ✅     | LLMOps Platform                                          |
| [Ollama](https://ollama.com/)                                                                                                                            | ✅     | Local LLM Platform                                     |
| [LMStudio](https://lmstudio.ai/)                                                                                                                           | ✅     | Local LLM Platform                                     |
| [GiteeAI](https://ai.gitee.com/)                                                                                                                         | ✅     | LLM API Aggregation Platform                           |
| [SiliconFlow](https://siliconflow.cn/)                                                                                                                    | ✅     | LLM Aggregation Platform                               |
| [阿里云百炼](https://bailian.console.aliyun.com/)                                                                                                               | ✅     | LLM Aggregation Platform, LLMOps Platform                |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | LLM Aggregation Platform, LLMOps Platform                |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅     | LLM Aggregation Platform |
| [MCP](https://modelcontextprotocol.io/) | ✅     |                                                    |

### Text-to-Speech (TTS)

| Platform/Model                             | Notes                                                      |
| ------------------------------------------ | ---------------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice)           |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice)           |
| [AzureTTS](https://portal.azure.com/)      | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)           |

### Text-to-Image (TTI)

| Platform/Model             | Notes                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 阿里云百炼                  | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

Thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members for their contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>