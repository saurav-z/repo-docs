# LangBot: Your Open-Source AI Chatbot Development Platform

**Unlock the power of AI with LangBot, a versatile open-source platform designed for building feature-rich IM chatbots with ease.** Explore the LangBot repository on [GitHub](https://github.com/langbot-app/LangBot).

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

*   **Versatile AI Capabilities:** Leverage Agent, RAG (Retrieval-Augmented Generation), and MCP (Model Context Protocol) functionalities for advanced LLM applications.
*   **Multi-Platform Support:** Seamlessly integrate with a wide range of popular messaging platforms, including QQ, QQ Channel, WeChat Enterprise, WeChat, Feishu, Discord, Telegram, and more.
*   **Robust and Feature-Rich:** Benefit from built-in access control, rate limiting, and profanity filtering; easily configurable with multiple deployment options. Implement different pipelines for diverse chatbot scenarios.
*   **Extensible with Plugins & a Thriving Community:** Extend functionality through event-driven and component-based plugin architecture; supports the Anthropic MCP protocol; explore hundreds of available plugins.
*   **User-Friendly Web Interface:** Manage your LangBot instances directly through a web UI, eliminating the need for manual configuration.

## Get Started

### Quick Deployment with Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access LangBot at `http://localhost:5300`.

For detailed deployment instructions, refer to the [Docker Deployment](https://docs.langbot.app/zh/deploy/langbot/docker.html) documentation.

### Alternative Deployment Options

*   **BaoTa Panel Deployment:**  Available on the BaoTa panel for easy one-click installation (see [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html)).
*   **Zeabur Cloud Deployment:** Deploy using a community-contributed Zeabur template ( [Deploy on Zeabur](https://zeabur.com/zh-CN/templates/ZKTBDH)).
*   **Railway Cloud Deployment:** Deploy on Railway ( [Deploy on Railway](https://railway.app/template/yRrAyL?referralCode=vogKPF)).
*   **Manual Deployment:** Run from the release binaries. See [Manual Deployment](https://docs.langbot.app/zh/deploy/langbot/manual.html) for instructions.

## Stay Updated

Star and watch the repository to receive the latest updates!

![star gif](https://docs.langbot.app/star.gif)

## Key Features in Detail

*   **AI Chat and Agents:** Supports various large language models (LLMs), works with both group and private chats. Features multi-turn conversations, tool usage, multimodal capabilities, integrated RAG (knowledge base) implementation, and in-depth integration with [Dify](https://dify.ai).
*   **Web Management Panel:** Manage LangBot instances through a browser-based interface, simplifying configuration.

For detailed specifications, visit the [Features Documentation](https://docs.langbot.app/zh/insight/features.html).

Or, try the demo environment: [https://demo.langbot.dev/](https://demo.langbot.dev/)
    *   Login: Email: `demo@langbot.app`, Password: `langbot123456`
    *   Note: This is a public demo; please refrain from entering any sensitive information.

### Messaging Platform Support

| Platform         | Status | Notes                    |
| ---------------- | ------ | ------------------------ |
| QQ (Personal)    | ✅     | Private & Group Chats    |
| QQ (Official Bot) | ✅     | Channels, Private & Group Chats |
| WeChat Enterprise| ✅     |                          |
| WeChat Customer Service| ✅     |                          |
| WeChat           | ✅     |                          |
| WeChat Official Account| ✅     |                          |
| Feishu           | ✅     |                          |
| DingTalk         | ✅     |                          |
| Discord          | ✅     |                          |
| Telegram         | ✅     |                          |
| Slack            | ✅     |                          |

### Large Language Model (LLM) Support

| Model                                  | Status | Notes                                  |
| -------------------------------------- | ------ | -------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Compatible with any OpenAI API models |
| [DeepSeek](https://www.deepseek.com/) | ✅     |                                        |
| [Moonshot](https://www.moonshot.cn/)   | ✅     |                                        |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                        |
| [xAI](https://x.ai/)                   | ✅     |                                        |
| [智谱AI](https://open.bigmodel.cn/)     | ✅     |                                        |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅ | LLMs and GPU Resources |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅ | LLMs and GPU Resources |
| [302.AI](https://share.302.ai/SuTG99) | ✅ | LLM Aggregation Platform  |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅ | |
| [Dify](https://dify.ai) | ✅ | LLMOps Platform |
| [Ollama](https://ollama.com/) | ✅ | Local LLM Platform |
| [LMStudio](https://lmstudio.ai/) | ✅ | Local LLM Platform |
| [GiteeAI](https://ai.gitee.com/) | ✅ | LLM API Aggregation Platform |
| [SiliconFlow](https://siliconflow.cn/) | ✅ | LLM Aggregation Platform |
| [阿里云百炼](https://bailian.console.aliyun.com/) | ✅ | LLM Aggregation Platform, LLMOps Platform |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅ | LLM Aggregation Platform, LLMOps Platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅ | LLM Aggregation Platform |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports tool access via MCP protocol |

### Text-to-Speech (TTS)

| Platform/Model                  | Notes                                       |
| ------------------------------- | ------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice)  |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)       | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)   |

### Text-to-Image

| Platform/Model             | Notes                                                         |
| -------------------------- | ------------------------------------------------------------- |
| 阿里云百炼           | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin)          |

## Community Contributions

A huge thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and the entire community for their invaluable contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>