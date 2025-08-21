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

</div>

# LangBot: Open-Source LLM-Powered Instant Messaging Bot Platform

**LangBot empowers you to build your own AI-powered chatbots for various instant messaging platforms, offering a flexible, extensible, and feature-rich solution.**

## Key Features

*   💬 **Advanced Conversational AI:**
    *   Supports a wide range of large language models (LLMs).
    *   Offers multi-turn conversations, tool utilization, multimodal capabilities, and streaming output.
    *   Includes built-in RAG (Retrieval-Augmented Generation) for knowledge base integration.
    *   Seamlessly integrates with [Dify](https://dify.ai).
*   🤖 **Multi-Platform Support:**
    *   Compatible with popular messaging platforms: QQ (personal & official bots), WeChat (personal & official accounts), Enterprise WeChat, Feishu, Discord, Telegram, and more.
*   🛠️ **Robust and Feature-Rich:**
    *   Built-in access control, rate limiting, and profanity filters.
    *   Simple configuration and multiple deployment options.
    *   Supports multi-pipeline configurations for diverse bot applications.
*   🧩 **Extensible with Plugins & Active Community:**
    *   Leverages event-driven and component-based plugin architecture.
    *   Adapts the Anthropic [MCP Protocol](https://modelcontextprotocol.io/).
    *   Hundreds of available plugins to enhance functionality.
*   😻 **Web-Based Management:**
    *   Manage your LangBot instances through a user-friendly web interface, eliminating the need for manual configuration file editing.

## Get Started

### Deployment Options

*   **Docker Compose:**

    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    docker compose up -d
    ```

    Access at http://localhost:5300.  For detailed instructions, see the [Docker Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).
*   **Baota Panel (宝塔面板):** Available in the Baota Panel app store.  Refer to the [Baota Panel Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Deploy easily with the community-contributed Zeabur template: [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:** Deploy using Railway: [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Explore manual deployment options in the [Manual Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch this repository to receive the latest updates!

![star gif](https://docs.langbot.app/star.gif)

## Platforms Supported

| Platform            | Status | Notes                                    |
| ------------------- | ------ | ---------------------------------------- |
| QQ (Personal)       | ✅      | Private & Group Chats                    |
| QQ (Official Bot)   | ✅      | Channels, Private & Group Chats         |
| Enterprise WeChat   | ✅      |                                          |
| Enterprise WeChat Customer Service | ✅ | |
| WeChat (Personal)     | ✅      |                                          |
| WeChat Official Account  | ✅ |                                          |
| Feishu              | ✅      |                                          |
| DingTalk            | ✅      |                                          |
| Discord             | ✅      |                                          |
| Telegram            | ✅      |                                          |
| Slack               | ✅      |                                          |

## Large Language Model (LLM) Support

| Model                       | Status | Notes                                |
| --------------------------- | ------ | ------------------------------------ |
| [OpenAI](https://platform.openai.com/)   | ✅      | Supports any OpenAI API format model  |
| [DeepSeek](https://www.deepseek.com/)     | ✅      |                                      |
| [Moonshot](https://www.moonshot.cn/)   | ✅      |                                      |
| [Anthropic](https://www.anthropic.com/)    | ✅      |                                      |
| [xAI](https://x.ai/)    | ✅      |                                      |
| [智谱AI](https://open.bigmodel.cn/)       | ✅      |                                      |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅ | Large Model and GPU Resource Platform |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅ | Large Model and GPU Resource Platform |
| [胜算云](https://www.shengsuanyun.com/login?code=7DS2QLH5) | ✅ | Large Model and GPU Resource Platform |
| [302.AI](https://share.302.ai/SuTG99) | ✅ | Large Model Aggregation Platform |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅ |                                      |
| [Dify](https://dify.ai)           | ✅      | LLMOps Platform                       |
| [Ollama](https://ollama.com/)        | ✅      | Local Large Model Platform           |
| [LMStudio](https://lmstudio.ai/)      | ✅      | Local Large Model Platform           |
| [GiteeAI](https://ai.gitee.com/)     | ✅      | Large Model API Aggregation Platform  |
| [SiliconFlow](https://siliconflow.cn/)      | ✅      | Large Model Aggregation Platform           |
| [阿里云百炼](https://bailian.console.aliyun.com/)        | ✅      | Large Model Aggregation Platform, LLMOps Platform           |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅ | Large Model Aggregation Platform, LLMOps Platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅ | Large Model Aggregation Platform |
| [MCP](https://modelcontextprotocol.io/)       | ✅      | Supports retrieving tools via MCP    |

## Text-to-Speech (TTS)

| Platform/Model                      | Notes                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice)                                         |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice)                                         |
| [AzureTTS](https://portal.azure.com/)      | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)                                       |

## Text-to-Image (TTI)

| Platform/Model                      | Notes                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| 阿里云百炼 | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin)                                        |

## Community Contributions

A big thank you to the [contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and all community members for their valuable contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>

**[Explore the LangBot GitHub Repository](https://github.com/langbot-app/LangBot) to learn more and contribute!**
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:** The one-sentence hook immediately establishes what LangBot is and its core benefit.
*   **SEO-Friendly Headings:** Uses descriptive headings (Key Features, Get Started, etc.) to improve readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points for easy scanning and highlights the most important aspects of the project.
*   **Keywords:**  Includes relevant keywords like "LLM," "AI-powered chatbot," "open-source," and platform names throughout the description.
*   **Structured Information:**  Organizes information logically for better readability and search engine indexing.
*   **Stronger Calls to Action:** Encourages users to explore the deployment options and contribute to the project.
*   **Direct Links:**  Provides direct links to the original repo and all the external resources.
*   **Updated Content:**  Uses the latest documentation and platform list.
*   **Cleaned Up Formatting:** Uses a consistent and modern markdown style.
*   **Removed Irrelevant Content:** The readme is focused on the project itself.