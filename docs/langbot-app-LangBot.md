# LangBot: Open-Source IM Robot Development Platform (AI Chatbot Framework)

**Unleash the power of Large Language Models (LLMs) to build versatile and engaging chatbots across various messaging platforms with LangBot!**  ([View the original repository](https://github.com/langbot-app/LangBot))

<div align="center">
  <a href="https://langbot.app">
    <img src="https://docs.langbot.app/social_zh.png" alt="LangBot" width="300"/>
  </a>
</div>

<div align="center">
  <a href="https://hellogithub.com/repository/langbot-app/LangBot" target="_blank">
    <img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=5ce8ae2aa4f74316bf393b57b952433c&claim_uid=gtmc6YWjMZkT21R" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" />
  </a>

  [English](README_EN.md) / 简体中文 / [繁體中文](README_TW.md) / [日本語](README_JP.md) / (PR for your language)

  [![Discord](https://img.shields.io/discord/1335141740050649118?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb)](https://discord.gg/wdNEHETs87)
  [![QQ Group](https://img.shields.io/badge/%E7%A4%BE%E5%8C%BAQQ%E7%BE%A4-966235608-blue)](https://qm.qq.com/q/JLi38whHum)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/langbot-app/LangBot)
  [![GitHub release (latest by date)](https://img.shields.io/github/v/release/langbot-app/LangBot)](https://github.com/langbot-app/LangBot/releases/latest)
  <img src="https://img.shields.io/badge/python-3.10 ~ 3.13 -blue.svg" alt="python">
  [![star](https://gitcode.com/RockChinQ/LangBot/star/badge.svg)](https://gitcode.com/RockChinQ/LangBot)

  <a href="https://langbot.app">Project Homepage</a> |
  <a href="https://docs.langbot.app/zh/insight/guide.html">Deployment Documentation</a> |
  <a href="https://docs.langbot.app/zh/plugin/plugin-intro.html">Plugin Introduction</a> |
  <a href="https://github.com/langbot-app/LangBot/issues/new?assignees=&labels=%E7%8B%AC%E7%AB%8B%E6%8F%92%E4%BB%B6&projects=&template=submit-plugin.yml&title=%5BPlugin%5D%3A+%E8%AF%B7%E6%B1%82%E7%99%BB%E8%AE%B0%E6%96%B0%E6%8F%92%E4%BB%B6">Submit Plugin</a>
</div>

LangBot is an open-source platform designed for building native instant messaging (IM) robots powered by large language models. It provides a ready-to-use development experience with features like Agents, Retrieval-Augmented Generation (RAG), and Model Context Protocol (MCP) support.  It integrates with major IM platforms worldwide and offers extensive API interfaces for customized development.

## Key Features

*   **AI-Powered Conversations:**  Engage in dynamic conversations with various LLMs. Supports multi-turn dialogue, tool usage, and multimodal capabilities, along with built-in RAG for knowledge retrieval and seamless integration with [Dify](https://dify.ai).
*   **Cross-Platform Compatibility:**  Connect with users across a wide range of platforms, including QQ, QQ Channels, Enterprise WeChat, Personal WeChat, Feishu, Discord, Telegram, Slack, and more.
*   **Robust and Feature-Rich:**  Benefit from features such as access control, rate limiting, and profanity filtering.  Enjoy easy configuration and multiple deployment options for diverse use cases, including multi-pipeline configurations for different bot roles.
*   **Extensible with Plugins and Community Support:**  Expand functionality with an event-driven plugin system and component extensions.  Supports the Anthropic [MCP Protocol](https://modelcontextprotocol.io/) and boasts a thriving community with hundreds of plugins available.
*   **Web-Based Management:**  Manage your LangBot instances through an intuitive web UI, eliminating the need for manual configuration file editing.

Explore the detailed feature specifications in the [documentation](https://docs.langbot.app/zh/insight/features.html).

**Try it now!** Explore the demo environment at [https://demo.langbot.dev/](https://demo.langbot.dev/)
  - Login details:  Email: `demo@langbot.app`, Password: `langbot123456`
  - *Note: This is a public demo; please refrain from entering sensitive information.*

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access http://localhost:5300 to start using LangBot.

For detailed instructions, refer to the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options

*   **BaoTa Panel Deployment:** Available on BaoTa Panel.  See the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html) for installation instructions.
*   **Zeabur Cloud Deployment:**  Community-contributed Zeabur template.  [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud Deployment:**  [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:**  Run the release version directly. See the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Staying Updated

Stay informed about the latest developments by starring and watching the repository.
![star gif](https://docs.langbot.app/star.gif)

## Supported Platforms

| Platform           | Status | Notes                                  |
| ------------------ | ------ | -------------------------------------- |
| QQ Personal        | ✅     | Private and group chats                |
| QQ Official Bot    | ✅     | Supports Channels, private and group chats |
| Enterprise WeChat  | ✅     |                                        |
| Enterprise WeChat Customer Service | ✅     | |
| Personal WeChat    | ✅     |                                        |
| WeChat Official Account | ✅     |                                        |
| Feishu             | ✅     |                                        |
| DingTalk           | ✅     |                                        |
| Discord            | ✅     |                                        |
| Telegram           | ✅     |                                        |
| Slack              | ✅     |                                        |

## Supported Large Language Models (LLMs)

| Model                                 | Status | Notes                                     |
| ------------------------------------- | ------ | ----------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Supports any OpenAI API format model     |
| [DeepSeek](https://www.deepseek.com/) | ✅     |                                           |
| [Moonshot](https://www.moonshot.cn/)   | ✅     |                                           |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                           |
| [xAI](https://x.ai/)                  | ✅     |                                           |
| [智谱AI](https://open.bigmodel.cn/)  | ✅     |                                           |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     | LLM and GPU resource platform            |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | LLM and GPU resource platform            |
| [302.AI](https://share.302.ai/SuTG99)  | ✅     | LLM aggregation platform                   |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅ | |
| [Dify](https://dify.ai)                 | ✅     | LLMOps platform                           |
| [Ollama](https://ollama.com/)           | ✅     | Local LLM runtime platform                |
| [LMStudio](https://lmstudio.ai/)        | ✅     | Local LLM runtime platform                |
| [GiteeAI](https://ai.gitee.com/)       | ✅     | LLM API aggregation platform               |
| [SiliconFlow](https://siliconflow.cn/) | ✅     | LLM aggregation platform                   |
| [阿里云百炼](https://bailian.console.aliyun.com/) | ✅ | LLM aggregation platform, LLMOps platform |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅ | LLM aggregation platform, LLMOps platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅ | LLM aggregation platform |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports tool access via MCP protocol     |

## Text-to-Speech (TTS)

| Platform/Model                             | Notes                                      |
| ------------------------------------------ | ------------------------------------------ |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)      | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image

| Platform/Model                  | Notes                                      |
| ------------------------------- | ------------------------------------------ |
| 阿里云百炼                       | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

We extend our sincere gratitude to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and all community members for their invaluable contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO considerations:

*   **Strong Headline:**  Uses keywords like "AI Chatbot Framework," "Open-Source," and relevant terms to attract search traffic.
*   **One-Sentence Hook:** The opening sentence immediately grabs attention and conveys the core value proposition.
*   **Concise and Organized Structure:**  Clear headings and bullet points make the information easy to scan and digest.
*   **Keyword Optimization:**  Incorporates relevant keywords (e.g., "LLM," "chatbot," "IM robot," specific platform names, and feature terms) naturally throughout the content.
*   **Call to Action:**  Encourages users to try the demo and get involved.
*   **Internal Linking:**  Includes links to various documentation sections and the demo site, to help users explore the project, and to guide search engine crawlers.
*   **Emphasis on Key Benefits:**  The feature list highlights the most important selling points.
*   **Clear Structure:** Improves readability.
*   **Removed Redundancy:** Cleans up unnecessary repetition from original readme.
*   **Expanded Deployment info:** Improves users' getting started experience.
*   **Added Table of Contents:** Provides information on platforms, LLMs and TTS/Image generators.
*   **Contextual Use of Images:**  Improved use of the provided images.