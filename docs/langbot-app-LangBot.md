# LangBot: The Open-Source LLM-Powered Chatbot Platform 🤖

**Empower your communication and streamline your workflows with LangBot, the open-source platform for building advanced, AI-driven chatbots.** ([View on GitHub](https://github.com/langbot-app/LangBot))

[<img src="https://docs.langbot.app/social_zh.png" alt="LangBot" width="200"/>](https://langbot.app)

LangBot is a versatile platform designed for developing and deploying cutting-edge AI chatbots. It offers a robust set of features, including support for Agents, Retrieval-Augmented Generation (RAG), Model Context Protocol (MCP), and integration with a wide range of messaging platforms, all while being easily customizable with a rich API and plugin ecosystem.

## Key Features:

*   **💬 Advanced LLM Capabilities:** Supports a variety of large language models (LLMs), enabling multi-turn conversations, tool usage, multimodal input, and streaming output. Includes built-in RAG (knowledge base) integration and seamless compatibility with [Dify](https://dify.ai).
*   **🤖 Broad Platform Support:** Currently compatible with QQ, QQ Channels, Enterprise WeChat, Personal WeChat, Feishu, Discord, Telegram, Slack, and more.
*   **🛠️ Robust and Feature-Rich:** Offers essential features like access control, rate limiting, and profanity filtering. Configuration is simplified, and multiple deployment options are available. Supports multi-pipeline configurations for diverse chatbot applications.
*   **🧩 Plugin Ecosystem & Active Community:** Extensible with an event-driven plugin system and component extensions. Supports the Anthropic [MCP protocol](https://modelcontextprotocol.io/). Hundreds of plugins are currently available.
*   **😻 Web-Based Management:** Manage LangBot instances directly through a web interface, eliminating the need for manual configuration file editing.

## Quick Start - Deploying LangBot:

Choose your preferred deployment method:

### Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access at http://localhost:5300.  See [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html) for details.

### Other Deployment Options:

*   **Baota Panel:** Available on the Baota Panel, follow the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur:** Deploy using the community-contributed [Zeabur template](https://zeabur.com/zh-CN/templates/ZKTBDH).
*   **Railway:** Deploy using the [Railway template](https://railway.app/template/yRrAyL?referralCode=vogKPF).
*   **Manual Deployment:** Deploy directly from the release packages, see the [manual deployment guide](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated:

Star and watch the repository to receive the latest updates and announcements!

![star gif](https://docs.langbot.app/star.gif)

## Platform and Model Support:

### Messaging Platforms

| Platform          | Status | Notes                       |
| ----------------- | ------ | --------------------------- |
| QQ (Personal)     | ✅     | Private/Group Chats         |
| QQ (Official Bot) | ✅     | Channels, Private/Group Chats |
| Enterprise WeChat | ✅     |                             |
| WeChat External   | ✅     |                             |
| Personal WeChat   | ✅     |                             |
| WeChat Official Account | ✅     |                            |
| Feishu            | ✅     |                             |
| DingTalk          | ✅     |                             |
| Discord           | ✅     |                             |
| Telegram          | ✅     |                             |
| Slack             | ✅     |                             |

### Large Language Model (LLM) Support:

| Model Provider       | Status | Notes                                      |
| -------------------- | ------ | ------------------------------------------ |
| OpenAI               | ✅     | Supports all OpenAI-compatible models      |
| DeepSeek             | ✅     |                                            |
| Moonshot             | ✅     |                                            |
| Anthropic            | ✅     |                                            |
| xAI                  | ✅     |                                            |
| Zhipu AI             | ✅     |                                            |
| Youyun Zhisuan       | ✅     | Model and GPU resources platform           |
| PPIO                 | ✅     | Model and GPU resources platform           |
| 302.AI               | ✅     | LLM Aggregation Platform                   |
| Google Gemini        | ✅     |                                            |
| Dify                 | ✅     | LLMOps Platform                            |
| Ollama               | ✅     | Local LLM Platform                         |
| LMStudio             | ✅     | Local LLM Platform                         |
| GiteeAI              | ✅     | LLM Aggregation Platform                   |
| SiliconFlow          | ✅     | LLM Aggregation Platform                   |
| Alibaba Cloud Bailian | ✅     | LLM Aggregation & LLMOps Platform           |
| VolcEngine Ark       | ✅     | LLM Aggregation & LLMOps Platform           |
| ModelScope           | ✅     | LLM Aggregation Platform                   |
| MCP                  | ✅     | Supports tool access via MCP protocol      |

### Text-to-Speech (TTS)

| Platform/Model | Notes                       |
| -------------- | --------------------------- |
| FishAudio      | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| Haitun AI      | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| AzureTTS       | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image

| Platform/Model    | Notes                          |
| ----------------- | ------------------------------ |
| Alibaba Cloud Bailian | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions:

A big thank you to all the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members who have contributed to LangBot.

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>