# LangBot: Open-Source LLM-Powered IM Bot Development Platform

**Empower your instant messaging with AI!** LangBot is a cutting-edge, open-source platform designed to simplify the development of Large Language Model (LLM)-powered chatbots for various messaging platforms, offering a seamless and customizable experience for users and developers alike. Learn more and contribute at the [original repo](https://github.com/langbot-app/LangBot).

[![Discord](https://img.shields.io/discord/1335141740050649118?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb)](https://discord.gg/wdNEHETs87)
[![QQ Group](https://img.shields.io/badge/%E7%A4%BE%E5%8C%BAQQ%E7%BE%A4-966235608-blue)](https://qm.qq.com/q/JLi38whHum)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/langbot-app/LangBot)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/langbot-app/LangBot)](https://github.com/langbot-app/LangBot/releases/latest)
<img src="https://img.shields.io/badge/python-3.10 ~ 3.13 -blue.svg" alt="python">
[![star](https://gitcode.com/RockChinQ/LangBot/star/badge.svg)](https://gitcode.com/RockChinQ/LangBot)

**Key Features:**

*   💬 **Advanced LLM Capabilities:** Integrate various large language models, enabling multi-turn conversations, tool usage, and multimodal functionality. Includes built-in RAG (Retrieval-Augmented Generation) support and seamless integration with [Dify](https://dify.ai).
*   🤖 **Multi-Platform Support:** Compatible with popular messaging platforms, including QQ, QQ Channels, WeChat Work, Personal WeChat, Feishu, Discord, Telegram, Slack, and more.
*   🛠️ **Robust and Feature-Rich:** Offers features such as access control, rate limiting, and profanity filtering. Highly configurable and supports multiple deployment methods. Provides multi-pipeline configurations for diverse bot applications.
*   🧩 **Extensible with Plugins & Active Community:** Supports event-driven and component-based plugin mechanisms. Compatible with the Anthropic [MCP protocol](https://modelcontextprotocol.io/), with hundreds of existing plugins.
*   😻 **Web-Based Management:** Manage LangBot instances directly through a web browser, eliminating the need for manual configuration file editing.

**Get Started:**

*   **Docker Compose:** Deploy quickly using Docker Compose.
    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    docker compose up -d
    ```
    Access at http://localhost:5300. More details in the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

*   **Other Deployment Options:**  Available on Baota Panel, Zeabur, and Railway. See the links in the original README.

**Stay Updated:**  Star and Watch the repository to receive the latest updates!

**Try the Demo:**  Experience LangBot's features in the demo environment: https://demo.langbot.dev/
*   Login details:  email: `demo@langbot.app`, password: `langbot123456`

**Supported Platforms:**

| Platform            | Status | Notes                                    |
| ------------------- | ------ | ---------------------------------------- |
| QQ Personal         | ✅     | Private and group chats                  |
| QQ Official Bot     | ✅     | Supports channels, private, and group chats |
| WeChat Work         | ✅     |                                          |
| WeChat External     | ✅     |                                          |
| WeChat Official Account | ✅     |                                          |
| Feishu              | ✅     |                                          |
| DingTalk            | ✅     |                                          |
| Discord             | ✅     |                                          |
| Telegram            | ✅     |                                          |
| Slack               | ✅     |                                          |

**Supported LLMs:**

| Model                       | Status | Notes                                    |
| --------------------------- | ------ | ---------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Compatible with all OpenAI-compatible models |
| [DeepSeek](https://www.deepseek.com/)      | ✅     |                                          |
| [Moonshot](https://www.moonshot.cn/)     | ✅     |                                          |
| [Anthropic](https://www.anthropic.com/)   | ✅     |                                          |
| [xAI](https://x.ai/)   | ✅     |                                          |
| [智谱AI](https://open.bigmodel.cn/)        | ✅     |                                          |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)      | ✅     | LLM and GPU resources platform             |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | LLM and GPU resources platform             |
| [302.AI](https://share.302.ai/SuTG99)      | ✅     | LLM aggregation platform                 |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     |                                          |
| [Dify](https://dify.ai)                   | ✅     | LLMOps platform                          |
| [Ollama](https://ollama.com/)                 | ✅     | Local LLM Platform                     |
| [LMStudio](https://lmstudio.ai/)                | ✅     | Local LLM Platform                     |
| [GiteeAI](https://ai.gitee.com/)              | ✅     | LLM interface aggregation platform       |
| [SiliconFlow](https://siliconflow.cn/)         | ✅     | LLM aggregation platform                 |
| [阿里云百炼](https://bailian.console.aliyun.com/)       | ✅     | LLM aggregation and LLMOps platform      |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW)   | ✅     | LLM aggregation and LLMOps platform      |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro)       | ✅     | LLM aggregation platform      |
| [MCP](https://modelcontextprotocol.io/)      | ✅     | Supports tool access via MCP protocol     |

**TTS Support:**

| Platform/Model         | Notes                              |
| ---------------------- | ---------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)   | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)            | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

**Text-to-Image Support:**

| Platform/Model       | Notes                            |
| -------------------- | -------------------------------- |
| 阿里云百炼 | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

**Community Contributions:**

We are grateful to all the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members for their contributions to LangBot.

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>