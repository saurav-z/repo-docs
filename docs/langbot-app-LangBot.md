# LangBot: Your All-in-One AI Chatbot Solution

LangBot is a versatile and feature-rich AI chatbot platform designed to connect you with powerful language models across multiple platforms, offering a seamless and customizable conversational experience.  ([See the original repo](https://github.com/langbot-app/LangBot))

## Key Features

*   **💬 Advanced AI Chat & Agent Capabilities:** Supports a variety of large language models (LLMs), including OpenAI, DeepSeek, Moonshot, Anthropic, and more.  Offers multi-turn conversations, tool usage, and multi-modal support. Includes built-in RAG (Retrieval-Augmented Generation) and deep integration with [Dify](https://dify.ai).
*   **🤖 Multi-Platform Support:**  Works with popular messaging platforms:
    *   QQ (Personal & Official Accounts)
    *   Enterprise WeChat
    *   Personal WeChat
    *   Feishu (Lark)
    *   Discord
    *   Telegram
    *   Slack
    *   DingTalk
    *   LINE (In Development)
    *   WhatsApp (In Development)
*   **🛠️ High Stability & Comprehensive Features:**  Offers robust features like access control, rate limiting, and profanity filtering. Simple configuration with multiple deployment options. Supports multiple pipeline configurations for different use cases.
*   **🧩 Extensible with Plugins & Active Community:** Supports event-driven architecture and component extension through plugins. Compatible with the Anthropic [MCP protocol](https://modelcontextprotocol.io/).  Currently boasts hundreds of plugins.
*   **😻 Web Management Panel:** Manage your LangBot instance through a user-friendly web interface, eliminating the need for manual configuration file edits.

## Getting Started

### Docker Compose Deployment

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    ```

2.  **Start with Docker Compose:**

    ```bash
    docker compose up -d
    ```

3.  **Access:**  Visit http://localhost:5300 to start using LangBot.

    *   Detailed documentation: [Docker Deployment](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Alternative Deployment Methods

*   **BaoTa Panel:**  Available on the BaoTa panel. Refer to the documentation for installation: [BaoTa Panel Deployment](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).

*   **Zeabur Cloud:** Community-contributed Zeabur template:  [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)

*   **Railway Cloud:** [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)

*   **Manual Deployment:** Use the releases for manual deployment. See [Manual Deployment](https://docs.langbot.app/zh/deploy/langbot/manual.html) for instructions.

## Screenshots

[Include a concise summary of the screenshots with the key features shown. Example below.]

*   **WebUI Demonstration:** The LangBot web interface, showing bot management, model creation, pipeline editing, and plugin market.
*   **Example Responses:** Examples of the chatbot’s responses, including one with a web-browsing plugin.

    [Include the images in this section - replace the links with the appropriate image tag]

## Compatibility

### Messaging Platforms

| Platform           | Status | Notes                                |
| ------------------ | ------ | ------------------------------------ |
| QQ Personal        | ✅     | Private & Group Chats                 |
| QQ Official Bot    | ✅     | Channels, Private & Group Chats     |
| Enterprise WeChat | ✅     |                                      |
| Enterprise WeChat External | ✅     |                                      |
| Personal WeChat    | ✅     |                                      |
| WeChat Official Account | ✅     |                                      |
| Feishu             | ✅     |                                      |
| DingTalk           | ✅     |                                      |
| Discord            | ✅     |                                      |
| Telegram           | ✅     |                                      |
| Slack              | ✅     |                                      |
| LINE               | 🚧     | In Development                      |
| WhatsApp           | 🚧     | In Development                      |

### Large Language Models (LLMs)

| Model                            | Status | Notes                                      |
| -------------------------------- | ------ | ------------------------------------------ |
| [OpenAI](https://platform.openai.com/)  | ✅     | Supports any OpenAI API compatible model  |
| [DeepSeek](https://www.deepseek.com/)    | ✅     |                                            |
| [Moonshot](https://www.moonshot.cn/)   | ✅     |                                            |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                            |
| [xAI](https://x.ai/)                   | ✅     |                                            |
| [智谱AI](https://open.bigmodel.cn/)    | ✅     |                                            |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)   | ✅     | LLM and GPU resources platform           |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | LLM and GPU resources platform            |
| [302.AI](https://share.302.ai/SuTG99)    | ✅     | LLM Aggregation Platform                   |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     |                                            |
| [Dify](https://dify.ai)                  | ✅     | LLMOps Platform                            |
| [Ollama](https://ollama.com/)             | ✅     | Local LLM Platform                         |
| [LMStudio](https://lmstudio.ai/)          | ✅     | Local LLM Platform                         |
| [GiteeAI](https://ai.gitee.com/)         | ✅     | LLM API Aggregation Platform               |
| [SiliconFlow](https://siliconflow.cn/)  | ✅     | LLM Aggregation Platform                   |
| [阿里云百炼](https://bailian.console.aliyun.com/)  | ✅     | LLM Aggregation Platform, LLMOps Platform |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW)   | ✅     | LLM Aggregation Platform, LLMOps Platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro)    | ✅     | LLM Aggregation Platform                   |
| [MCP](https://modelcontextprotocol.io/)    | ✅     | Supports tools via MCP Protocol          |

### Text-to-Speech (TTS)

| Platform/Model                    | Notes                                                                  |
| --------------------------------- | ---------------------------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice)           |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)   | [Plugin](https://github.com/the-lazy-me/NewChatVoice)           |
| [AzureTTS](https://portal.azure.com/)          | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)          |

### Text-to-Image (TTI)

| Platform/Model          | Notes                                                                         |
| ----------------------- | ----------------------------------------------------------------------------- |
| 阿里云百炼              | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin)       |

## Community Contributions

[Thank the contributors and include a contributors chart image.]

We appreciate the contributions from the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and other community members:

[Insert contributor image here: <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />]

## Stay Updated

Stay informed about the latest updates by starring and watching the repository:

[Include the star gif - replace the link with the correct image tag]