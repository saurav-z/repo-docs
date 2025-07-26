# LangBot: Build Your Own AI Chatbot for Any Platform

**LangBot empowers you to create intelligent, multi-platform AI chatbots quickly and easily.**  This open-source platform provides a streamlined experience for developing IM bots with features like Agents, RAG, MCP, and extensive API support.

[Visit the original repository](https://github.com/langbot-app/LangBot)

## Key Features

*   **Versatile AI Capabilities:**
    *   💬 **Advanced Chatbot Features:** Supports multi-turn conversations, tool utilization, and multimodal interactions.
    *   🤖 **Agent Framework:**  Includes Agent capabilities for advanced automation.
    *   📚 **RAG (Retrieval-Augmented Generation):** Integrated RAG for knowledge-based responses, and seamless integration with [Dify](https://dify.ai).

*   **Multi-Platform Support:**
    *   QQ (Personal and Official Bots)
    *   QQ Channels
    *   WeChat
    *   WeCom (Enterprise WeChat)
    *   WeChat Official Accounts
    *   Feishu
    *   DingTalk
    *   Discord
    *   Telegram
    *   Slack

*   **Robust and Extensible:**
    *   🛡️ **Security:**  Built-in access control, rate limiting, and profanity filters.
    *   ⚙️ **Configuration:** Simple to set up, with various deployment options and support for multiple bot configurations.
    *   🔌 **Plugin Ecosystem:** Extensive plugin support with event-driven architecture and component extension, compatible with the [Anthropic MCP Protocol](https://modelcontextprotocol.io/). Hundreds of plugins available.

*   **User-Friendly Interface:**
    *   🖥️ **Web Management Panel:**  Manage your LangBot instances directly through a web browser, eliminating the need for manual configuration file edits.

## Quick Start

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access the bot at `http://localhost:5300`.

For detailed instructions, see the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options

*   **BaoTa Panel:**  Available in the Baota Panel (see documentation for details).
*   **Zeabur:**  [Deploy on Zeabur](https://zeabur.com/zh-CN/templates/ZKTBDH).
*   **Railway:**  [Deploy on Railway](https://railway.app/template/yRrAyL?referralCode=vogKPF).
*   **Manual Deployment:**  See the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html) for direct execution.

## Stay Updated

Star and watch the repository to stay informed about the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Demo

Explore the web UI at:  [https://demo.langbot.dev/](https://demo.langbot.dev/)
*   Login: demo@langbot.app
*   Password: langbot123456
*   Please note: This is a public demo environment; do not enter sensitive information.

## Supported Features & Models

### Message Platforms

| Platform           | Status | Notes                                      |
| ------------------ | ------ | ------------------------------------------ |
| QQ (Personal)      | ✅     | Private and Group Chats                     |
| QQ (Official Bot)  | ✅     | Supports Channels, Private and Group Chats |
| WeChat             | ✅     |                                            |
| WeCom              | ✅     |                                            |
| WeChat Official Account | ✅     |                                            |
| Feishu             | ✅     |                                            |
| DingTalk           | ✅     |                                            |
| Discord            | ✅     |                                            |
| Telegram           | ✅     |                                            |
| Slack              | ✅     |                                            |

### Large Language Models (LLMs)

| Model                  | Status | Notes                                       |
| ---------------------- | ------ | ------------------------------------------- |
| OpenAI                 | ✅     | Supports any OpenAI API-compatible model     |
| DeepSeek               | ✅     |                                             |
| Moonshot               | ✅     |                                             |
| Anthropic              | ✅     |                                             |
| xAI                    | ✅     |                                             |
| ZhipuAI                | ✅     |                                             |
| YouCloud                | ✅     | LLM & GPU Resources                         |
| PPIO                   | ✅     | LLM & GPU Resources                         |
| 302.AI                 | ✅     | LLM Aggregation Platform                  |
| Google Gemini          | ✅     |                                             |
| Dify                   | ✅     | LLMOps Platform                             |
| Ollama                 | ✅     | Local LLM Platform                         |
| LMStudio               | ✅     | Local LLM Platform                         |
| GiteeAI                | ✅     | LLM API Aggregation Platform              |
| SiliconFlow            | ✅     | LLM Aggregation Platform                  |
| Alibaba Cloud Baichuan | ✅     | LLM Aggregation Platform, LLMOps Platform |
| VolcEngine Ark         | ✅     | LLM Aggregation Platform, LLMOps Platform |
| ModelScope             | ✅     | LLM Aggregation Platform                  |
| MCP                    | ✅     | Supports tool usage via MCP protocol        |

### Text-to-Speech (TTS)

| Platform/Model  | Notes                                      |
| --------------- | ------------------------------------------ |
| FishAudio       | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| HaiTing AI      | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| AzureTTS       | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image (TTI)

| Platform/Model    | Notes                                  |
| ----------------- | -------------------------------------- |
| Alibaba Cloud Baichuan | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

Thank you to all the [contributors](https://github.com/langbot-app/LangBot/graphs/contributors) for their valuable contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>