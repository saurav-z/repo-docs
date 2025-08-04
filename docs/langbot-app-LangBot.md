# LangBot: The Open-Source LLM Communication Platform

**Effortlessly build and deploy intelligent chatbots on popular messaging platforms with LangBot, the open-source platform designed for developers.**

[<img src="https://docs.langbot.app/social_zh.png" alt="LangBot" width="300"/>](https://langbot.app)

**Key Features:**

*   💬 **AI-Powered Conversations:** Engage in dynamic dialogues with support for a variety of Large Language Models (LLMs), including OpenAI, DeepSeek, and Anthropic, across multiple messaging platforms.
*   🤖 **Multi-Platform Compatibility:** Seamlessly connect your chatbots to popular messaging apps such as QQ, WeChat, Discord, Telegram, and more.
*   🛠️ **Robust & Customizable:** Benefit from features like access control, rate limiting, and profanity filters. Easily configure and deploy LangBot through various methods.
*   🧩 **Extensible with Plugins:** Enhance LangBot's capabilities through a rich ecosystem of plugins, including integrations with Anthropic's MCP protocol.
*   😻 **Intuitive Web Interface:** Manage your LangBot instance easily with a user-friendly web interface, eliminating the need for manual configuration file adjustments.

**Get Started:**

*   **[Project Homepage](https://langbot.app)**
*   **[Deployment Documentation](https://docs.langbot.app/zh/insight/guide.html)**
*   **[Plugin Introduction](https://docs.langbot.app/zh/plugin/plugin-intro.html)**
*   **[Submit a Plugin](https://github.com/langbot-app/LangBot/issues/new?assignees=&labels=%E7%8B%AC%E7%AB%8B%E6%8F%92%E4%BB%B6&projects=&template=submit-plugin.yml&title=%5BPlugin%5D%3A+%E8%AF%B7%E6%B1%82%E7%99%BB%E8%AE%B0%E6%96%B0%E6%8F%92%E4%BB%B6)**

**Deployment Options:**

*   **Docker Compose:**
    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    docker compose up -d
    ```
    Access at `http://localhost:5300`.
*   **[Baota Panel Deployment](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html)**
*   **[Zeabur Cloud Deployment](https://zeabur.com/zh-CN/templates/ZKTBDH)**
*   **[Railway Cloud Deployment](https://railway.app/template/yRrAyL?referralCode=vogKPF)**
*   **[Manual Deployment](https://docs.langbot.app/zh/deploy/langbot/manual.html)**

**Stay Updated:**

*   Star and Watch the repository to receive the latest updates.

[<img src="https://docs.langbot.app/star.gif" alt="Star gif" width="200"/>](https://github.com/langbot-app/LangBot)

**Supported Platforms:**

| Platform              | Status | Notes                                |
| --------------------- | ------ | ------------------------------------ |
| QQ Personal           | ✅     | Private and group chats              |
| QQ Official Bot       | ✅     | Supports channels, private & group chats |
| WeChat                | ✅     |                                      |
| Enterprise WeChat     | ✅     |                                      |
| WeChat Official Account | ✅     |                                      |
| Feishu                | ✅     |                                      |
| DingTalk              | ✅     |                                      |
| Discord               | ✅     |                                      |
| Telegram              | ✅     |                                      |
| Slack                 | ✅     |                                      |

**LLM Support:**

| Model                            | Status | Notes                                |
| -------------------------------- | ------ | ------------------------------------ |
| OpenAI                          | ✅     | Any OpenAI API format model           |
| DeepSeek                        | ✅     |                                      |
| Moonshot                        | ✅     |                                      |
| Anthropic                       | ✅     |                                      |
| xAI                             | ✅     |                                      |
| ZhiPu AI                        | ✅     |                                      |
| YouCloud Compute              | ✅     | LLM and GPU resource platform      |
| PPIO                            | ✅     | LLM and GPU resource platform      |
| 302.AI                          | ✅     | LLM Aggregation Platform              |
| Google Gemini                   | ✅     |                                      |
| Dify                            | ✅     | LLMOps Platform                       |
| Ollama                          | ✅     | Local LLM platform                     |
| LMStudio                        | ✅     | Local LLM platform                     |
| GiteeAI                         | ✅     | LLM API Aggregation Platform          |
| SiliconFlow                     | ✅     | LLM Aggregation Platform              |
| Alibaba Baichuan                | ✅     | LLM Aggregation Platform, LLMOps      |
| Volcengine Ark                  | ✅     | LLM Aggregation Platform, LLMOps      |
| ModelScope                      | ✅     | LLM Aggregation Platform              |
| MCP                             | ✅     | Supports tool access via MCP protocol |

**TTS Integrations:**

| Platform/Model | Notes                                    |
| -------------- | ---------------------------------------- |
| FishAudio      | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| Dolphin AI     | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| AzureTTS       | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

**Text-to-Image Integrations:**

| Platform/Model | Notes |
| -------------- | ----- |
| Alibaba Baichuan | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

**Community Contributions:**

A big thank you to the [contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and the community for their valuable contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>

**[View the original repository](https://github.com/langbot-app/LangBot)**