# LangBot: The Open-Source LLM Chatbot Platform for Seamless Integration

**Create powerful, AI-powered chatbots with LangBot, an open-source platform designed for easy integration with popular messaging platforms.** ([Original Repo](https://github.com/langbot-app/LangBot))

LangBot is your go-to solution for building versatile LLM (Large Language Model) chatbots. It offers a streamlined experience for developers, providing agent capabilities, Retrieval-Augmented Generation (RAG), MCP support, and more, all while supporting a wide range of messaging platforms and offering extensive API integrations for custom development.

## Key Features

*   **Versatile LLM Integration:** Seamlessly integrates with leading large language models like OpenAI, DeepSeek, Moonshot, and more, offering multi-turn conversations, tool usage, and multimodal capabilities.
*   **RAG (Retrieval-Augmented Generation) for Knowledge Base:** Includes built-in RAG support for integrating knowledge bases.
*   **Cross-Platform Compatibility:** Supports major messaging platforms including QQ, QQ Channels, WeChat, Enterprise WeChat, Feishu, Discord, Telegram, Slack and others.
*   **Robust and Feature-Rich:** Provides essential features like access control, rate limiting, and profanity filtering. It's easy to configure and supports multiple deployment options. Multi-pipeline configuration supports different bot scenarios.
*   **Extensible with Plugins:** Offers a modular plugin architecture with event-driven capabilities for expanding functionalities.  Supports Anthropic's [MCP protocol](https://modelcontextprotocol.io/) and includes hundreds of available plugins.
*   **Web-Based Management:** Manage your LangBot instance through a user-friendly web interface, eliminating the need for manual configuration file editing.

## Quick Start

### Docker Compose Deployment

1.  Clone the repository:

    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    ```

2.  Deploy with Docker Compose:

    ```bash
    docker compose up -d
    ```

3.  Access the bot at http://localhost:5300.

    For detailed instructions, see the [Docker Deployment Guide](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options

*   **Baota Panel:** Deployed with one-click installation on Baota Panel, see the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur:** Community contributed Zeabur template. [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway:** Deploy on Railway: [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Run from release builds, refer to the [manual deployment instructions](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and Watch the repository to stay informed about the latest developments!

![star gif](https://docs.langbot.app/star.gif)

## Demo

Experience the WebUI by visiting: https://demo.langbot.dev/
  *   Login: demo@langbot.app / langbot123456
  *   Note: Public Demo - avoid entering sensitive information.

## Supported Messaging Platforms

| Platform         | Status | Notes                             |
| ---------------- | ------ | --------------------------------- |
| QQ (Personal)    | ✅     | Private and group chat            |
| QQ (Official Bot) | ✅     | QQ Official Bot, supports channels, private chats and group chats |
| WeChat           | ✅     |                                   |
| Enterprise WeChat | ✅     |                                   |
| WeChat Official Account | ✅     |                                   |
| Feishu           | ✅     |                                   |
| DingTalk         | ✅     |                                   |
| Discord          | ✅     |                                   |
| Telegram         | ✅     |                                   |
| Slack            | ✅     |                                   |

## Supported LLMs and Integrations

| Model/Platform                     | Status | Notes                                                                 |
| ---------------------------------- | ------ | --------------------------------------------------------------------- |
| OpenAI (and compatible)            | ✅     | Compatible with any OpenAI API-formatted models                      |
| DeepSeek                           | ✅     |                                                                       |
| Moonshot                           | ✅     |                                                                       |
| Anthropic                          | ✅     |                                                                       |
| xAI                                | ✅     |                                                                       |
| ZhiPu AI                           | ✅     |                                                                       |
| YouCloud Intelligence            | ✅     | LLM and GPU resource platform                                         |
| PPIO                               | ✅     | LLM and GPU resource platform                                         |
| 302.AI                             | ✅     | LLM Aggregation Platform                                              |
| Google Gemini                      | ✅     |                                                                       |
| Dify                               | ✅     | LLMOps platform                                                       |
| Ollama                             | ✅     | Local LLM platform                                                  |
| LMStudio                           | ✅     | Local LLM platform                                                  |
| GiteeAI                            | ✅     | LLM API Aggregation Platform                                          |
| SiliconFlow                        | ✅     | LLM Aggregation Platform                                              |
| Alibaba Cloud Baichuan             | ✅     | LLM and LLMOps platform                                               |
| Volcengine Ark                     | ✅     | LLM and LLMOps platform                                               |
| ModelScope                         | ✅     | LLM Aggregation Platform                                              |
| MCP (Model Context Protocol)       | ✅     | Supports tool access via the MCP protocol                              |

## TTS Integration

| Platform/Model                  | Notes                                        |
| --------------------------------- | -------------------------------------------- |
| FishAudio                         | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| HaiDolphin AI                     | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| AzureTTS                          | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)   |

## Text-to-Image Integration

| Platform/Model      | Notes                                          |
| ------------------- | ---------------------------------------------- |
| Alibaba Cloud Baichuan | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Contributors

A special thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members for their contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:** The first sentence immediately establishes the value proposition and purpose of LangBot.
*   **Targeted Keywords:** Includes keywords like "open-source," "LLM," "chatbot," "AI-powered," "messaging platforms," "RAG," and "plugins."
*   **Informative Headings:** Uses clear and descriptive headings to structure the content and improve readability.
*   **Bulleted Lists:** Uses bullet points to highlight key features and benefits.
*   **Action-Oriented Language:** Uses phrases like "Create," "Explore," and "Get Started."
*   **Internal Links:** Links to key documentation sections for easy navigation.
*   **Mobile-Friendly:**  Formatted to be readable on mobile devices.
*   **Links back to the original repo**
*   **Comprehensive Feature Lists:** Provides a detailed list of features and integrations.
*   **Community Section:** Highlights the community contributions.