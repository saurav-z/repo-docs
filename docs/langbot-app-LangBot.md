# LangBot: Build Your Own AI Chatbot on Any Platform

**Create powerful, customizable AI chatbots for your favorite messaging platforms with LangBot, the open-source platform designed for seamless integration and extensive features.**  [Go to the LangBot Repository](https://github.com/langbot-app/LangBot)

## Key Features:

*   **Versatile AI Chatbot Capabilities:**
    *   Supports advanced LLM functionalities, including Agent, RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol)
    *   Offers multi-turn conversations, tool usage, multimodal support, and streaming output capabilities
    *   Deeply integrates with [Dify](https://dify.ai) for LLMOps
*   **Multi-Platform Compatibility:**
    *   Works with a wide range of messaging platforms, including QQ, QQ Channels, WeChat (Personal and Official Accounts), Feishu, Discord, Telegram, Slack, and Enterprise WeChat.
*   **Robust and Feature-Rich:**
    *   Includes built-in access control, rate limiting, and profanity filters for enhanced security and control.
    *   Easy configuration with support for multiple deployment methods and pipeline configurations for diverse use cases.
*   **Extensible and Community-Driven:**
    *   Leverages event-driven architecture and component extensions for flexible plugin development.
    *   Supports the Anthropic [MCP protocol](https://modelcontextprotocol.io/) for wider compatibility.
    *   Access to a growing library of hundreds of plugins.
*   **User-Friendly Management:**
    *   Provides a web-based management panel for easy configuration and management of your LangBot instances.

## Getting Started:

Choose your preferred deployment method:

### 📦 Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access your LangBot instance at `http://localhost:5300`.

### ☁️ Cloud Deployments:

*   **Zeabur:**  [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway:** [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)

### ⚙️ Manual Deployment:

See the [Manual Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated:

Star and watch the repository to receive the latest updates!

##  🚀 Model and Platform Support:

LangBot offers comprehensive support for various models and platforms, ensuring flexibility and adaptability:

### Supported Messaging Platforms:

*   QQ (Personal & Group)
*   QQ Official Bot (Channels, Private & Group)
*   Enterprise WeChat
*   Personal WeChat
*   WeChat Official Account
*   Feishu
*   DingTalk
*   Discord
*   Telegram
*   Slack

### Supported Large Language Models (LLMs) & APIs:

*   OpenAI (Supports all OpenAI API models)
*   DeepSeek
*   Moonshot
*   Anthropic
*   xAI
*   Zhipu AI
*   YouCloud Intelligence
*   PPIO
*   ShengSuan Cloud
*   302.AI
*   Google Gemini
*   Dify (LLMOps Platform)
*   Ollama (Local LLM)
*   LMStudio (Local LLM)
*   GiteeAI
*   SiliconFlow
*   Alibaba Cloud Bailian (LLMOps Platform)
*   Volcengine Ark
*   ModelScope
*   MCP (Model Context Protocol)

### Text-to-Speech (TTS) Integrations:

*   FishAudio ([Plugin](https://github.com/the-lazy-me/NewChatVoice))
*   HaiTing AI ([Plugin](https://github.com/the-lazy-me/NewChatVoice))
*   AzureTTS ([Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS))

### Text-to-Image (TTI) Integrations:

*   Alibaba Cloud Bailian ([Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin))

## Demo:

Try out the web UI at https://demo.langbot.dev/

*   Login: `demo@langbot.app`, Password: `langbot123456`
    *   Note: This is a public demo. Please do not enter any sensitive information.

##  🤝 Contributing:

A big thanks to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and the community!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>