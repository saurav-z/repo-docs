# LangBot: The Open-Source IM Bot Platform for LLMs

**Unleash the power of Large Language Models (LLMs) in your favorite messaging platforms with LangBot, a versatile, open-source platform for building intelligent, conversational bots.**  [Explore the original repository](https://github.com/langbot-app/LangBot)

LangBot provides a ready-to-use development experience for building LLM-powered chatbots, supporting various applications like Agents, Retrieval-Augmented Generation (RAG), Model Context Protocol (MCP), and more. It seamlessly integrates with popular instant messaging platforms and offers extensive API interfaces for custom development.

## Key Features:

*   **LLM-Powered Conversations & Agents:** Engage in multi-turn dialogues, leverage tool utilization, and experience multimodal capabilities.  Includes built-in RAG for knowledge base integration and full compatibility with [Dify](https://dify.ai).
*   **Cross-Platform Compatibility:**  Works seamlessly with a wide array of messaging platforms, including:
    *   QQ (personal and official bot)
    *   QQ Channels
    *   WeCom (Enterprise WeChat)
    *   Personal WeChat
    *   Feishu
    *   Discord
    *   Telegram
    *   Slack
*   **Robust & Feature-Rich:** Benefit from built-in features such as access control, rate limiting, and profanity filtering.  Offers easy configuration and supports various deployment methods. Supports multi-pipeline configurations for different use cases.
*   **Extensible with Plugins & Active Community:**  Supports event-driven and component-based plugin architecture, including Anthropic's [MCP protocol](https://modelcontextprotocol.io/).  Hundreds of plugins already available.
*   **Intuitive Web Management Panel:**  Manage your LangBot instance directly through a web browser, simplifying configuration.

## Get Started:

### Deploy with Docker Compose:

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access your bot at `http://localhost:5300`.
Detailed instructions can be found in the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Additional Deployment Options:

*   **BaoTa Panel:** Available on the BaoTa panel.  Refer to the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Deploy using the community-contributed Zeabur template. [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:** Deploy on Railway.  [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** For manual setup, see the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated:

Star and watch the repository to stay informed about the latest developments.

![star gif](https://docs.langbot.app/star.gif)

##  Example Demo:

Explore the web UI demo:  [https://demo.langbot.dev/](https://demo.langbot.dev/)
*   Login:  `demo@langbot.app`
*   Password:  `langbot123456`
*   Note: This is a public demo; please avoid entering sensitive information.

## Supported Platforms, Large Language Models, TTS & Image Generation

*(Table formats have been removed to avoid formatting issues with markdown)*

*   **Message Platforms**: QQ (Personal and Bot), WeCom, WeChat, Feishu, DingTalk, Discord, Telegram, Slack
*   **LLMs**: OpenAI, DeepSeek, Moonshot, Anthropic, xAI, ZhiPu AI, 302.AI, Google Gemini, Dify, Ollama, LMStudio, GiteeAI, SiliconFlow, AliYun BaiLian, ModelScope, and MCP
*   **TTS**: FishAudio and AzureTTS
*   **Image Generation**: AliYun BaiLian

##  Community Contributions

A special thank you to all [contributors](https://github.com/langbot-app/LangBot/graphs/contributors).

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>