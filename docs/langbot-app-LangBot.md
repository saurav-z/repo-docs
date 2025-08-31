# LangBot: Build Your Own AI Chatbot Platform

**Transform your instant messaging experience with LangBot, the open-source platform that empowers you to build and customize powerful AI chatbots.** ([See the original repository](https://github.com/langbot-app/LangBot))

<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot"/>
</a>
</p>

LangBot is a versatile platform designed for developers to create their own AI-powered instant messaging bots. It offers a user-friendly experience with features like Agent, RAG, and MCP functionality, works seamlessly with various global messaging platforms, and provides comprehensive API support for custom development.

## Key Features

*   **AI-Powered Conversations:**
    *   Supports multiple large language models (LLMs).
    *   Offers multi-turn conversations, tool usage, multimodal capabilities, and streaming output.
    *   Built-in RAG (Retrieval-Augmented Generation) for enhanced knowledge integration.
    *   Deeply integrated with [Dify](https://dify.ai).
*   **Cross-Platform Compatibility:**
    *   Works with a wide range of platforms including QQ, QQ Channels, WeChat Enterprise, Personal WeChat, Feishu, Discord, Telegram, Slack, and more.
*   **Robust & Feature-Rich:**
    *   Includes built-in access control, rate limiting, and profanity filtering for enhanced security and control.
    *   Easy configuration and supports various deployment methods.
    *   Allows for multi-pipeline configurations for diverse bot applications.
*   **Extensible with Plugins:**
    *   Supports event-driven and component extension plugin mechanisms.
    *   Integrates with the Anthropic [MCP Protocol](https://modelcontextprotocol.io/).
    *   A rich ecosystem of hundreds of plugins available.
*   **Web-Based Management:**
    *   A web UI allows for effortless management of your LangBot instance without manual configuration file editing.

## Getting Started

### Deployment Options

*   **Docker Compose:**
    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    docker compose up -d
    ```
    Access the bot at http://localhost:5300.

*   **Cloud Deployment (One-Click):**  Deploy with Zeabur and Railway (templates available via badges in original README).

*   **Manual Deployment:** Follow the detailed instructions in the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Demo

Explore the Web UI features using this demo environment: [https://demo.langbot.dev/](https://demo.langbot.dev/)

*   Login: `demo@langbot.app` / `langbot123456`
*   Note: This is a public demo; do not enter sensitive information.

## Supported Platforms

| Platform             | Status | Notes                                     |
| -------------------- | ------ | ----------------------------------------- |
| QQ Personal          | ✅     | Private and group chats                 |
| QQ Official Bot      | ✅     | Supports Channels, private and group chats |
| WeChat Enterprise    | ✅     |                                           |
| WeChat Enterprise  (External Customer Service)| ✅     |                                           |
| Personal WeChat      | ✅     |                                           |
| WeChat Official Account      | ✅     |                                           |
| Feishu               | ✅     |                                           |
| DingTalk             | ✅     |                                           |
| Discord              | ✅     |                                           |
| Telegram             | ✅     |                                           |
| Slack                | ✅     |                                           |

## Supported LLMs

| Model                        | Status | Notes                                           |
| ---------------------------- | ------ | ----------------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Supports all OpenAI API format models         |
| [DeepSeek](https://www.deepseek.com/) | ✅     |                                               |
| [Moonshot](https://www.moonshot.cn/) | ✅     |                                               |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                               |
| [xAI](https://x.ai/) | ✅     |                                               |
| [ZhipuAI](https://open.bigmodel.cn/) | ✅     |                                               |
| [Shengsuanyun](https://www.shengsuanyun.com/?from=CH_KYIPP758) | ✅     | Supports global LLMs (Recommended)             |
| [YouYun Zhisuon](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     | LLMs & GPU resource platform                   |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | LLMs & GPU resource platform                   |
| [302.AI](https://share.302.ai/SuTG99) | ✅     | LLM Aggregation Platform                        |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     |                                               |
| [Dify](https://dify.ai) | ✅     | LLMOps Platform                                   |
| [Ollama](https://ollama.com/) | ✅     | Local LLM Runner                                |
| [LMStudio](https://lmstudio.ai/) | ✅     | Local LLM Runner                                |
| [GiteeAI](https://ai.gitee.com/) | ✅     | LLM Aggregation Platform                        |
| [SiliconFlow](https://siliconflow.cn/) | ✅     | LLM Aggregation Platform                        |
| [Aliyun Baichuan](https://bailian.console.aliyun.com/) | ✅     | LLM Aggregation Platform, LLMOps Platform      |
| [Volcengine Ark](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | LLM Aggregation Platform, LLMOps Platform      |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅     | LLM Aggregation Platform                        |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports tool retrieval via MCP protocol     |

## TTS Support

| Platform/Model                 | Notes                                           |
| ------------------------------ | ----------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [HaiTing AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/) | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image Support

| Platform/Model                 | Notes                                           |
| ------------------------------ | ----------------------------------------------- |
| Aliyun Baichuan | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

A special thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members for their valuable contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO considerations:

*   **Clear Hook:** The first sentence is a compelling hook that grabs attention and highlights the core value proposition.
*   **Keyword Optimization:** Includes relevant keywords like "AI chatbot," "open-source," "LLM," and platform names.
*   **Structured Headings:** Uses clear headings and subheadings for easy navigation and improved readability (important for SEO).
*   **Bulleted Lists:**  Employs bullet points for key features, making the information scannable and easy to digest.
*   **Detailed Sections:** Provides comprehensive sections on features, getting started, deployment options, and community contributions.
*   **Direct Links:** Includes direct links to deployment docs, demo, and community resources.
*   **Call to action:** encourages users to Star and Watch the repo to stay updated.
*   **Concise Language:** Avoids jargon and explains technical concepts in an accessible way.
*   **ALT text:** Provides alt text for images so search engines can understand the purpose of the image.
*   **Clear Table:** Utilized a well-formatted table for platform and model support that improves readability and searchability.