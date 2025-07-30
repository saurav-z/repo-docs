# LangBot: The Open-Source LLM-Powered Chatbot Platform

**Empower your instant messaging experience with LangBot, a powerful and versatile open-source platform for building intelligent chatbots.** ([Back to Original Repo](https://github.com/langbot-app/LangBot))

<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot"/>
</a>
</p>

LangBot is your all-in-one solution for creating IM bots leveraging the power of large language models (LLMs). It offers a user-friendly, out-of-the-box experience for developing advanced chatbots with Agent, RAG, and MCP capabilities, all while supporting major global instant messaging platforms. Expand your bot's capabilities with extensive API integrations and custom development options.

## Key Features

*   **LLM-Powered Conversations:** Engage in sophisticated conversations with multiple LLM support, featuring multi-turn dialogues, tool usage, and multimodal capabilities. Includes built-in RAG (Retrieval-Augmented Generation) for knowledge retrieval and seamless integration with [Dify](https://dify.ai).
*   **Broad Platform Support:** Connect your bot to various platforms, including QQ, QQ Channels, WeChat, Enterprise WeChat, Feishu, Discord, Telegram, Slack, and more.
*   **Robust & Feature-Rich:** Benefit from stability with built-in access control, rate limiting, and profanity filters. Easily configured and supports multiple deployment methods. Supports multi-pipeline configurations for various applications.
*   **Extensible with Plugins:** Expand your bot's functionality with a plugin architecture based on event-driven and component-based design. Adapt Anthropic's [MCP Protocol](https://modelcontextprotocol.io/). Currently supports hundreds of plugins.
*   **Web-Based Management:** Manage your LangBot instance through an intuitive web interface, eliminating the need for manual configuration file edits.

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access your LangBot instance at: `http://localhost:5300`.

For detailed instructions, see the [Docker deployment guide](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options

*   **BaoTa Panel:** Available for one-click installation if you have Baota Panel installed.  See the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Deploy quickly using the community-contributed Zeabur template.  [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:** Deploy with ease using Railway.  [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Deploy directly from the releases. See the [manual deployment guide](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to stay up-to-date with the latest releases and features!
![star gif](https://docs.langbot.app/star.gif)

## Feature Details

For a comprehensive list of features, visit the [features documentation](https://docs.langbot.app/zh/insight/features.html).

**Demo:** Explore the WebUI at https://demo.langbot.dev/
  - Login: Email: `demo@langbot.app`, Password: `langbot123456`
  - *Note:* This is a public environment; do not enter sensitive information.

### Supported Messaging Platforms

| Platform          | Status | Notes                       |
| ----------------- | ------ | --------------------------- |
| QQ Personal       | ✅      | Private & Group Chat      |
| QQ Official Bot   | ✅      | Channels, Private & Group |
| WeChat            | ✅      |                             |
| Enterprise WeChat | ✅      |                             |
| WeChat Official Account | ✅      |                             |
| Feishu            | ✅      |                             |
| DingTalk          | ✅      |                             |
| Discord           | ✅      |                             |
| Telegram          | ✅      |                             |
| Slack             | ✅      |                             |

### Supported LLMs

| Model                | Status | Notes                                          |
| -------------------- | ------ | ---------------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅      | Supports any OpenAI API compatible models |
| [DeepSeek](https://www.deepseek.com/)   | ✅      |                                              |
| [Moonshot](https://www.moonshot.cn/)   | ✅      |                                              |
| [Anthropic](https://www.anthropic.com/) | ✅      |                                              |
| [xAI](https://x.ai/)                 | ✅      |                                              |
| [智谱AI](https://open.bigmodel.cn/)   | ✅      |                                              |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅      | Large Model and GPU Resource Platform |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅      | Large Model and GPU Resource Platform |
| [302.AI](https://share.302.ai/SuTG99) | ✅      | Large Model Aggregation Platform |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅      |                                              |
| [Dify](https://dify.ai)              | ✅      | LLMOps Platform                              |
| [Ollama](https://ollama.com/)          | ✅      | Local LLM runtime                            |
| [LMStudio](https://lmstudio.ai/)       | ✅      | Local LLM runtime                            |
| [GiteeAI](https://ai.gitee.com/)      | ✅      | LLM API Aggregation Platform                 |
| [SiliconFlow](https://siliconflow.cn/)  | ✅      | LLM Aggregation Platform                 |
| [阿里云百炼](https://bailian.console.aliyun.com/)  | ✅      | LLM Aggregation Platform, LLMOps Platform                 |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅      | LLM Aggregation Platform, LLMOps Platform                 |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro)  | ✅      | LLM Aggregation Platform                 |
| [MCP](https://modelcontextprotocol.io/) | ✅      | Supports tool access through MCP protocol   |

### TTS Support

| Platform/Model          | Notes                                          |
| ----------------------- | ---------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/) | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image

| Platform/Model          | Notes                                          |
| ----------------------- | ---------------------------------------------- |
| 阿里云百炼 | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

A big thank you to all the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO optimizations:

*   **Headline:** Clear and concise, including the primary keywords "LangBot" and "chatbot."
*   **One-Sentence Hook:** Immediately introduces the project and its core value proposition.
*   **SEO Keywords:** Strategically used terms like "open-source," "LLM," "chatbot," "AI," "IM bots," "Agent," and platform names.
*   **Clear Headings:** Improves readability and structure for search engines.
*   **Bulleted Lists:** Highlights key features with clear, concise descriptions.
*   **Actionable Language:** Uses verbs like "Empower," "Build," and "Connect."
*   **Internal and External Links:** Keeps the user in the document and offers relevant information.
*   **Concise Summarization:**  Removes redundant information and streamlines the content.
*   **Direct links:** Provides the link to the original repository at the beginning.
*   **Emphasis on Deployment:** Prioritizes the "Getting Started" section.
*   **Platform names:** Added the names of supported platforms to enhance SEO.