# LangBot: The Open-Source LLM-Powered Chatbot Platform

**Quickly build your own AI chatbot with LangBot, a versatile and open-source platform designed for easy integration with major messaging platforms.**  [Explore the LangBot project on GitHub](https://github.com/langbot-app/LangBot).

<div align="center">

简体中文 / [English](README_EN.md) / [日本語](README_JP.md) / (PR for your language)

[![Discord](https://img.shields.io/discord/1335141740050649118?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb)](https://discord.gg/wdNEHETs87)
[![QQ Group](https://img.shields.io/badge/%E7%A4%BE%E5%8C%BAQQ%E7%BE%A4-966235608-blue)](https://qm.qq.com/q/JLi38whHum)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/langbot-app/LangBot)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/langbot-app/LangBot)](https://github.com/langbot-app/LangBot/releases/latest)
<img src="https://img.shields.io/badge/python-3.10 ~ 3.13 -blue.svg" alt="python">
[![star](https://gitcode.com/RockChinQ/LangBot/star/badge.svg)](https://gitcode.com/RockChinQ/LangBot)

<a href="https://langbot.app">项目主页</a> ｜
<a href="https://docs.langbot.app/zh/insight/guide.html">部署文档</a> ｜
<a href="https://docs.langbot.app/zh/plugin/plugin-intro.html">插件介绍</a> ｜
<a href="https://github.com/langbot-app/LangBot/issues/new?assignees=&labels=%E7%8B%AC%E7%AB%8B%E6%8F%92%E4%BB%B6&projects=&template=submit-plugin.yml&title=%5BPlugin%5D%3A+%E8%AF%B7%E6%B1%82%E7%99%BB%E8%AE%B0%E6%96%B0%E6%8F%92%E4%BB%B6">提交插件</a>

</div>

## Key Features of LangBot

*   **Advanced LLM Capabilities:**
    *   Supports multiple large language models (LLMs) for conversations, agents, and more.
    *   Offers multi-turn conversations, tool usage, and multimodal functionalities.
    *   Includes built-in RAG (Retrieval-Augmented Generation) capabilities.
    *   Deep integration with [Dify](https://dify.ai).
*   **Broad Platform Support:**
    *   Works with popular messaging platforms, including QQ, QQ Channels, Enterprise WeChat, personal WeChat, Feishu, Discord, Telegram, and more.
*   **Robust and Feature-Rich:**
    *   Provides access control, rate limiting, and sensitive word filtering.
    *   Simple configuration and multiple deployment options.
    *   Supports multi-pipeline configuration for various use cases.
*   **Extensible with Plugins and Community:**
    *   Offers a plugin system for event-driven and component-based extensions.
    *   Compliant with the Anthropic [MCP protocol](https://modelcontextprotocol.io/).
    *   Hundreds of plugins are currently available.
*   **User-Friendly Web Interface:**
    *   Manage LangBot instances directly through a web UI, eliminating the need to manually edit configuration files.

For a comprehensive overview, explore the [detailed feature specifications](https://docs.langbot.app/zh/insight/features.html).

Or, experience the demo environment: https://demo.langbot.dev/
    *   Login: `demo@langbot.app` / Password: `langbot123456`
    *   Note: This is a public demo; please refrain from entering sensitive information.

## Getting Started

### Deploy with Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access the platform at http://localhost:5300.

Detailed instructions are available in the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Options

*   **BaoTa Panel:** Available for deployment via BaoTa Panel, with instructions in the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur:** Deploy using the community-contributed Zeabur template: [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway:** Deploy with the Railway template: [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Refer to the [manual deployment instructions](https://docs.langbot.app/zh/deploy/langbot/manual.html) for direct deployment.

## Stay Updated

Star and watch the repository to receive the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Supported Messaging Platforms

| Platform           | Status | Notes                               |
| ------------------ | ------ | ----------------------------------- |
| QQ Personal Account | ✅     | Private and group chats.          |
| QQ Official Bot    | ✅     | Supports channels, private & group chats. |
| WeChat             | ✅     |                                     |
| Enterprise WeChat  | ✅     |                                     |
| WeChat Official Account | ✅     |                                     |
| Feishu             | ✅     |                                     |
| DingTalk           | ✅     |                                     |
| Discord            | ✅     |                                     |
| Telegram           | ✅     |                                     |
| Slack              | ✅     |                                     |

## Supported LLMs

| Model                      | Status | Notes                                       |
| -------------------------- | ------ | ------------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Compatible with any OpenAI API format models. |
| [DeepSeek](https://www.deepseek.com/) | ✅     |                                             |
| [Moonshot](https://www.moonshot.cn/) | ✅     |                                             |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                             |
| [xAI](https://x.ai/) | ✅     |                                             |
| [ZhipuAI](https://open.bigmodel.cn/) | ✅     |                                             |
| [Uyun Zhisuo](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     | Model and GPU resource platform             |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | Model and GPU resource platform             |
| [302.AI](https://share.302.ai/SuTG99) | ✅     | Model aggregation platform                   |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     |                                             |
| [Dify](https://dify.ai)    | ✅     | LLMOps platform                            |
| [Ollama](https://ollama.com/)   | ✅     | Local model platform                      |
| [LMStudio](https://lmstudio.ai/)   | ✅     | Local model platform                      |
| [GiteeAI](https://ai.gitee.com/) | ✅     | LLM API aggregation                        |
| [SiliconFlow](https://siliconflow.cn/) | ✅     | LLM aggregation platform                        |
| [Aliyun Baichuan](https://bailian.console.aliyun.com/) | ✅ | LLM aggregation platform, LLMOps platform         |
| [Volcano Ark](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅ | LLM aggregation platform, LLMOps platform         |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅ | LLM aggregation platform         |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports tools via the MCP protocol.       |

## Text-to-Speech (TTS) Integrations

| Platform/Model              | Notes                                  |
| --------------------------- | -------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [HaiTun AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)    | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image (TTI) Integrations

| Platform/Model        | Notes                                           |
| --------------------- | ----------------------------------------------- |
| Aliyun Baichuan        | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

A big thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and all community members for their invaluable contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO optimizations:

*   **Clear Title and Hook:** The title is more descriptive and includes a compelling hook to grab attention.
*   **SEO-Friendly Headings:** Uses clear, descriptive headings (e.g., "Key Features of LangBot", "Getting Started") to structure the content.
*   **Bulleted Lists:** Uses bullet points to highlight key features, making the information easier to scan.
*   **Keyword Optimization:**  Incorporates relevant keywords like "AI chatbot," "open-source," "LLM," "messaging platforms," and platform names to improve search visibility.
*   **Concise Summaries:** Provides brief summaries of features to maintain reader interest.
*   **Call to Action:** Includes clear calls to action (e.g., "Explore the LangBot project on GitHub").
*   **Internal Linking:** Links to important sections of the original README and external resources to guide users.
*   **Improved Formatting:** Enhances readability with better spacing and formatting.
*   **Complete and Accurate:** Includes all the original content, presented in a more organized manner.
*   **Focus on Benefits:** Highlights the benefits of using LangBot (e.g., easy chatbot creation, platform support).
*   **Community Emphasis:** Promotes the community aspect.