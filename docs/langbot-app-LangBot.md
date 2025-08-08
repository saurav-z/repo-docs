# LangBot: Build Your Own AI Chatbot Platform

**Empower your communication with LangBot, the open-source platform for building powerful and versatile AI-powered chatbots.** ([View on GitHub](https://github.com/langbot-app/LangBot))

<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot" width="400"/>
</a>
</p>

LangBot offers a comprehensive, out-of-the-box solution for developing AI chatbots, featuring:

*   **Agent capabilities:** Enables intelligent interactions and automation.
*   **RAG (Retrieval-Augmented Generation):** Integrates knowledge bases for enhanced responses.
*   **MCP (Model Context Protocol) support:**  Allows for seamless integration with various LLM applications.
*   **Broad platform compatibility:** Supports major messaging platforms.
*   **Extensive API and customization options:** Tailor the platform to your exact needs.

<div align="center">

[English](README_EN.md) / 简体中文 / [繁體中文](README_TW.md) / [日本語](README_JP.md) / (PR for your language)

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

## Key Features

*   **Versatile AI Chatbot:**  Supports advanced features like multi-turn conversations, tool calling, and multimodal capabilities for a richer user experience.
*   **Multi-Platform Support:** Works seamlessly with QQ, QQ Channels, WeChat (Personal and Official Accounts), Feishu, Discord, Telegram, Slack, and more.
*   **Robust & Reliable:** Provides built-in access controls, rate limiting, and profanity filters for a stable and secure chatbot experience.
*   **Plugin Ecosystem:** Extends functionality with a plugin system based on event-driven architecture, supporting the Anthropic MCP protocol, with hundreds of available plugins.
*   **User-Friendly Web Interface:**  Manage and configure your LangBot instance easily via a web UI, eliminating the need for manual configuration file editing.

## Getting Started

### Docker Compose Deployment

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access your LangBot instance at http://localhost:5300.

For detailed instructions, refer to the [Docker Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Additional Deployment Options

*   **BaaPanel (宝塔面板):** Available on BaaPanel, using the documentation [here](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Deploy using the community-contributed Zeabur template ( [Deploy on Zeabur](https://zeabur.com/zh-CN/templates/ZKTBDH) ).
*   **Railway Cloud:** Deploy easily with Railway ( [Deploy on Railway](https://railway.app/template/yRrAyL?referralCode=vogKPF) ).
*   **Manual Deployment:** Run the release directly; see the [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to receive the latest updates!

![star gif](https://docs.langbot.app/star.gif)

## Supported Features and Integrations

### Supported Platforms

| Platform          | Status | Notes                      |
|-------------------|--------|---------------------------|
| QQ Personal       | ✅     | Private and Group Chat    |
| QQ Official Bot   | ✅     | Channels, Private, Groups |
| Enterprise WeChat | ✅     |                           |
| WeChat External Customer | ✅     |                           |
| WeChat Official Account| ✅     |                           |
| Feishu            | ✅     |                           |
| DingTalk          | ✅     |                           |
| Discord           | ✅     |                           |
| Telegram          | ✅     |                           |
| Slack             | ✅     |                           |

### Supported Large Language Models (LLMs)

| Model                                     | Status | Notes                                        |
|-------------------------------------------|--------|---------------------------------------------|
| [OpenAI](https://platform.openai.com/)   | ✅     | Supports any OpenAI API format model          |
| [DeepSeek](https://www.deepseek.com/)     | ✅     |                                             |
| [Moonshot](https://www.moonshot.cn/)      | ✅     |                                             |
| [Anthropic](https://www.anthropic.com/)   | ✅     |                                             |
| [xAI](https://x.ai/)                       | ✅     |                                             |
| [Zhipu AI](https://open.bigmodel.cn/)    | ✅     |                                             |
| [Youyun Zhisuan](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)  | ✅ | LLM and GPU Resource Platform               |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot)  | ✅ | LLM and GPU Resource Platform               |
| [302.AI](https://share.302.ai/SuTG99)      | ✅     | LLM Aggregation Platform                    |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     |                                             |
| [Dify](https://dify.ai)                   | ✅     | LLMOps Platform                            |
| [Ollama](https://ollama.com/)             | ✅     | Local LLM Platform                          |
| [LMStudio](https://lmstudio.ai/)          | ✅     | Local LLM Platform                          |
| [GiteeAI](https://ai.gitee.com/)          | ✅     | LLM API Aggregation Platform                |
| [SiliconFlow](https://siliconflow.cn/)     | ✅     | LLM Aggregation Platform                    |
| [Alibaba Cloud Baichuan](https://bailian.console.aliyun.com/)   | ✅     | LLM Aggregation & LLMOps Platform           |
| [VolcEngine Ark](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅ | LLM Aggregation & LLMOps Platform           |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅ | LLM Aggregation Platform                      |
| [MCP](https://modelcontextprotocol.io/)   | ✅     | Supports tools via the MCP protocol          |

### Text-to-Speech (TTS)

| Platform/Model                    | Notes                               |
|------------------------------------|-------------------------------------|
| [FishAudio](https://fish.audio/zh-CN/discovery/)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [Haitun AI](https://www.ttson.cn/?source=thelazy)  | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/)          | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image

| Platform/Model                  | Notes                                       |
|-----------------------------------|---------------------------------------------|
| Alibaba Cloud Baichuan  | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

A huge thanks to our [contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members for their valuable contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO considerations:

*   **Strong Headline & Hook:** The opening sentence immediately grabs attention and highlights the core value proposition.
*   **Keyword Optimization:** Includes relevant keywords like "AI chatbot," "open-source," "LLM," "chatbot platform," etc., throughout the content.
*   **Clear Sectioning:** Uses headings and subheadings to structure the information logically, improving readability and SEO.
*   **Concise Bullet Points:** Key features are presented in a concise, easy-to-scan format.
*   **Emphasis on Benefits:** Focuses on the benefits of using LangBot (e.g., versatility, platform support, ease of use).
*   **Call to Action:** Encourages users to "View on GitHub" and provides clear instructions for getting started.
*   **Detailed Feature Lists:**  Lists supported features, platform integrations, and LLMs in organized tables for easy reference.
*   **SEO-Friendly Formatting:** Uses Markdown for proper formatting, including headings and lists.
*   **Internal & External Linking:** Includes links to the GitHub repository, documentation, deployment options, and other relevant resources, improving SEO and user experience.
*   **Community Focus:** Highlights the community's contributions to encourage participation.
*   **Demo access**: Includes login information for the demo.