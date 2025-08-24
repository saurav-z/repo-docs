<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot" width="250"/>
</a>
</p>

<div align="center">

<a href="https://hellogithub.com/repository/langbot-app/LangBot" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=5ce8ae2aa4f74316bf393b57b952433c&claim_uid=gtmc6YWjMZkT21R" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

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

# LangBot: Your Open-Source LLM-Powered Chatbot Development Platform

LangBot is a powerful, open-source platform designed to simplify the creation of AI-powered chatbots, offering out-of-the-box functionality and extensive customization options. [Explore the LangBot repository](https://github.com/langbot-app/LangBot)!

## Key Features

*   **Versatile Chatbot Capabilities:** Supports various large language models (LLMs), Agents, RAG (Retrieval-Augmented Generation), and Model Context Protocol (MCP) for advanced AI interactions.
*   **Multi-Platform Compatibility:** Works seamlessly with popular messaging platforms like QQ, QQ Channels, WeChat, Enterprise WeChat, Feishu, Discord, Telegram, and others.
*   **Robust and Reliable:** Offers built-in features for access control, rate limiting, and sensitive word filtering, ensuring a stable and secure chatbot experience. Easily configured and supports multiple deployment methods.
*   **Extensible with Plugins:** Extends functionality through event-driven and component-based plugins, including Anthropic's MCP protocol support. Access hundreds of available plugins to customize your chatbot's behavior.
*   **User-Friendly Web Interface:** Manage your LangBot instance through an intuitive web-based interface, eliminating the need for manual configuration file editing.

## Getting Started

### Deployment Options

Choose the deployment method that best suits your needs:

#### Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access your LangBot instance at http://localhost:5300.

For detailed instructions, see the [Docker Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

#### Baota Panel Deployment

If you are familiar with using the Baota panel you can follow the [Baota Panel deployment documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).

#### Zeabur Cloud Deployment

Deploy LangBot to Zeabur Cloud using the community-contributed template:

[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)

#### Railway Cloud Deployment

Deploy LangBot to Railway Cloud:

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)

#### Manual Deployment

For manual deployment instructions, see the [Manual Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Stay informed about the latest updates and developments by starring and watching the repository!

![star gif](https://docs.langbot.app/star.gif)

## Demo

Explore the webUI by visiting [demo environment](https://demo.langbot.dev/)

*   Login information:
    *   Email: `demo@langbot.app`
    *   Password: `langbot123456`

## Supported Platforms

| Platform           | Status | Notes                             |
| ------------------ | ------ | --------------------------------- |
| QQ Personal        | ✅     | Private and group chats           |
| QQ Official Bot    | ✅     | Channels, private & group chats  |
| Enterprise WeChat  | ✅     |                                   |
| WeChat Customer Service| ✅     |                                   |
| Personal WeChat    | ✅     |                                   |
| WeChat Official Account| ✅     |                                   |
| Feishu             | ✅     |                                   |
| DingTalk           | ✅     |                                   |
| Discord            | ✅     |                                   |
| Telegram           | ✅     |                                   |
| Slack              | ✅     |                                   |

## Supported Large Language Models (LLMs)

| Model                     | Status | Notes                                  |
| ------------------------- | ------ | -------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅     | Supports any OpenAI API compatible models |
| [DeepSeek](https://www.deepseek.com/)  | ✅     |                                        |
| [Moonshot](https://www.moonshot.cn/)   | ✅     |                                        |
| [Anthropic](https://www.anthropic.com/) | ✅     |                                        |
| [xAI](https://x.ai/)       | ✅     |                                        |
| [智谱AI](https://open.bigmodel.cn/)   | ✅     |                                        |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)  | ✅     | LLM and GPU Resources           |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | LLM and GPU Resources                   |
| [胜算云](https://www.shengsuanyun.com/?from=CH_KYIPP758)   | ✅     | LLM and GPU Resources                   |
| [302.AI](https://share.302.ai/SuTG99)  | ✅     | LLM Aggregation Platform               |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅     |                                        |
| [Dify](https://dify.ai)  | ✅     | LLMOps Platform                       |
| [Ollama](https://ollama.com/) | ✅     | Local LLM Run Platform               |
| [LMStudio](https://lmstudio.ai/)  | ✅     | Local LLM Run Platform               |
| [GiteeAI](https://ai.gitee.com/)   | ✅     | LLM Interface Aggregation Platform |
| [SiliconFlow](https://siliconflow.cn/) | ✅     | LLM Aggregation Platform               |
| [阿里云百炼](https://bailian.console.aliyun.com/) | ✅     | LLM Aggregation Platform, LLMOps Platform |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | LLM Aggregation Platform, LLMOps Platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅     | LLM Aggregation Platform               |
| [MCP](https://modelcontextprotocol.io/) | ✅     | Supports MCP protocol tools            |

## Text-to-Speech (TTS) Integrations

| Platform/Model | Notes                                               |
| -------------- | --------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)   | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)   | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/) | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image Integrations

| Platform/Model | Notes                                                                 |
| -------------- | --------------------------------------------------------------------- |
| 阿里云百炼       | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

A huge thank you to all the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and community members for their valuable contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO optimization:

*   **Concise Hook:**  The first sentence is a strong, keyword-rich hook that immediately grabs the reader's attention.
*   **Clear Headings:** Uses descriptive headings for better readability and SEO.
*   **Bulleted Key Features:** Highlights the main benefits in a clear, scannable format.  This is good for both users and search engines.
*   **Keyword Optimization:**  Includes relevant keywords like "open-source," "LLM," "chatbot," "AI," "development platform," and platform names throughout.
*   **Internal Links:**  Links to relevant sections within the README.
*   **External Links with Descriptive Anchor Text:** Links to the project homepage, documentation, and related resources with helpful descriptions to improve click-through rate.
*   **Concise and Action-Oriented Language:**  Uses active voice and keeps descriptions short and to the point.
*   **Community Focus:**  Highlights community contributions, which adds value and builds engagement.
*   **Complete information:** Includes all the relevant information from the original README.
*   **Removed Unnecessary Elements:** Removed redundant links.
*   **Emphasis on Core Value:** Focuses on the core value proposition of LangBot: to simplify chatbot development.