<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot"/>
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

# LangBot: Your Open-Source AI Chatbot Development Platform 🤖

LangBot is a powerful, open-source platform designed to make building AI-powered chatbots for various messaging platforms a breeze, featuring Agent, RAG, MCP functionality, and extensive customization options.  [Explore the LangBot Repository](https://github.com/langbot-app/LangBot).

## Key Features

*   **AI-Powered Conversation:**  Engage in dynamic conversations with support for multiple large language models (LLMs), perfect for both group and private chats, with multi-turn conversations, tool use, and multimodal capabilities, including built-in RAG (Retrieval-Augmented Generation) and Dify integration.
*   **Cross-Platform Compatibility:** Seamlessly integrate with popular messaging platforms including QQ, QQ Channels, WeChat Work, personal WeChat, Lark, Discord, Telegram, and more.
*   **Robust & Feature-Rich:** Benefit from built-in access controls, rate limiting, and profanity filters. Enjoy easy configuration and a variety of deployment options. Supports multiple pipeline configurations for different chatbot applications.
*   **Extensible with Plugins & Active Community:**  Extend functionality through event-driven and component-based plugin architecture, supporting the Anthropic [MCP protocol](https://modelcontextprotocol.io/) with hundreds of available plugins.
*   **Intuitive Web Interface:** Manage your LangBot instance easily through a web-based UI, eliminating the need for manual configuration file editing.

## Getting Started

### Deployment Options

*   **Docker Compose:**

    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    docker compose up -d
    ```
    Access at:  `http://localhost:5300`

    Detailed instructions are available in the [Docker Deployment Guide](https://docs.langbot.app/zh/deploy/langbot/docker.html).
*   **BaoTa Panel:**  Deploy with a single click via the BaoTa Panel.  See the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html) for instructions.
*   **Zeabur Cloud:**  Utilize the community-contributed Zeabur template. [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:**  Deploy on Railway.  [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Run from releases, instructions at [Manual Deployment](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to stay informed about the latest updates!

![star gif](https://docs.langbot.app/star.gif)

## Demo

Experience the Web UI: [https://demo.langbot.dev/](https://demo.langbot.dev/)
* Login Info: Email: `demo@langbot.app` Password: `langbot123456`

## Platform Support

| Platform        | Status | Notes                      |
| --------------- | ------ | -------------------------- |
| QQ (Personal)   | ✅     | Private & Group Chats      |
| QQ (Official Bot) | ✅     | Channels, Private & Group |
| WeChat Work     | ✅     |                             |
| WeChat Public Account | ✅ |                             |
| Lark              | ✅     |                             |
| DingTalk          | ✅     |                             |
| Discord         | ✅     |                             |
| Telegram        | ✅     |                             |
| Slack | ✅ |       |

## LLM Integration

| Model                                          | Status | Notes                                             |
| ---------------------------------------------- | ------ | ------------------------------------------------- |
| [OpenAI](https://platform.openai.com/)         | ✅     | Supports any OpenAI API-compatible models          |
| [DeepSeek](https://www.deepseek.com/)          | ✅     |                                                   |
| [Moonshot](https://www.moonshot.cn/)            | ✅     |                                                   |
| [Anthropic](https://www.anthropic.com/)        | ✅     |                                                   |
| [xAI](https://x.ai/)                          | ✅     |                                                   |
| [ZhipuAI](https://open.bigmodel.cn/)           | ✅     |                                                   |
| [Youyun Zhisuan](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot)   | ✅   | LLM and GPU Resource Platform |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot)   | ✅   | LLM and GPU Resource Platform |
| [302.AI](https://share.302.ai/SuTG99)      | ✅     | LLM Aggregation Platform                             |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat)         | ✅     | |
| [Dify](https://dify.ai)                        | ✅     | LLMOps Platform                                   |
| [Ollama](https://ollama.com/)                  | ✅     | Local LLM Runner                                  |
| [LMStudio](https://lmstudio.ai/)               | ✅     | Local LLM Runner                                  |
| [GiteeAI](https://ai.gitee.com/)           | ✅     | LLM API Aggregation Platform                             |
| [SiliconFlow](https://siliconflow.cn/)          | ✅     | LLM Aggregation Platform |
| [Aliyun Bailian](https://bailian.console.aliyun.com/)          | ✅     | LLM Aggregation Platform, LLMOps Platform|
| [Volcano Ark](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW)   | ✅   | LLM Aggregation Platform, LLMOps Platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro)     | ✅     | LLM Aggregation Platform                                  |
| [MCP](https://modelcontextprotocol.io/)       | ✅     | Supports tool usage via MCP protocol               |

## TTS (Text-to-Speech)

| Platform/Model                                       | Notes                                                   |
| ---------------------------------------------------- | ------------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)    | [Plugin](https://github.com/the-lazy-me/NewChatVoice)  |
| [Haitun AI](https://www.ttson.cn/?source=thelazy)   | [Plugin](https://github.com/the-lazy-me/NewChatVoice)  |
| [AzureTTS](https://portal.azure.com/)                | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image

| Platform/Model | Notes                                               |
| -------------- | --------------------------------------------------- |
| Aliyun Bailian | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

A huge thank you to the [contributors](https://github.com/langbot-app/LangBot/graphs/contributors) for their amazing work!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>
```
Key improvements and SEO enhancements:

*   **Clear and Concise Hook:** Immediately establishes the core value proposition.
*   **Optimized Headings:**  Uses H1 and H2 headings for better structure and SEO.
*   **Keyword Integration:** Uses relevant keywords like "AI chatbot," "open-source," "LLM," and platform names.
*   **Bulleted Feature List:**  Provides an easy-to-scan overview of key features.
*   **Structured Information:**  Organized content for readability and clarity.
*   **Direct Links to Documentation:** Includes links to crucial resources.
*   **Call to Action (CTA):** Encourages users to get started.
*   **Alt Text:**  Includes relevant alt text for images to improve accessibility and SEO.
*   **Mobile-Friendly:**  Uses a responsive design to look good on any device.
*   **Concise Language:**  Uses clear and direct language.
*   **Removed Redundancy:** Streamlined the README by removing unnecessary details.
*   **Focus on Value:** Highlights what the user can *do* with LangBot.