<div align="center">
  <a href="https://langbot.app">
    <img src="https://docs.langbot.app/social_zh.png" alt="LangBot" width="400"/>
  </a>
</div>

# LangBot: Your Open-Source LLM-Powered Chatbot Platform

**Empower your instant messaging with LangBot, a versatile and open-source platform for building AI-powered chatbots compatible with major communication platforms.** ([Back to the original repo](https://github.com/langbot-app/LangBot))

<div align="center">
  <a href="https://hellogithub.com/repository/langbot-app/LangBot" target="_blank">
    <img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=5ce8ae2aa4f74316bf393b57b952433c&claim_uid=gtmc6YWjMZkT21R" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" />
  </a>
</div>

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

---

## Key Features of LangBot:

*   **Versatile LLM Capabilities:** Leverage multiple large language models (LLMs) with agent functionalities, RAG (Retrieval-Augmented Generation), and MCP support.
*   **Extensive Platform Compatibility:** Integrate seamlessly with popular platforms like QQ, WeChat, Feishu, Discord, Telegram, and more.
*   **Robust and Configurable:** Enjoy stability features such as access control, rate limiting, and profanity filtering. Easily configure your setup with multiple deployment options.
*   **Plugin Ecosystem:** Extend functionality with a rich plugin system supporting event-driven architecture and components. Support Anthropic MCP protocols.
*   **Web-Based Management:** Utilize the web UI for effortless LangBot instance management.

---

## Getting Started

### Deployment Options:

*   **Docker Compose:**
    ```bash
    git clone https://github.com/langbot-app/LangBot
    cd LangBot
    docker compose up -d
    ```
    Access at http://localhost:5300. [Docker Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

*   **BaoTa Panel Deployment:**  Available on BaoTa Panel (see [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html)).

*   **Zeabur Cloud Deployment:**  Community-contributed Zeabur template.
    [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)

*   **Railway Cloud Deployment:**
    [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)

*   **Manual Deployment:**  Run using the releases. See [Manual Deployment Documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

---

## Stay Updated

Star and Watch this repository to receive the latest updates and insights.

![star gif](https://docs.langbot.app/star.gif)

---

## Core Functionality

### Supported Platforms:

| Platform          | Status | Notes                        |
| ----------------- | ------ | ---------------------------- |
| QQ (Personal)     | ✅     | Private and group chats      |
| QQ (Official Bot) | ✅     | Channels, private, & group chats |
| WeChat Work       | ✅     |                              |
| WeChat External    | ✅     |                              |
| WeChat            | ✅     |                              |
| WeChat Official   | ✅     |                              |
| Feishu            | ✅     |                              |
| DingTalk          | ✅     |                              |
| Discord           | ✅     |                              |
| Telegram          | ✅     |                              |
| Slack             | ✅     |                              |

### Supported LLMs:

| Model                              | Status | Notes                                      |
| ---------------------------------- | ------ | ------------------------------------------ |
| OpenAI                             | ✅     | Supports all OpenAI API format models      |
| DeepSeek                           | ✅     |                                            |
| Moonshot                           | ✅     |                                            |
| Anthropic                          | ✅     |                                            |
| xAI                                | ✅     |                                            |
| ZhipuAI                            | ✅     |                                            |
| Shengsuanyun (Recommend) | ✅     | Access global models                          |
| Youyun Zhisuan                     | ✅     | GPU resources platform                     |
| PPIO                               | ✅     | GPU resources platform                     |
| 302.AI                             | ✅     | LLM aggregation platform                   |
| Google Gemini                      | ✅     |                                            |
| Dify                               | ✅     | LLMOps platform                            |
| Ollama                             | ✅     | Local LLM platform                         |
| LMStudio                           | ✅     | Local LLM platform                         |
| GiteeAI                            | ✅     | LLM API aggregation platform               |
| SiliconFlow                        | ✅     | LLM aggregation platform                   |
| Alibaba Bailian                   | ✅     | LLM & LLMOps platform                        |
| Volcengine Ark                     | ✅     | LLM & LLMOps platform                       |
| ModelScope                         | ✅     | LLM aggregation platform                   |
| MCP                                | ✅     | Supports MCP (Model Context Protocol) tools |

### Text-to-Speech (TTS)

| Platform/Model | Notes                                  |
| ------------- | -------------------------------------- |
| FishAudio     | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| Haitun AI      | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| AzureTTS      | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

### Text-to-Image (TTI)

| Platform/Model | Notes                                       |
| ------------- | ------------------------------------------- |
| Alibaba Bailian | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

---

## Contributions

A special thanks to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and the community for their ongoing support and contributions to LangBot!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>