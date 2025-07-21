# LangBot: Your Open-Source IM Bot Platform Powered by LLMs

**LangBot is an open-source platform for building intelligent, Large Language Model (LLM)-powered chatbots for instant messaging, making it easy to integrate AI into your favorite communication platforms.** Learn more on the original repository: [https://github.com/langbot-app/LangBot](https://github.com/langbot-app/LangBot)

<div align="center">
  <a href="https://langbot.app">
    <img src="https://docs.langbot.app/social_zh.png" alt="LangBot"/>
  </a>

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

## Key Features

*   **Versatile LLM Capabilities:** Supports diverse large language models and features like Agents, RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) for advanced conversational AI.
*   **Multi-Platform Support:** Works seamlessly with a wide array of messaging platforms, including QQ, QQ Channels, WeChat, Feishu, Discord, and Telegram.
*   **Robust & Feature-Rich:**  Offers built-in access control, rate limiting, and profanity filtering for stability and control. Easily configurable with various deployment options.
*   **Extensible with Plugins:**  Supports event-driven and component-based plugin architectures, including Anthropic's MCP protocol, and has a community-driven collection of plugins.
*   **Web-Based Management:**  Manage your LangBot instance with an intuitive web UI, eliminating the need for manual configuration.

Find more details about the features: [https://docs.langbot.app/zh/insight/features.html](https://docs.langbot.app/zh/insight/features.html).

## Getting Started

### Deployment Options

#### Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access your LangBot instance at http://localhost:5300.

Learn more: [Docker Deployment Guide](https://docs.langbot.app/zh/deploy/langbot/docker.html).

#### Other Deployment Options:

*   **BaoTa Panel:** Deploy with a single click if you've installed BaoTa Panel. See [deployment documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Deploy using a Zeabur template provided by the community.
    [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:** Deploy on Railway with a pre-built template.
    [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:**  Run the release version directly. Check the [manual deployment guide](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to stay informed about the latest updates:

![star gif](https://docs.langbot.app/star.gif)

## Live Demo

Explore the WebUI at: https://demo.langbot.dev/

*   Login:  Email: `demo@langbot.app`, Password: `langbot123456`
*   Note: This is a public environment. Please refrain from entering any sensitive information.

## Supported Platforms

| Platform            | Status | Notes                                    |
| ------------------- | ------ | ---------------------------------------- |
| QQ Personal         | ✅      | QQ personal chat and group chats         |
| QQ Official Bot     | ✅      | QQ official bots, supports channels  |
| WeChat Enterprise   | ✅      |                                          |
| WeChat Enterprise Customer Service | ✅ |                                          |
| WeChat Personal     | ✅      |                                          |
| WeChat Official Account | ✅ |                                         |
| Feishu             | ✅      |                                          |
| DingTalk | ✅ |                                          |
| Discord             | ✅      |                                          |
| Telegram          | ✅      |                                          |
| Slack               | ✅      |                                          |

## Supported LLMs

| Model                          | Status | Notes                                  |
| ------------------------------ | ------ | -------------------------------------- |
| [OpenAI](https://platform.openai.com/) | ✅      | Supports any OpenAI API format models |
| [DeepSeek](https://www.deepseek.com/) | ✅      |                                        |
| [Moonshot](https://www.moonshot.cn/)  | ✅      |                                        |
| [Anthropic](https://www.anthropic.com/) | ✅      |                                        |
| [xAI](https://x.ai/)        | ✅      |                                        |
| [ZhipuAI](https://open.bigmodel.cn/)   | ✅      |                                        |
| [Youyun Zhisuan](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅ | Large model and GPU resource platform |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot)  | ✅ | Large model and GPU resource platform |
| [302.AI](https://share.302.ai/SuTG99) | ✅ | Large model aggregator platform |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅ | |
| [Dify](https://dify.ai)             | ✅      | LLMOps platform                        |
| [Ollama](https://ollama.com/)        | ✅      | Local Large Model Platform             |
| [LMStudio](https://lmstudio.ai/)      | ✅      | Local Large Model Platform             |
| [GiteeAI](https://ai.gitee.com/)    | ✅      | Large model API aggregator platform   |
| [SiliconFlow](https://siliconflow.cn/) | ✅      | Large model aggregator platform |
| [Alibaba Cloud Baichuan](https://bailian.console.aliyun.com/) | ✅      | Large model aggregator platform, LLMOps platform |
| [Volcengine Ark](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅      | Large model aggregator platform, LLMOps platform |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅      | Large model aggregator platform |
| [MCP](https://modelcontextprotocol.io/) | ✅      | Supports tools via MCP protocol          |

## Text-to-Speech (TTS)

| Platform/Model            | Notes                                                   |
| -------------------------- | ------------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [Haitun AI](https://www.ttson.cn/?source=thelazy) | [Plugin](https://github.com/the-lazy-me/NewChatVoice) |
| [AzureTTS](https://portal.azure.com/) | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS) |

## Text-to-Image

| Platform/Model | Notes                                     |
| ---------------- | ----------------------------------------- |
| Alibaba Cloud Baichuan | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

A special thank you to the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and other community members for their valuable contributions to LangBot:

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>