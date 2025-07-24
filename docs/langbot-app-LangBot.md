<p align="center">
<a href="https://langbot.app">
<img src="https://docs.langbot.app/social_zh.png" alt="LangBot"/>
</a>

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

</p>

# LangBot: Your Open-Source LLM-Powered Chatbot Platform

**LangBot empowers you to build and deploy intelligent, large language model (LLM)-driven chatbots for various communication platforms.**  For the latest updates, visit the [LangBot GitHub repository](https://github.com/langbot-app/LangBot).

## Key Features

*   **💬 Advanced LLM Capabilities:**  Supports LLM conversations, Agent functionalities, RAG (Retrieval-Augmented Generation) and MCP, including multi-turn dialogues, tool usage, and multimodal capabilities. Seamlessly integrates with [Dify](https://dify.ai).
*   **🤖 Cross-Platform Compatibility:**  Works with popular platforms, including QQ, QQ Channels, WeChat (personal and official accounts), Enterprise WeChat, Feishu, Discord, Telegram, Slack, and DingTalk.
*   **🛠️ Robust & Feature-Rich:**  Provides built-in features like access control, rate limiting, and profanity filtering. Configuration is simple and deployment is flexible. Offers multi-pipeline configuration for different chatbot use cases.
*   **🧩 Extensible with Plugins:**  Supports event-driven and component-based plugin architecture, including Anthropic [MCP Protocol](https://modelcontextprotocol.io/).  Hundreds of plugins are already available.
*   **😻 Web Management Panel:** Manage your LangBot instances via a web UI, eliminating the need for manual configuration file editing.

For a complete list of features, see the [detailed documentation](https://docs.langbot.app/zh/insight/features.html).

Explore a live demo at: https://demo.langbot.dev/  (Login: demo@langbot.app / Password: langbot123456)
*   **Note:** This demo showcases the WebUI. Please avoid entering sensitive information.

## Deployment Options

### Docker Compose

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access the platform at http://localhost:5300.

For more information, see the [Docker deployment documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

### Other Deployment Methods

*   **BaoTa Panel:** Available on the BaoTa Panel. See the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html).
*   **Zeabur Cloud:** Deploy using a community-contributed Zeabur template. [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:** [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:**  Run from the release version. See [manual deployment documentation](https://docs.langbot.app/zh/deploy/langbot/manual.html).

## Stay Updated

Star and watch the repository to receive the latest updates.

![star gif](https://docs.langbot.app/star.gif)

## Supported Platforms

| Platform          | Status | Notes                                   |
| :---------------- | :----- | :-------------------------------------- |
| QQ Personal      | ✅     | Private and group chats               |
| QQ Official Bot  | ✅     | Supports Channels, private, and group chats |
| WeChat            | ✅     |                                         |
| Enterprise WeChat | ✅     |                                         |
| WeChat Official Account | ✅     |                                         |
| Feishu            | ✅     |                                         |
| DingTalk          | ✅     |                                         |
| Discord           | ✅     |                                         |
| Telegram          | ✅     |                                         |
| Slack             | ✅     |                                         |

## Supported LLMs

| Model                                                        | Status | Notes                                                                         |
| :----------------------------------------------------------- | :----- | :---------------------------------------------------------------------------- |
| [OpenAI](https://platform.openai.com/)                     | ✅     | Supports any OpenAI API-compatible model                                       |
| [DeepSeek](https://www.deepseek.com/)                        | ✅     |                                                                               |
| [Moonshot](https://www.moonshot.cn/)                        | ✅     |                                                                               |
| [Anthropic](https://www.anthropic.com/)                      | ✅     |                                                                               |
| [xAI](https://x.ai/)                                         | ✅     |                                                                               |
| [智谱AI](https://open.bigmodel.cn/)                      | ✅     |                                                                               |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     | LLMs and GPU resources platform                                                |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     | LLMs and GPU resources platform                                                |
| [302.AI](https://share.302.ai/SuTG99)                      | ✅     | LLM Aggregator                                                              |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat)        | ✅     |                                                                               |
| [Dify](https://dify.ai)                                        | ✅     | LLMOps Platform                                                              |
| [Ollama](https://ollama.com/)                                 | ✅     | Local LLM Execution Platform                                                  |
| [LMStudio](https://lmstudio.ai/)                             | ✅     | Local LLM Execution Platform                                                  |
| [GiteeAI](https://ai.gitee.com/)                             | ✅     | LLM Interface Aggregator                                                      |
| [SiliconFlow](https://siliconflow.cn/)                        | ✅     | LLM Aggregator                                                              |
| [阿里云百炼](https://bailian.console.aliyun.com/)                 | ✅     | LLM Aggregator, LLMOps Platform                                                 |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅     | LLM Aggregator, LLMOps Platform                                                 |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro)  | ✅     | LLM Aggregator                                                              |
| [MCP](https://modelcontextprotocol.io/)                     | ✅     | Supports tool usage via MCP protocol                                           |

## Text-to-Speech (TTS)

| Platform/Model                            | Notes                                                                |
| :------------------------------------------ | :------------------------------------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)        | [Plugin](https://github.com/the-lazy-me/NewChatVoice)                  |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)       | [Plugin](https://github.com/the-lazy-me/NewChatVoice)                  |
| [AzureTTS](https://portal.azure.com/)                 | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)                |

## Text-to-Image (TTI)

| Platform/Model                  | Notes                                            |
| :-------------------------------- | :----------------------------------------------- |
| 阿里云百炼           | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

## Community Contributions

We are grateful for the contributions from the [code contributors](https://github.com/langbot-app/LangBot/graphs/contributors) and other community members!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>