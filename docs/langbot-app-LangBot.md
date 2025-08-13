<p align="center">
  <a href="https://langbot.app">
    <img src="https://docs.langbot.app/social_zh.png" alt="LangBot" />
  </a>
</p>

<div align="center">

  <a href="https://hellogithub.com/repository/langbot-app/LangBot" target="_blank">
    <img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=5ce8ae2aa4f74316bf393b57b952433c&claim_uid=gtmc6YWjMZkT21R" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" />
  </a>

  [English](README_EN.md) / 简体中文 / [繁體中文](README_TW.md) / [日本語](README_JP.md) / (PR for your language)

  [![Discord](https://img.shields.io/discord/1335141740050649118?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb)](https://discord.gg/wdNEHETs87)
  [![QQ Group](https://img.shields.io/badge/%E7%A4%BE%E5%8C%BAQQ%E7%BE%A4-966235608-blue)](https://qm.qq.com/q/JLi38whHum)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/langbot-app/LangBot)
  [![GitHub release (latest by date)](https://img.shields.io/github/v/release/langbot-app/LangBot)](https://github.com/langbot-app/LangBot/releases/latest)
  <img src="https://img.shields.io/badge/python-3.10 ~ 3.13 -blue.svg" alt="python">
  [![star](https://gitcode.com/RockChinQ/LangBot/star/badge.svg)](https://gitcode.com/RockChinQ/LangBot)

  <a href="https://langbot.app">项目主页</a> |
  <a href="https://docs.langbot.app/zh/insight/guide.html">部署文档</a> |
  <a href="https://docs.langbot.app/zh/plugin/plugin-intro.html">插件介绍</a> |
  <a href="https://github.com/langbot-app/LangBot/issues/new?assignees=&labels=%E7%8B%AC%E7%AB%8B%E6%8F%92%E4%BB%B6&projects=&template=submit-plugin.yml&title=%5BPlugin%5D%3A+%E8%AF%B7%E6%B1%82%E7%99%BB%E8%AE%B0%E6%96%B0%E6%8F%92%E4%BB%B6">提交插件</a>
</div>

## LangBot: Build Your Own AI Chatbot with Ease

LangBot is an open-source platform empowering you to create and deploy intelligent, versatile AI chatbots.

**Key Features:**

*   **🤖 Comprehensive LLM Integration:** Supports a wide array of Large Language Models (LLMs) including OpenAI, DeepSeek, Moonshot, and more, providing flexible options for your chatbot's intelligence.
*   **💬 Multi-Platform Compatibility:** Works seamlessly with popular messaging platforms like QQ, WeChat, Discord, Telegram, and others.
*   **🛠️ Robust & Feature-Rich:** Offers essential features like access control, rate limiting, and content filtering, along with a simple configuration process and diverse deployment options.
*   **🧩 Extensible with Plugins:** Supports a plugin architecture for expanding functionality, including Anthropic's MCP protocol, with hundreds of available plugins.
*   **😻 Web-Based Management:** Simplifies chatbot management with a user-friendly web interface, eliminating the need for manual configuration file editing.
*   **📦 Easy Deployment:**  Deploy quickly using Docker Compose, Zeabur, Railway, or manual installation.

### Getting Started

**Quick Deployment with Docker Compose:**

```bash
git clone https://github.com/langbot-app/LangBot
cd LangBot
docker compose up -d
```

Access LangBot at `http://localhost:5300`.  For detailed deployment instructions, see the [official documentation](https://docs.langbot.app/zh/deploy/langbot/docker.html).

**Other Deployment Options:**

*   **宝塔面板部署:** Available on the Baota Panel.  See the [documentation](https://docs.langbot.app/zh/deploy/langbot/one-click/bt.html) for instructions.
*   **Zeabur Cloud:** Community-contributed Zeabur template.  [![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/zh-CN/templates/ZKTBDH)
*   **Railway Cloud:**  [![Deploy on Railway](https://railway.com/button.svg)](https://railway.app/template/yRrAyL?referralCode=vogKPF)
*   **Manual Deployment:** Use the [manual deployment](https://docs.langbot.app/zh/deploy/langbot/manual.html) instructions.

### Stay Updated

Star and watch the repository to receive updates.

![star gif](https://docs.langbot.app/star.gif)

### Available Integrations

| Platform          | Status | Notes                                 |
| ----------------- | ------ | ------------------------------------- |
| QQ Personal       | ✅     | QQ personal messaging, group chats   |
| QQ Official Bot   | ✅     | QQ official bot, channels, private chat |
| Enterprise WeChat | ✅     |                                         |
| Enterprise WeChat External | ✅ |                                         |
| WeChat Personal   | ✅     |                                         |
| WeChat Official Account | ✅ |                                         |
| Feishu            | ✅     |                                         |
| DingTalk          | ✅     |                                         |
| Discord           | ✅     |                                         |
| Telegram          | ✅     |                                         |
| Slack             | ✅     |                                         |

### Large Language Model Support

| Model                     | Status | Notes                                      |
| ------------------------- | ------ | ------------------------------------------ |
| [OpenAI](https://platform.openai.com/)       | ✅     | Supports any OpenAI API compatible models      |
| [DeepSeek](https://www.deepseek.com/)          | ✅     |                                              |
| [Moonshot](https://www.moonshot.cn/)          | ✅     |                                              |
| [Anthropic](https://www.anthropic.com/)        | ✅     |                                              |
| [xAI](https://x.ai/)           | ✅     |                                              |
| [智谱AI](https://open.bigmodel.cn/)           | ✅     |                                              |
| [优云智算](https://www.compshare.cn/?ytag=GPU_YY-gh_langbot) | ✅     |                                              |
| [PPIO](https://ppinfra.com/user/register?invited_by=QJKFYD&utm_source=github_langbot) | ✅     |                                              |
| [302.AI](https://share.302.ai/SuTG99)       | ✅     |                                              |
| [Google Gemini](https://aistudio.google.com/prompts/new_chat) | ✅ |                                              |
| [Dify](https://dify.ai) | ✅ | LLMOps 平台 |
| [Ollama](https://ollama.com/) | ✅ | 本地大模型运行平台 |
| [LMStudio](https://lmstudio.ai/) | ✅ | 本地大模型运行平台 |
| [GiteeAI](https://ai.gitee.com/) | ✅ | 大模型接口聚合平台 |
| [SiliconFlow](https://siliconflow.cn/) | ✅ | 大模型聚合平台 |
| [阿里云百炼](https://bailian.console.aliyun.com/) | ✅ | 大模型聚合平台, LLMOps 平台 |
| [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=LIST_VIEW) | ✅ | 大模型聚合平台, LLMOps 平台 |
| [ModelScope](https://modelscope.cn/docs/model-service/API-Inference/intro) | ✅ | 大模型聚合平台 |
| [MCP](https://modelcontextprotocol.io/) | ✅ | 支持通过 MCP 协议获取工具 |

### Text-to-Speech (TTS)

| Platform/Model                    | Notes                                   |
| --------------------------------- | --------------------------------------- |
| [FishAudio](https://fish.audio/zh-CN/discovery/)      | [Plugin](https://github.com/the-lazy-me/NewChatVoice)      |
| [海豚 AI](https://www.ttson.cn/?source=thelazy)       | [Plugin](https://github.com/the-lazy-me/NewChatVoice)      |
| [AzureTTS](https://portal.azure.com/)               | [Plugin](https://github.com/Ingnaryk/LangBot_AzureTTS)  |

### Text-to-Image

| Platform/Model                  | Notes                                     |
| ------------------------------- | ----------------------------------------- |
| 阿里云百炼 | [Plugin](https://github.com/Thetail001/LangBot_BailianTextToImagePlugin) |

### Contributing

We appreciate the contributions of our [contributors](https://github.com/langbot-app/LangBot/graphs/contributors)!

<a href="https://github.com/langbot-app/LangBot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=langbot-app/LangBot" />
</a>

**[Visit the LangBot GitHub Repository](https://github.com/langbot-app/LangBot) to get started!**
```

Key improvements and SEO considerations:

*   **Clear Headline:** "LangBot: Build Your Own AI Chatbot with Ease" – instantly communicates the project's value.
*   **SEO Keywords:** Incorporated keywords like "AI chatbot," "open source," "LLM," "chatbot development," and platform names.
*   **Concise Hook:** The one-sentence description clearly states the project's purpose.
*   **Key Features:**  Uses bullet points for easy readability, highlighting the most important aspects.
*   **Deployment Section:**  Provides clear, step-by-step Docker Compose instructions, and other deployment options are well-organized.
*   **Platform and LLM Tables:** Uses tables to showcase integrations for search engine indexing and user-friendliness.
*   **Call to Action:** Includes a direct call to visit the GitHub repository at the end.
*   **Clean Formatting:**  Consistent headings, bold text for emphasis, and appropriate use of links enhance readability.
*   **Updated links:** Keeps all the links relevant.