# NoneBot: Build Powerful, Cross-Platform Chatbots with Python

**Create versatile and efficient chatbots for any platform with NoneBot, the asynchronous Python framework.**  [Explore the project on GitHub](https://github.com/nonebot/nonebot2).

<p align="center">
  <a href="https://nonebot.dev/"><img src="https://nonebot.dev/logo.png" width="200" height="200" alt="nonebot"></a>
</p>

<div align="center">

## Key Features of NoneBot

</div>

*   **Asynchronous Architecture:** Handle massive message volumes with ease thanks to Python's async capabilities.
*   **Simplified Development:** The NB-CLI scaffolding tool streamlines code creation, allowing you to focus on core logic.
*   **Reliable & Robust:** Benefit from 100% type annotations and editor support for enhanced code quality and reduced bugs.
*   **Thriving Community:** Join a large and active community of users and developers.
*   **Multi-Platform Support:** Build chatbots that work across multiple platforms.

### Supported Platforms & Protocols

| Protocol                                                                                                   | Status | Notes                                                                                                                                                                |
| :--------------------------------------------------------------------------------------------------------- | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OneBot ([Adapter Repo](https://github.com/nonebot/adapter-onebot), [Protocol](https://onebot.dev/))         |  ✅  | Supports QQ, Telegram, WeChat Official Accounts, KOOK, and more.                                                                                       |
| Telegram ([Adapter Repo](https://github.com/nonebot/adapter-telegram), [Protocol](https://core.telegram.org/bots/api)) |  ✅  |                                                                                                                                                                       |
| Feishu ([Adapter Repo](https://github.com/nonebot/adapter-feishu), [Protocol](https://open.feishu.cn/document/home/index)) |  ✅  |                                                                                                                                                                       |
| GitHub ([Adapter Repo](https://github.com/nonebot/adapter-github), [Protocol](https://docs.github.com/en/apps))       |  ✅  | GitHub APP & OAuth APP                                                                                                                                                  |
| QQ ([Adapter Repo](https://github.com/nonebot/adapter-qq), [Protocol](https://bot.q.qq.com/wiki/))                  |  ✅  | QQ official interface changes frequently.                                                                                                                               |
| Console ([Adapter Repo](https://github.com/nonebot/adapter-console))                                              |  ✅  | Console interaction                                                                                                                                                     |
| Red ([Adapter Repo](https://github.com/nonebot/adapter-red), [Protocol](https://chrononeko.github.io/QQNTRedProtocol/))        |  ✅  | QQNT protocol                                                                                                                                                       |
| Satori ([Adapter Repo](https://github.com/nonebot/adapter-satori), [Protocol](https://satori.js.org/zh-CN))           |  ✅  | Supports Onebot, Telegram, Feishu, WeChat Official Accounts, Koishi, etc.                                                                                              |
| Discord ([Adapter Repo](https://github.com/nonebot/adapter-discord), [Protocol](https://discord.com/developers/docs/intro)) |  ✅  | Discord Bot protocol                                                                                                                                                     |
| DoDo ([Adapter Repo](https://github.com/nonebot/adapter-dodo), [Protocol](https://open.imdodo.com/))                 |  ✅  | DoDo Bot protocol                                                                                                                                                       |
| Kritor ([Adapter Repo](https://github.com/nonebot/adapter-kritor), [Protocol](https://github.com/KarinJS/kritor))          |  ✅  | Kritor (OnebotX) protocol, QQNT bot interface standard                                                                                                                   |
| Mirai ([Adapter Repo](https://github.com/nonebot/adapter-mirai), [Protocol](https://docs.mirai.mamoe.net/mirai-api-http/))   |  ✅  | QQ protocol                                                                                                                                                          |
| Milky ([Adapter Repo](https://github.com/nonebot/adapter-milky), [Protocol](https://milky.ntqqrev.org/))                   |  ✅  | QQNT bot application interface standard                                                                                                                              |
| DingTalk ([Adapter Repo](https://github.com/nonebot/adapter-ding), [Protocol](https://open.dingtalk.com/document/))       |  🤗  | Seeking a maintainer (currently unavailable)                                                                                                                             |
| Kaiheila ([Adapter Repo](https://github.com/Tian-que/nonebot-adapter-kaiheila), [Protocol](https://developer.kookapp.cn/)) |  ↗️  | Community contribution                                                                                                                                                   |
| Ntchat ([Adapter Repo](https://github.com/JustUndertaker/adapter-ntchat))                                               |  ↗️  | WeChat protocol, community contribution                                                                                                                               |
| MineCraft ([Adapter Repo](https://github.com/17TheWord/nonebot-adapter-minecraft))                                         |  ↗️  | Community contribution                                                                                                                                                   |
| Walle-Q ([Adapter Repo](https://github.com/onebot-walle/nonebot_adapter_walleq))                                           |  ↗️  | QQ protocol, community contribution                                                                                                                                   |
| Villa ([Adapter Repo](https://github.com/CMHopeSunshine/nonebot-adapter-villa))                                            |  ❌  | Mihoyo Dabieye Bot protocol, officially offline                                                                                                                     |
| Rocket.Chat ([Adapter Repo](https://github.com/IUnlimit/nonebot-adapter-rocketchat), [Protocol](https://developer.rocket.chat/)) |  ↗️  | Rocket.Chat Bot protocol, community contribution                                                                                                                          |
| Tailchat ([Adapter Repo](https://github.com/eya46/nonebot-adapter-tailchat), [Protocol](https://tailchat.msgbyte.com/))     |  ↗️  | Tailchat Open Platform Bot protocol, community contribution                                                                                                             |
| Mail ([Adapter Repo](https://github.com/mobyw/nonebot-adapter-mail))                                                     |  ↗️  | Mail sending and receiving protocol, community contribution                                                                                                          |
| Heybox ([Adapter Repo](https://github.com/lclbm/adapter-heybox), [Protocol](https://github.com/QingFengOpen/HeychatDoc))     |  ↗️  | Heybox voice bot protocol, community contribution                                                                                                                       |
| WeChat Official Account ([Adapter Repo](https://github.com/YangRucheng/nonebot-adapter-wxmp), [Protocol](https://developers.weixin.qq.com/doc/)) |  ↗️  | WeChat Official Account protocol, community contribution                                                                                                            |
| Gewechat ([Adapter Repo](https://github.com/Shine-Light/nonebot-adapter-gewechat), [Protocol](https://github.com/Devo919/Gewechat)) |  ❌  | Gewechat WeChat protocol, Gewechat is no longer maintained and available                                                                                            |
| EFChat ([Adapter Repo](https://github.com/molanp/nonebot_adapter_efchat), [Protocol](https://irinu-live.melon.fish/efc-help/)) |  ↗️  | Hengwu Chat platform protocol, community contribution                                                                                                                 |
| VoceChat ([Adapter Repo](https://github.com/5656565566/nonebot-adapter-vocechat), [Protocol](https://doc.voce.chat/zh-cn/bot/bot-and-webhook)) |  ↗️  | VoceChat platform protocol, community contribution                                                                                                                 |
| Bilibili Live ([Adapter Repo](https://github.com/MingxuanGame/nonebot-adapter-bilibili-live), [Web API Protocol](https://github.com/SocialSisterYi/bilibili-API-collect/blob/master/docs/live), [Open Platform Protocol](https://open-live.bilibili.com/document)) |  ↗️  | Bilibili Live (Web API/Open Platform) protocol, community contribution                                                                                             |

### Backend Frameworks

NoneBot supports various web frameworks.

| Framework                           | Type     |
| :---------------------------------- | :-------: |
| [FastAPI](https://fastapi.tiangolo.com/)   | Server |
| [Quart](https://quart.palletsprojects.com/en/latest/) (Async Flask) | Server |
| [aiohttp](https://docs.aiohttp.org/en/stable/)       | Client |
| [httpx](https://www.python-httpx.org/)            | Client |
| [websockets](https://websockets.readthedocs.io/en/stable/)   | Client |

## Getting Started

1.  **Install `pipx`:**

    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2.  **Install the CLI Tool:**

    ```bash
    pipx install nb-cli
    ```

3.  **Create a New Project:**

    ```bash
    nb create
    ```

4.  **Run Your Bot:**

    ```bash
    nb run
    ```

## Resources

*   [Documentation](https://nonebot.dev/)
*   [Quick Start](https://nonebot.dev/docs/quick-start)
*   [FAQ](https://faq.nonebot.dev/)
*   [Forum (Discussion)](https://discussions.nonebot.dev/)
*   [Awesome NoneBot](https://github.com/nonebot/awesome-nonebot)
*   [Plugin Store](https://nonebot.dev/store/plugins)

## Plugin: Local Offline Documentation

Install the official offline docs plugin to access documentation locally:

```bash
nb plugin install nonebot_plugin_docs
```

## License

`NoneBot` is open-sourced under the [MIT License](./LICENSE).

## Contributing

See the [contribution guidelines](./CONTRIBUTING.md) for details on how to contribute.

## Acknowledgements

This project is supported by many sponsors and contributors.

### Sponsors

<p align="center">
  <a href="https://github.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://assets.nonebot.dev/github-dark.png">
      <img src="https://assets.nonebot.dev/github-light.png" height="50" alt="GitHub">
    </picture>
  </a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.netlify.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://assets.nonebot.dev/netlify-dark.svg">
      <img src="https://assets.nonebot.dev/netlify-light.svg" height="50" alt="netlify">
    </picture>
  </a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://sentry.io/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://assets.nonebot.dev/sentry-dark.svg">
      <img src="https://assets.nonebot.dev/sentry-light.svg" height="50" alt="sentry">
    </picture>
  </a>
</p>
<p align="center">
  <a href="https://www.docker.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://assets.nonebot.dev/docker-dark.svg">
      <img src="https://assets.nonebot.dev/docker-light.svg" height="50" alt="docker">
    </picture>
  </a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.algolia.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://assets.nonebot.dev/algolia-dark.svg">
      <img src="https://assets.nonebot.dev/algolia-light.svg" height="50" alt="algolia">
    </picture>
  </a>
</p>
<p align="center">
  <a href="https://www.jetbrains.com/">
    <img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.svg" height="80" alt="JetBrains" >
  </a>
</p>

### Financial Support

<a href="https://assets.nonebot.dev/sponsors.svg">
  <img src="https://assets.nonebot.dev/sponsors.svg" alt="sponsors" />
</a>

### Contributors

<a href="https://github.com/nonebot/nonebot2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nonebot/nonebot2&max=1000" alt="contributors" />
</a>