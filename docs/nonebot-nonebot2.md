# NoneBot: Build Powerful, Cross-Platform Chatbots with Python

**Create feature-rich, asynchronous chatbots that work across multiple platforms with NoneBot, a cutting-edge Python framework.**  [Explore the original repository](https://github.com/nonebot/nonebot2).

<p align="center">
  <a href="https://nonebot.dev/"><img src="https://nonebot.dev/logo.png" width="200" height="200" alt="nonebot"></a>
</p>

## Key Features

*   **Asynchronous Architecture:** Handles high message volumes efficiently using Python's asynchronous capabilities.
*   **Easy Development:** Streamlines development with the NB-CLI scaffolding tool, focusing on your bot's core logic.
*   **Type Safety & Reliability:** Leverages 100% type annotations for robust code and enhanced error detection in your IDE.
*   **Thriving Community:** Benefit from a large and active community, providing extensive support and resources.
*   **Cross-Platform Compatibility:** Supports a wide array of chat platforms, with customizable communication protocols.

## Supported Platforms & Adapters

| Protocol Name | Status | Notes |
| :-------------------------------------------------------------------------------------------------------------------: | :--: | :-----------------------------------------------------------------------: |
|               OneBot（[Repository](https://github.com/nonebot/adapter-onebot)，[Protocol](https://onebot.dev/)）                |  ✅  | Supports QQ, TG, WeChat Official Accounts, KOOK, and more. |
|      Telegram（[Repository](https://github.com/nonebot/adapter-telegram)，[Protocol](https://core.telegram.org/bots/api)）      |  ✅  |                                                                           |
|     Feishu（[Repository](https://github.com/nonebot/adapter-feishu)，[Protocol](https://open.feishu.cn/document/home/index)）     |  ✅  |                                                                           |
|         GitHub（[Repository](https://github.com/nonebot/adapter-github)，[Protocol](https://docs.github.com/en/apps)）          |  ✅  |                          GitHub APP & OAuth APP                           |
|                QQ（[Repository](https://github.com/nonebot/adapter-qq)，[Protocol](https://bot.q.qq.com/wiki/)）                |  ✅  |                            QQ Official Interface                           |
|                             Console（[Repository](https://github.com/nonebot/adapter-console)）                             |  ✅  |                                Console Interaction                                 |
|     Red（[Repository](https://github.com/nonebot/adapter-red)，[Protocol](https://chrononeko.github.io/QQNTRedProtocol/)）      |  ✅  |                                 QQNT Protocol                                  |
|           Satori（[Repository](https://github.com/nonebot/adapter-satori)，[Protocol](https://satori.js.org/zh-CN)）            |  ✅  |               Supports Onebot, TG, Feishu, WeChat Official Accounts, Koishi, etc.                |
|   Discord（[Repository](https://github.com/nonebot/adapter-discord)，[Protocol](https://discord.com/developers/docs/intro)）    |  ✅  |                             Discord Bot Protocol                              |
|               DoDo（[Repository](https://github.com/nonebot/adapter-dodo)，[Protocol](https://open.imdodo.com/)）               |  ✅  |                               DoDo Bot Protocol                               |
|        Kritor（[Repository](https://github.com/nonebot/adapter-kritor)，[Protocol](https://github.com/KarinJS/kritor)）         |  ✅  |                Kritor (OnebotX) Protocol, QQNT bot interface standard                  |
|    Mirai（[Repository](https://github.com/nonebot/adapter-mirai)，[Protocol](https://docs.mirai.mamoe.net/mirai-api-http/)）    |  ✅  |                                  QQ Protocol                                  |
|    Milky（[Repository](https://github.com/nonebot/adapter-milky)，[Protocol](https://milky.ntqqrev.org/)）                      |  ✅  |                           QQNT bot application interface standard                          |
|         DingTalk（[Repository](https://github.com/nonebot/adapter-ding)，[Protocol](https://open.dingtalk.com/document/)）          |  🤗  |                        Seeking Maintainer (Unavailable)                        |
|     Kaiheila（[Repository](https://github.com/Tian-que/nonebot-adapter-kaiheila)，[Protocol](https://developer.kookapp.cn/)）     |  ↗️  |                                 Community Contribution                                 |
|                          Ntchat（[Repository](https://github.com/JustUndertaker/adapter-ntchat)）                           |  ↗️  |                           WeChat Protocol, Community Contribution                            |
|                      MineCraft（[Repository](https://github.com/17TheWord/nonebot-adapter-minecraft)）                      |  ↗️  |                                 Community Contribution                                 |
|                       Walle-Q（[Repository](https://github.com/onebot-walle/nonebot_adapter_walleq)）                       |  ↗️  |                            QQ Protocol, Community Contribution                            |
|                       Villa（[Repository](https://github.com/CMHopeSunshine/nonebot-adapter-villa)）                        |  ❌  |                     Mi You She Da Bie Ye Bot Protocol, Official Offline                     |
| Rocket.Chat（[Repository](https://github.com/IUnlimit/nonebot-adapter-rocketchat)，[Protocol](https://developer.rocket.chat/)） |  ↗️  |                     Rocket.Chat Bot Protocol, Community Contribution                      |
|     Tailchat（[Repository](https://github.com/eya46/nonebot-adapter-tailchat)，[Protocol](https://tailchat.msgbyte.com/)）      |  ↗️  |                  Tailchat Open Platform Bot Protocol, Community Contribution                   |
|                             Mail（[Repository](https://github.com/mobyw/nonebot-adapter-mail)）                             |  ↗️  |                         Email Sending and Receiving Protocol, Community Contribution                          |
|     Heybox Voice（[Repository](https://github.com/lclbm/adapter-heybox)，[Protocol](https://github.com/QingFengOpen/HeychatDoc)）     |  ↗️  |                       Heybox Voice Bot Protocol, Community Contribution                             |
| WeChat Official Account（[Repository](https://github.com/YangRucheng/nonebot-adapter-wxmp)，[Protocol](https://developers.weixin.qq.com/doc/)）|  ↗️  |                       WeChat Official Account Protocol, Community Contribution                             |
| Gewechat（[Repository](https://github.com/Shine-Light/nonebot-adapter-gewechat)，[Protocol](https://github.com/Devo919/Gewechat)）|  ❌  |                      Gewechat WeChat Protocol, Gewechat is no longer maintained and available                            |
|  EFChat（[Repository](https://github.com/molanp/nonebot_adapter_efchat)，[Protocol](https://irinu-live.melon.fish/efc-help/)）   |  ↗️  |                            Hengwu Chat Platform Protocol, Community Contribution                          |
|  VoceChat （[Repository](https://github.com/5656565566/nonebot-adapter-vocechat)，[Protocol](https://doc.voce.chat/zh-cn/bot/bot-and-webhook)）   |  ↗️  |                            VoceChat Platform Protocol, Community Contribution                          |
|  Bilibili Live Room（[Repository](https://github.com/MingxuanGame/nonebot-adapter-bilibili-live)，[Web API Protocol](https://github.com/SocialSisterYi/bilibili-API-collect/blob/master/docs/live)，[Open Platform Protocol](https://open-live.bilibili.com/document)）   |  ↗️  |                            Bilibili Live Room（Web API/Open Platform）Protocol, Community Contribution                          |

## Web Framework Support

*   FastAPI (Server-side)
*   Quart (Async Flask) (Server-side)
*   aiohttp (Client-side)
*   httpx (Client-side)
*   websockets (Client-side)

## Getting Started

1.  Install [pipx](https://pypa.github.io/pipx/):

    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2.  Install NB-CLI:

    ```bash
    pipx install nb-cli
    ```

3.  Create a new project:

    ```bash
    nb create
    ```

4.  Run your project:

    ```bash
    nb run
    ```

## Resources

### FAQs

*   [Frequently Asked Questions](https://faq.nonebot.dev/)
*   [Discussions](https://discussions.nonebot.dev/)

### Tutorials / Projects / Sharing

*   [awesome-nonebot](https://github.com/nonebot/awesome-nonebot)

### Plugins

*   [NoneBot-Plugin-Docs](https://github.com/nonebot/nonebot2/tree/master/packages/nonebot-plugin-docs): Offline documentation for your project

    Run this in your project directory:

    ```bash
    nb plugin install nonebot_plugin_docs
    ```
    or use mirror (Chinese mainland): [Documentation Mirror (Chinese mainland)](https://nb2.baka.icu)

*   Browse more plugins in the [Store](https://nonebot.dev/store/plugins).

## License

NoneBot is licensed under the [MIT License](https://raw.githubusercontent.com/nonebot/nonebot2/master/LICENSE).

## Contributing

See the [Contributing Guide](./CONTRIBUTING.md) for details.

## Acknowledgements

### Sponsors

[Sponsor logos and links - As per original README]

### Contributors

[Contributors' image - As per original README]