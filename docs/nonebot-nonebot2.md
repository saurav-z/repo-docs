# NoneBot: The Modern, Cross-Platform, Asynchronous Python Chatbot Framework

**Build powerful and versatile chatbots with NoneBot, a Python framework designed for flexibility and ease of use. Explore the original repo here: [NoneBot2 GitHub](https://github.com/nonebot/nonebot2)**

<p align="center">
  <a href="https://nonebot.dev/"><img src="https://nonebot.dev/logo.png" width="200" height="200" alt="nonebot"></a>
</p>

## Key Features

*   **Asynchronous Architecture:** Handle a massive volume of messages efficiently using Python's asynchronous features.
*   **Simplified Development:** Easily write and deploy chatbots with the help of the NB-CLI scaffold, allowing you to focus on your business logic.
*   **Reliable & Type-Safe:** Benefit from 100% type annotation coverage, ensuring code quality and reducing bugs with editor integration.
*   **Extensive Community:** Join a vibrant community of over 100,000 users with plenty of resources and support.
*   **Cross-Platform Compatibility:** Support multiple chat platforms through adaptable communication protocols.

## Supported Platforms

| Protocol                                                                 | Status | Notes                                                                                              |
| :----------------------------------------------------------------------- | :-----: | :------------------------------------------------------------------------------------------------- |
| OneBot (QQ, Telegram, WeChat, KOOK, etc.)                               |   ✅   | See [OneBot Ecosystem](https://onebot.dev/ecosystem.html)                                          |
| Telegram                                                                 |   ✅   |                                                                                                    |
| Feishu                                                                   |   ✅   |                                                                                                    |
| GitHub                                                                   |   ✅   | GitHub APP & OAuth APP                                                                               |
| QQ                                                                       |   ✅   | QQ official API adjustments are frequent                                                             |
| Console                                                                  |   ✅   | Console interaction                                                                                |
| Red (QQNT)                                                               |   ✅   | QQNT protocol                                                                                      |
| Satori                                                                   |   ✅   | Supports Onebot, TG, Feishu, WeChat Official Accounts, Koishi, etc.                                 |
| Discord                                                                  |   ✅   | Discord Bot protocol                                                                                 |
| DoDo                                                                     |   ✅   | DoDo Bot Protocol                                                                                    |
| Kritor                                                                   |   ✅   | Kritor (OnebotX) protocol, QQNT bot interface standard                                                |
| Mirai                                                                    |   ✅   | QQ protocol                                                                                        |
| Milky                                                                    |   ✅   | QQNT bot application interface standard                                                               |
| DingTalk                                                                 |   🤗   | Seeking Maintainer (Currently unavailable)                                                        |
| Kaiheila (Community)                                                     |   ↗️   | Contributed by the community                                                                       |
| Ntchat (Community)                                                       |   ↗️   | WeChat protocol, contributed by the community                                                        |
| MineCraft (Community)                                                    |   ↗️   | Contributed by the community                                                                       |
| BiliBili Live (Community)                                                |   ↗️   | Contributed by the community                                                                       |
| Walle-Q (Community)                                                      |   ↗️   | QQ protocol, contributed by the community                                                        |
| Villa                                                                    |   ❌   | MiYouShe Da BieYe Bot protocol, officially discontinued                                           |
| Rocket.Chat (Community)                                                  |   ↗️   | Rocket.Chat Bot protocol, contributed by the community                                              |
| Tailchat (Community)                                                     |   ↗️   | Tailchat open platform Bot protocol, contributed by the community                                   |
| Mail (Community)                                                         |   ↗️   | Email sending and receiving protocol, contributed by the community                                  |
| HeyBox (Community)                                                       |   ↗️   | Heybox voice bot protocol, contributed by the community                                             |
| WeChat Official Accounts (Community)                                     |   ↗️   | WeChat Official Accounts protocol, contributed by the community                                     |
| Gewechat                                                                  |   ❌   | Gewechat WeChat protocol, Gewechat is no longer maintained and available                                |
| EFChat (Community)                                                       |   ↗️   | Heng Wu Liao platform protocol, contributed by the community                                       |

## Supporting Frameworks

*   **FastAPI:** Server-side
*   **Quart (async Flask):** Server-side
*   **aiohttp:** Client-side
*   **httpx:** Client-side
*   **websockets:** Client-side

## Get Started

1.  Install [pipx](https://pypa.github.io/pipx/)

    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2.  Install the scaffolding tool

    ```bash
    pipx install nb-cli
    ```

3.  Create your project

    ```bash
    nb create
    ```

4.  Run your project

    ```bash
    nb run
    ```

## Community Resources

*   [FAQ](https://faq.nonebot.dev/)
*   [Discussion Forum](https://discussions.nonebot.dev/)
*   [Awesome NoneBot](https://github.com/nonebot/awesome-nonebot)
*   [Plugins](https://nonebot.dev/store/plugins)

    *   [NoneBot-Plugin-Docs](https://github.com/nonebot/nonebot2/tree/master/packages/nonebot-plugin-docs): Offline documentation to the local project (Don't say the documentation can't be opened!)

        In the project directory, execute:

        ```bash
        nb plugin install nonebot_plugin_docs
        ```

        Or try the following mirror:

        *   [Documentation mirror (within China)](https://nb2.baka.icu)

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/nonebot/nonebot2/master/LICENSE).

## Contributing

Please refer to the [Contributing Guide](./CONTRIBUTING.md)

## Acknowledgements

**(List of sponsors and contributors)**