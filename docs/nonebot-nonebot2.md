# NoneBot: Build Powerful, Cross-Platform Python Chatbots

**Looking to build a feature-rich, adaptable chatbot?** NoneBot is a modern, asynchronous Python framework that empowers you to create chatbots for various platforms with ease.  [See the original repo](https://github.com/nonebot/nonebot2) for more details.

<p align="center">
  <a href="https://nonebot.dev/"><img src="https://nonebot.dev/logo.png" width="200" height="200" alt="nonebot"></a>
</p>

## Key Features

*   **Asynchronous Architecture:** Handles a massive volume of messages efficiently.
*   **Easy Development:**  Simple code structure with the NB-CLI makes building chatbots straightforward.
*   **Reliable and Type-Safe:** Benefit from 100% type annotation coverage, reducing bugs.
*   **Vibrant Community:**  Tap into a large and active community of users and developers.
*   **Cross-Platform Support:**  Build chatbots for multiple platforms with a single framework.

## Supported Platforms

NoneBot2 offers extensive support for various chat platforms, with new adapters being added regularly. Here's a glimpse:

| Platform                                                                                                       | Status | Notes                                                                                                  |
| :------------------------------------------------------------------------------------------------------------- | :-----: | :------------------------------------------------------------------------------------------------------ |
| OneBot ([Docs](https://onebot.dev/)) (QQ, Telegram, WeChat, KOOK, etc.)                                      |   ✅    | Supports multiple platforms, see [ecosystem](https://onebot.dev/ecosystem.html)                     |
| Telegram                                                                                                       |   ✅    |                                                                                                         |
| Feishu                                                                                                       |   ✅    |                                                                                                         |
| GitHub                                                                                                         |   ✅    | GitHub APP & OAuth APP                                                                                |
| QQ                                                                                                           |   ✅    | QQ official API adjustments are frequent.                                                                  |
| Console                                                                                                        |   ✅    | Console Interaction                                                                                    |
| Red (QQNT)                                                                                                     |   ✅    |                                                                                                         |
| Satori                                                                                                     |   ✅    | Supports Onebot, TG, Feishu, WeChat Official Accounts, Koishi, etc.                                       |
| Discord                                                                                                      |   ✅    |                                                                                                         |
| DoDo                                                                                                           |   ✅    |                                                                                                         |
| Kritor (QQNT)                                                                                                     |   ✅    |                                                                                                         |
| Mirai (QQ)                                                                                                      |   ✅    |                                                                                                         |
| Milky (QQNT)                                                                                                     |   ✅    |                                                                                                         |
| DingTalk (Seeking Maintainer)                                                                                   |  🤗   | Currently unavailable.                                                                                   |
| Kaiheila (Community)                                                                                           |   ↗️   |                                                                                                         |
| Ntchat (Community)                                                                                           |   ↗️   | WeChat support.                                                                                           |
| MineCraft (Community)                                                                                           |   ↗️   |                                                                                                         |
| Walle-Q (Community)                                                                                           |   ↗️   | QQ Support.                                                                                          |
| Villa (米游社)                                                                                                   |   ❌   |  Official shutdown.                                                                                          |
| Rocket.Chat (Community)                                                                                         |   ↗️   |                                                                                                         |
| Tailchat (Community)                                                                                         |   ↗️   |                                                                                                         |
| Mail (Community)                                                                                         |   ↗️   | Email support.                                                                                           |
| Heybox (Community)                                                                                         |   ↗️   |                                                                                                         |
| WeChat Official Account (Community)                                                                                         |   ↗️   |                                                                                                         |
| Gewechat (不再维护)                                                                                         |   ❌   | WeChat support.                                                                                           |
| EFChat (Community)                                                                                         |   ↗️   |                                                                                                         |
| VoceChat (Community)                                                                                         |   ↗️   |                                                                                                         |
| Bilibili Live (Community)                                                                                         |   ↗️   | Bilibili Live (Web API/Open Platform) support.                                                                                                         |

## Web Framework Support

NoneBot2 integrates seamlessly with various web frameworks:

*   FastAPI
*   Quart (asynchronous Flask)
*   aiohttp
*   httpx
*   websockets

## Getting Started

1.  **Install pipx:**
    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2.  **Install the CLI:**
    ```bash
    pipx install nb-cli
    ```

3.  **Create a new project:**
    ```bash
    nb create
    ```

4.  **Run your project:**
    ```bash
    nb run
    ```

## Resources

*   [Documentation](https://nonebot.dev/)
*   [Quick Start](https://nonebot.dev/docs/quick-start)
*   [FAQ](https://faq.nonebot.dev/)
*   [Discussions](https://discussions.nonebot.dev/)
*   [Awesome NoneBot](https://github.com/nonebot/awesome-nonebot)
*   [Plugin Store](https://nonebot.dev/store/plugins)

### Plugins

*   [NoneBot-Plugin-Docs](https://github.com/nonebot/nonebot2/tree/master/packages/nonebot-plugin-docs) - Offline Documentation
    ```bash
    nb plugin install nonebot_plugin_docs
    ```
    *   [Documentation Mirror (China)](https://nb2.baka.icu)

## License

MIT License

## Contributing

See the [contribution guide](./CONTRIBUTING.md).

## Acknowledgments

Thanks to the sponsors and contributors.