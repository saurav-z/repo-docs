# NoneBot2: Build Powerful, Cross-Platform Chatbots with Python

**Create versatile and robust chatbots effortlessly with NoneBot2, a modern, asynchronous Python framework.**  [See the original repo](https://github.com/nonebot/nonebot2).

<p align="center">
  <a href="https://nonebot.dev/"><img src="https://nonebot.dev/logo.png" width="200" height="200" alt="nonebot"></a>
</p>

## Key Features

*   **Asynchronous by Design:** Handles high message volumes efficiently using Python's async features.
*   **Easy Development:**  Simplified coding with the NB-CLI scaffolding tool, allowing developers to focus on bot logic.
*   **Reliable and Type-Safe:** 100% type-annotated code, leveraging editor features for early bug detection.
*   **Vibrant Community:** Benefit from a large and active community with extensive resources and support.
*   **Cross-Platform Compatibility:** Supports multiple chat platforms through adaptable adapter integrations.

## Adapters Supported

NoneBot2 supports a wide array of platforms via adapter integrations:

*   **OneBot:**  (✅)  QQ, Telegram, WeChat Official Accounts, KOOK, and others
*   **Telegram:** (✅)
*   **Feishu:** (✅)
*   **GitHub:** (✅)  GitHub APP & OAuth APP
*   **QQ:** (✅) QQ Official Interface (Adjustments may be needed)
*   **Console:** (✅) Console Interaction
*   **Red:** (✅) QQNT Protocol
*   **Satori:** (✅) Onebot, Telegram, Feishu, WeChat Official Accounts, Koishi, etc.
*   **Discord:** (✅) Discord Bot Protocol
*   **DoDo:** (✅) DoDo Bot Protocol
*   **Kritor:** (✅) Kritor (OnebotX) Protocol, QQNT Robot Interface Standard
*   **Mirai:** (✅) QQ Protocol
*   **Milky:** (✅) QQNT Robot Application Interface Standard
*   **DingTalk:** (🤗)  Seeking Maintainers (Currently Unavailable)
*   **Kaiheila:** (↗️) Community Contributed
*   **Ntchat:** (↗️) WeChat Protocol, Community Contributed
*   **MineCraft:** (↗️) Community Contributed
*   **BiliBili Live:** (↗️) Community Contributed
*   **Walle-Q:** (↗️) QQ Protocol, Community Contributed
*   **Villa:** (❌)  Mihoyo Big Wild Bot Protocol, Official Offline
*   **Rocket.Chat:** (↗️) Rocket.Chat Bot Protocol, Community Contributed
*   **Tailchat:** (↗️) Tailchat Open Platform Bot Protocol, Community Contributed
*   **Mail:** (↗️) Mail Sending and Receiving Protocol, Community Contributed
*   **Heybox:** (↗️) Heybox Robot Protocol, Community Contributed
*   **Wxmp:** (↗️) WeChat Official Account Protocol, Community Contributed
*   **Gewechat:** (❌) Gewechat WeChat Protocol, Gewechat no longer maintained and available
*   **EFChat:** (↗️) Hengwu Chat Platform Protocol, Community Contributed

## Supported Web Frameworks (for server integration)

*   FastAPI (Server)
*   Quart (Asynchronous Flask) (Server)
*   aiohttp (Client)
*   httpx (Client)
*   websockets (Client)

## Get Started Quickly

1.  Install [pipx](https://pypa.github.io/pipx/)

    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2.  Install the scaffold

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

*   [Documentation](https://nonebot.dev/)
*   [Quick Start](https://nonebot.dev/docs/quick-start)
*   [FAQ](https://faq.nonebot.dev/)
*   [Discussions](https://discussions.nonebot.dev/)
*   [Awesome NoneBot](https://github.com/nonebot/awesome-nonebot)
*   [Plugins Store](https://nonebot.dev/store/plugins)

## Plugin for local documentation
```bash
nb plugin install nonebot_plugin_docs
```
## License

NoneBot2 is released under the [MIT License](./LICENSE).

## Contributing

See the [Contribution Guide](./CONTRIBUTING.md) to get involved.

## Thanks

*   [Sponsors and Developers (see the original README for details)](https://github.com/nonebot/nonebot2)