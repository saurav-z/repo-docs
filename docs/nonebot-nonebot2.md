# NoneBot 2: Build Powerful, Cross-Platform Chatbots with Python

**Create versatile and robust chatbots with NoneBot 2, a modern, asynchronous Python framework.**  Explore the power of NoneBot 2 with its easy-to-use framework that supports various platforms. Visit the [original repo](https://github.com/nonebot/nonebot2) for more details.

[![License](https://img.shields.io/github/license/nonebot/nonebot2?logo=github&logoColor=white)](https://raw.githubusercontent.com/nonebot/nonebot2/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/nonebot2?logo=python&logoColor=edb641)](https://pypi.python.org/pypi/nonebot2)
[![Python version](https://img.shields.io/badge/python-3.9+-blue?logo=python&logoColor=edb641)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=python&logoColor=edb641)](https://github.com/psf/black)
[![Type checking with pyright](https://img.shields.io/badge/types-pyright-797952.svg?logo=python&logoColor=edb641)](https://github.com/Microsoft/pyright)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Codecov](https://codecov.io/gh/nonebot/nonebot2/branch/master/graph/badge.svg?token=2P0G0VS7N4)](https://codecov.io/gh/nonebot/nonebot2)
[![Website Deployment](https://github.com/nonebot/nonebot2/actions/workflows/website-deploy.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/website-deploy.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/nonebot/nonebot2/master.svg)](https://results.pre-commit.ci/latest/github/nonebot/nonebot2/master)
[![Pyright](https://github.com/nonebot/nonebot2/actions/workflows/pyright.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/pyright.yml)
[![Ruff](https://github.com/nonebot/nonebot2/actions/workflows/ruff.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/ruff.yml)
[![OneBot v11](https://img.shields.io/badge/OneBot-v11-black?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABABAMAAABYR2ztAAAAIVBMVEUAAAAAAAADAwMHBwceHh4UFBQNDQ0ZGRkoKCgvLy8iIiLWSdWYAAAAAXRSTlMAQObYZgAAAQVJREFUSMftlM0RgjAQhV+0ATYK6i1Xb+iMd0qgBEqgBEuwBOxU2QDKsjvojQPvkJ/ZL5sXkgWrFirK4MibYUdE3OR2nEpuKz1/q8CdNxNQgthZCXYVLjyoDQftaKuniHHWRnPh2GCUetR2/9HsMAXyUT4/3UHwtQT2AggSCGKeSAsFnxBIOuAggdh3AKTL7pDuCyABcMb0aQP7aM4AnAbc/wHwA5D2wDHTTe56gIIOUA/4YYV2e1sg713PXdZJAuncdZMAGkAukU9OAn40O849+0ornPwT93rphWF0mgAbauUrEOthlX8Zu7P5A6kZyKCJy75hhw1Mgr9RAUvX7A3csGqZegEdniCx30c3agAAAABJRU5ErkJggg==")](https://onebot.dev/)
[![OneBot v12](https://img.shields.io/badge/OneBot-v12-black?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABABAMAAABYR2ztAAAAIVBMVEUAAAAAAAADAwMHBwceHh4UFBQNDQ0ZGRkoKCgvLy8iIiLWSdWYAAAAAXRSTlMAQObYZgAAAQVJREFUSMftlM0RgjAQhV+0ATYK6i1Xb+iMd0qgBEqgBEuwBOxU2QDKsjvojQPvkJ/ZL5sXkgWrFirK4MibYUdE3OR2nEpuKz1/q8CdNxNQgthZCXYVLjyoDQftaKuniHHWRnPh2GCUetR2/9HsMAXyUT4/3UHwtQT2AggSCGKeSAsFnxBIOuAggdh3AKTL7pDuCyABcMb0aQP7aM4AnAbc/wHwA5D2wDHTTe56gIIOUA/4YYV2e1sg713PXdZJAuncdZMAGkAukU9OAn40O849+0ornPwT93rphWF0mgAbauUrEOthlX8Zu7P5A6kZyKCJy75hhw1Mgr9RAUvX7A3csGqZegEdniCx30c3agAAAABJRU5ErkJggg==")](https://onebot.dev/)
[![QQ Bot](https://img.shields.io/badge/QQ-Bot-lightgrey?style=social&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMTIuODIgMTMwLjg5Ij48ZyBkYXRhLW5hbWU9IuWbvuWxgiAyIj48ZyBkYXRhLW5hbWU9IuWbvuWxgiAxIj48cGF0aCBkPSJNNTUuNjMgMTMwLjhjLTcgMC0xMy45LjA4LTIwLjg2IDAtMTkuMTUtLjI1LTMxLjcxLTExLjQtMzQuMjItMzAuMy00LjA3LTMwLjY2IDE0LjkzLTU5LjIgNDQuODMtNjYuNjQgMi0uNTEgNS4yMS0uMzEgNS4yMS0xLjYzIDAtMi4xMy4xNC0yLjEzLjE0LTUuNTcgMC0uODktMS4zLTEuNDYtMi4yMi0yLjMxLTYuNzMtNi4yMy03LjY3LTEzLjQxLTEtMjAuMTggNS40LTUuNTIgMTEuODctNS40IDE3LjgtLjU5IDYuNDkgNS4yNiA2LjMxIDEzLjA4LS44NiAyMS0uNjguNzQtMS43OCAxLjYtMS43OCAyLjY3djQuMjFjMCAxLjM1IDIuMiAxLjYyIDQuNzkgMi4zNSAzMS4wOSA4LjY1IDQ4LjE3IDM0LjEzIDQ1IDY2LjM3LTEuNzYgMTguMTUtMTQuNTYgMzAuMjMtMzIuNyAzMC42My04LjAyLjE5LTE2LjA3LS4wMS0yNC4xMy0uMDF6IiBmaWxsPSIjMDI5OWZlIi8+PHBhdGggZD0iTTMxLjQ2IDExOC4zOGMtMTAuNS0uNjktMTYuOC02Ljg2LTE4LjM4LTE3LjI3LTMtMTkuNDIgMi43OC0zNS44NiAxOC40Ni00Ny44MyAxNC4xNi0xMC44IDI5Ljg3LTEyIDQ1LjM4LTMuMTkgMTcuMjUgOS44NCAyNC41OSAyNS44MSAyNCA0NS4yOS0uNDkgMTUuOS04LjQyIDIzLjE0LTI0LjM4IDIzLjUtNi41OS4xNC0xMy4xOSAwLTE5Ljc5IDAiIGZpbGw9IiNmZWZlZmUiLz48cGF0aCBkPSJNNDYuMDUgNzkuNThjLjA5IDUgLjIzIDkuODItNyA5Ljc3LTcuODItLjA2LTYuMS01LjY5LTYuMjQtMTAuMTktLjE1LTQuODItLjczLTEwIDYuNzMtOS44NHM2LjM3IDUuNTUgNi41MSAxMC4yNnoiIGZpbGw9IiMxMDlmZmUiLz48cGF0aCBkPSJNODAuMjcgNzkuMjdjLS41MyAzLjkxIDEuNzUgOS42NC01Ljg4IDEwLTcuNDcuMzctNi44MS00LjgyLTYuNjEtOS41LjItNC4zMi0xLjgzLTEwIDUuNzgtMTAuNDJzNi41OSA0Ljg5IDYuNzEgOS45MnoiIGZpbGw9IiMwODljZmUiLz48L2c+PC9nPjwvc3ZnPg==")](https://bot.q.qq.com/wiki/)
[![Telegram Bot](https://img.shields.io/badge/telegram-Bot-lightgrey?style=social&logo=telegram)](https://core.telegram.org/bots/api)
[![Feishu Bot](https://img.shields.io/badge/%E9%A3%9E%E4%B9%A6-Bot-lightgrey?style=social&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz48c3ZnIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDQ4IDQ4IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xNyAyOUMyMSAyOSAyNSAyNi45MzM5IDI4IDIzLjQwNjVDMzYgMTQgNDEuNDI0MiAxNi44MTY2IDQ0IDE3Ljk5OThDMzguNSAyMC45OTk4IDQwLjUgMjkuNjIzMyAzMyAzNS45OTk4QzI4LjM4MiAzOS45MjU5IDIzLjQ5NDUgNDEuMDE0IDE5IDQxQzEyLjUyMzEgNDAuOTc5OSA2Ljg2MjI2IDM3Ljc2MzcgNCAzNS40MDYzVjE2Ljk5OTgiIHN0cm9rZT0iIzMzMyIgc3Ryb2tlLXdpZHRoPSI0IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48cGF0aCBkPSJNNS42NDgwOCAxNS44NjY5QzUuMDIyMzEgMTQuOTU2NyAzLjc3NzE1IDE0LjcyNjEgMi44NjY5NCAxNS4zNTE5QzEuOTU2NzMgMTUuOTc3NyAxLjcyNjE1IDE3LjIyMjggMi4zNTE5MiAxOC4xMzMxTDUuNjQ4MDggMTUuODY2OVpNMzYuMDAyMSAzNS43MzA5QzM2Ljk1OCAzNS4xNzc0IDM3LjI4NDMgMzMuOTUzOSAzNi43MzA5IDMyLjk5NzlDMzYuMTc3NCAzMi4wNDIgMzQuOTUzOSAzMS43MTU3IDMzLjk5NzkgMzIuMjY5MUwzNi4wMDIxIDM1LjczMDlaTTIuMzUxOTIgMTguMTMzMUM1LjI0MzUgMjIuMzM5IDEwLjc5OTIgMjguMTQ0IDE2Ljg4NjUgMzIuMjIzOUMxOS45MzQ1IDM0LjI2NjcgMjMuMjE3IDM1Ljk0NiAyNi40NDkgMzYuNzMyNEMyOS42OTQ2IDM3LjUyMiAzMy4wNDUxIDM3LjQ0MjggMzYuMDAyMSAzNS43MzA5TDMzLjk5NzkgMzIuMjY5MUMzMi4yMDQ5IDMzLjMwNzIgMjkuOTkyOSAzMy40NzggMjcuMzk0NyAzMi44NDU4QzI0Ljc4MyAzMi4yMTAzIDIxLjk0MDUgMzAuNzk1OCAxOS4xMTM1IDI4LjkwMTFDMTMuNDUwOCAyNS4xMDYgOC4yNTY1IDE5LjY2MSA1LjY0ODA4IDE1Ljg2NjlMMi4zNTE5MiAxOC4xMzMxWiIgZmlsbD0iIzMzMyIvPjxwYXRoIGQ9Ik0zMy41OTQ1IDE3QzMyLjgzOTggMTQuNzAyNyAzMC44NTQ5IDkuOTQwNTQgMjcuNTk0NSA3SDExLjU5NDVDMTUuMjE3MSAxMC42NzU3IDIzIDE2IDI3IDI0IiBzdHJva2U9IiMzMzMiIHN0cm9rZS13aWR0aD0iNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PC9zdmc+" alt="feishu">
[![GitHub Bot](https://img.shields.io/badge/GitHub-Bot-181717?style=social&logo=github)](https://docs.github.com/en/developers/apps)
<!-- [![Dingtalk Bot](https://img.shields.io/badge/%E9%92%89%E9%92%89-Bot-lightgrey?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAnFBMVEUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4jUzeAAAAM3RSTlMAQKSRaA+/f0YyFevh29R3cyklIfrlyrGsn41tVUs48c/HqJm9uZdhX1otGwkF9IN8V1CX0Q+IAAABY0lEQVRYw+3V2W7CMBAF0JuNQAhhX9OEfYdu9///rUVWpagE27Ef2gfO+0zGozsKnv6bMGzAhkNytIe5gDdzrwtTCwrbI8x4/NF668NAxgI3Q3UtFi3TyPwNQtPLUUmDd8YfqGLNe4v22XwEYb5zoOuF5baHq2UHtsKe5ivWfGAwrWu2mC34QM0PoCAuqZdOmiwV+5BLyMRtZ7dTSEcs48rzWfzwptMLyzpApka1SJ5FtR4kfCqNIBPEVDmqoqgwUYY5plQOlf6UEjNoOPnuKB6wzDyCrks///TDza8+PnR109WQdxLo8RKWq0PPnuXG0OXKQ6wWLFnCg75uYYbhmMIVVdQ709q33aHbGIj6Duz+2k1HQFX9VwqmY8xYsEJll2ahvhWgsjYLHFRXvIi2Qb0jzMQCzC3FAoydxCma88UCzE3JCWwkjCNYyMUCzHX4DiuTMawEwwhW6hnshPhjZzzJfAH0YacpbmRd7QAAAABJRU5ErkJggg==")](https://ding-doc.dingtalk.com/document#/org-dev-guide/elzz1p) -->
[![QQ Group](https://img.shields.io/badge/QQ%E7%BE%A4-768887710-orange?style=flat-square)](https://jq.qq.com/?_wv=1027&k=5OFifDh)
[![QQ Channel](https://img.shields.io/badge/QQ%E9%A2%91%E9%81%93-NoneBot-5492ff?style=flat-square)](https://qun.qq.com/qqweb/qunpro/share?_wv=3&_wwv=128&appChannel=share&inviteCode=7b4a3&appChannel=share&businessType=9&from=246610&biz=ka)
[![Telegram Channel](https://img.shields.io/badge/telegram-botuniverse-blue?style=flat-square)](https://t.me/botuniverse)
[![Discord Server](https://discordapp.com/api/guilds/847819937858584596/widget.png?style=shield)](https://discord.gg/VKtE6Gdc4h)

[Documentation](https://nonebot.dev/) · [Quick Start](https://nonebot.dev/docs/quick-start) · [Can't open the docs?](#plugin)

<a href="https://asciinema.org/a/569440">
  <img src="https://nonebot.dev/img/setup.svg" alt="setup" >
</a>

## Key Features:

*   **Asynchronous:** Built on Python's async capabilities for high performance and scalability.
*   **Easy Development:** Simplified coding with the NB-CLI scaffolding tool for focused development.
*   **Type-Safe:** 100% type-hinted code for enhanced code quality and reduced errors.
*   **Community-Driven:** A large and active community with extensive resources.
*   **Cross-Platform Support:** Adaptable to various chat platforms with customizable communication protocols.

### Supported Platforms:

NoneBot 2 offers adapters for a wide range of platforms, including:

| Protocol                                                                                                   | Status | Notes                                                                                                |
| :--------------------------------------------------------------------------------------------------------- | :----: | :---------------------------------------------------------------------------------------------------: |
| OneBot ([Repo](https://github.com/nonebot/adapter-onebot), [Spec](https://onebot.dev/))                    |   ✅   | Supports QQ, TG, WeChat Official Accounts, KOOK, and more ([Ecosystem](https://onebot.dev/ecosystem.html)) |
| Telegram ([Repo](https://github.com/nonebot/adapter-telegram), [Spec](https://core.telegram.org/bots/api)) |   ✅   |                                                                                                       |
| Feishu ([Repo](https://github.com/nonebot/adapter-feishu), [Spec](https://open.feishu.cn/document/home/index))   |   ✅   |                                                                                                       |
| GitHub ([Repo](https://github.com/nonebot/adapter-github), [Spec](https://docs.github.com/en/apps))         |   ✅   |                          GitHub APP & OAuth APP                           |
| QQ ([Repo](https://github.com/nonebot/adapter-qq), [Spec](https://bot.q.qq.com/wiki/))       |   ✅   |  QQ Official Interface changes frequently                           |
| Console ([Repo](https://github.com/nonebot/adapter-console))       |   ✅   |    Console Interaction                          |
| Red ([Repo](https://github.com/nonebot/adapter-red), [Spec](https://chrononeko.github.io/QQNTRedProtocol/))       |   ✅   |  QQNT Protocol                           |
| Satori ([Repo](https://github.com/nonebot/adapter-satori), [Spec](https://satori.js.org/zh-CN))       |   ✅   | Supports Onebot, TG, Feishu, WeChat Official Accounts, Koishi and more                             |
| Discord ([Repo](https://github.com/nonebot/adapter-discord), [Spec](https://discord.com/developers/docs/intro)) |   ✅   |                             Discord Bot Protocol                              |
| DoDo ([Repo](https://github.com/nonebot/adapter-dodo), [Spec](https://open.imdodo.com/)) |   ✅   |                               DoDo Bot Protocol                               |
| Kritor ([Repo](https://github.com/nonebot/adapter-kritor), [Spec](https://github.com/KarinJS/kritor))  |   ✅   |                Kritor (OnebotX) Protocol, QQNT 机器人接口标准                  |
| Mirai ([Repo](https://github.com/nonebot/adapter-mirai), [Spec](https://docs.mirai.mamoe.net/mirai-api-http/))   |   ✅   |                                  QQ Protocol                                  |
| Milky ([Repo](https://github.com/nonebot/adapter-milky), [Spec](https://milky.ntqqrev.org/)) |   ✅   |                           QQNT 机器人应用接口标准                          |
| DingTalk ([Repo](https://github.com/nonebot/adapter-ding), [Spec](https://open.dingtalk.com/document/)) |   🤗  |                        Seeking Maintainer (Unavailable)                        |
| Kaiheila ([Repo](https://github.com/Tian-que/nonebot-adapter-kaiheila), [Spec](https://developer.kookapp.cn/)) |   ↗️  |                                Contributed by Community                                 |
| Ntchat ([Repo](https://github.com/JustUndertaker/adapter-ntchat)) |   ↗️  |                           Wechat Protocol, contributed by the community                            |
| MineCraft ([Repo](https://github.com/17TheWord/nonebot-adapter-minecraft)) |   ↗️  |                                 Contributed by Community                                |
| BiliBili Live ([Repo](https://github.com/wwweww/adapter-bilibili)) |   ↗️  |                                 Contributed by Community                                 |
| Walle-Q ([Repo](https://github.com/onebot-walle/nonebot_adapter_walleq)) |   ↗️  |                            QQ Protocol, Contributed by Community                            |
| Villa ([Repo](https://github.com/CMHopeSunshine/nonebot-adapter-villa)) |   ❌   |                     Miyoushe Dabieye Bot Protocol, Officially Retired                     |
| Rocket.Chat ([Repo](https://github.com/IUnlimit/nonebot-adapter-rocketchat), [Spec](https://developer.rocket.chat/)) |   ↗️   |                     Rocket.Chat Bot Protocol, Contributed by Community                      |
| Tailchat ([Repo](https://github.com/eya46/nonebot-adapter-tailchat), [Spec](https://tailchat.msgbyte.com/)) |   ↗️   |                  Tailchat Open Platform Bot Protocol, Contributed by Community                   |
| Mail ([Repo](https://github.com/mobyw/nonebot-adapter-mail)) |   ↗️   |                         Email Protocol, Contributed by Community                          |
| Heybox ([Repo](https://github.com/lclbm/adapter-heybox), [Spec](https://github.com/QingFengOpen/HeychatDoc)) |   ↗️   |                       Heybox Protocol, Contributed by Community                             |
| WeChat Official Account ([Repo](https://github.com/YangRucheng/nonebot-adapter-wxmp), [Spec](https://developers.weixin.qq.com/doc/))|   ↗️   |                       WeChat Official Account Protocol, Contributed by Community                             |
| Gewechat ([Repo](https://github.com/Shine-Light/nonebot-adapter-gewechat), [Spec](https://github.com/Devo919/Gewechat))|   ❌  |                      Gewechat WeChat Protocol, Gewechat is no longer maintained and available                            |
| EFChat ([Repo](https://github.com/molanp/nonebot_adapter_efchat), [Spec](https://irinu-live.melon.fish/efc-help/))   |  ↗️  |                            EF Chat Protocol, Contributed by Community                          |

### Supported Web Frameworks

*   FastAPI
*   Quart (Async Flask)
*   aiohttp
*   httpx
*   websockets

[More Information](https://nonebot.dev/docs/)

## What NoneBot 2 Is Not

NoneBot 2 is not an implementation of a specific platform or protocol. It acts as an intermediary, communicating with existing protocol adapters and processing events. Therefore, questions about specific features on a particular platform should be directed to the platform's documentation or its adapter developers.

NoneBot 2 is not a replacement for NoneBot 1; both are actively maintained. However, NoneBot 2 is recommended for new features and broader platform support.

> ~~The difference between NoneBot 2 and NoneBot 1 is like the difference between Visual Studio Code and Visual Studio~~

## Get Started

Comprehensive documentation is available [here](https://nonebot.dev/).

**Quick Installation Guide:**

1.  Install [pipx](https://pypa.github.io/pipx/)

    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2.  Install the CLI scaffolding tool

    ```bash
    pipx install nb-cli
    ```

3.  Create a new project

    ```bash
    nb create
    ```

4.  Run your project

    ```bash
    nb run
    ```

## Community Resources

### FAQ

-   [FAQ](https://faq.nonebot.dev/)
-   [Discussions](https://discussions.nonebot.dev/)

### Tutorials / Projects / Sharing

-   [awesome-nonebot](https://github.com/nonebot/awesome-nonebot)

### Plugins

Extend NoneBot 2 with a rich selection of official and third-party plugins:

-   [NoneBot-Plugin-Docs](https://github.com/nonebot/nonebot2/tree/master/packages/nonebot-plugin-docs): Offline documentation for local projects (solve documentation issues!)

    In your project directory, run:

    ```bash
    nb plugin install nonebot_plugin_docs
    ```

    Alternatively, use this mirror:

    -   [Documentation Mirror (China)](https://nb2.baka.icu)

-   Find more plugins in the [Store](https://nonebot.dev/store/plugins)

## License

`NoneBot` is licensed under the `MIT` license.

```text
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Contributing

See the [contribution guide](./CONTRIBUTING.md).

## Acknowledgements

### Sponsors

Special thanks to the following for their support of the NoneBot project:

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

And to the following sponsors for their financial support:

<a href="https://assets.nonebot.dev/sponsors.svg">
  <img src="https://assets.nonebot.dev/sponsors.svg" alt="sponsors" />
</a>

### Developers

Thank you to the following developers for their contributions to NoneBot 2:

<a href="https://github.com/nonebot/nonebot2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nonebot/nonebot2&max=1000" alt="contributors" />
</a>