# NoneBot 2: Build Powerful, Cross-Platform Python Chatbots

**Create versatile chatbots with NoneBot 2, a modern, asynchronous Python framework for multiple platforms.** Dive deeper into the [original repository](https://github.com/nonebot/nonebot2).

[![License](https://img.shields.io/github/license/nonebot/nonebot2)](https://raw.githubusercontent.com/nonebot/nonebot2/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/nonebot2?logo=python&logoColor=edb641)](https://pypi.python.org/pypi/nonebot2)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue?logo=python&logoColor=edb641)](https://www.python.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=python&logoColor=edb641)](https://github.com/psf/black)
[![Type Checking: Pyright](https://img.shields.io/badge/types-pyright-797952.svg?logo=python&logoColor=edb641)](https://github.com/Microsoft/pyright)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code Coverage](https://codecov.io/gh/nonebot/nonebot2/branch/master/graph/badge.svg?token=2P0G0VS7N4)](https://codecov.io/gh/nonebot/nonebot2)
[![Website Deployment](https://github.com/nonebot/nonebot2/actions/workflows/website-deploy.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/website-deploy.yml)
[![Pre-commit](https://results.pre-commit.ci/badge/github/nonebot/nonebot2/master.svg)](https://results.pre-commit.ci/latest/github/nonebot/nonebot2/master)
[![Pyright Checks](https://github.com/nonebot/nonebot2/actions/workflows/pyright.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/pyright.yml)
[![Ruff Checks](https://github.com/nonebot/nonebot2/actions/workflows/ruff.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/ruff.yml)
[![OneBot v11](https://img.shields.io/badge/OneBot-v11-black?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABABAMAAABYR2ztAAAAIVBMVEUAAAAAAAADAwMHBwceHh4UFBQNDQ0ZGRkoKCgvLy8iIiLWSdWYAAAAAXRSTlMAQObYZgAAAQVJREFUSMftlM0RgjAQhV+0ATYK6i1Xb+iMd0qgBEqgBEuwBOxU2QDKsjvojQPvkJ/ZL5sXkgWrFirK4MibYUdE3OR2nEpuKz1/q8CdNxNQgthZCXYVLjyoDQftaKuniHHWRnPh2GCUetR2/9HsMAXyUT4/3UHwtQT2AggSCGKeSAsFnxBIOuAggdh3AKTL7pDuCyABcMb0aQP7aM4AnAbc/wHwA5D2wDHTTe56gIIOUA/4YYV2e1sg713PXdZJAuncdZMAGkAukU9OAn40O849+0ornPwT93rphWF0mgAbauUrEOthlX8Zu7P5A6kZyKCJy75hhw1Mgr9RAUvX7A3csGqZegEdniCx30c3agAAAABJRU5ErkJggg==")](https://onebot.dev/)
[![OneBot v12](https://img.shields.io/badge/OneBot-v12-black?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABABAMAAABYR2ztAAAAIVBMVEUAAAAAAAADAwMHBwceHh4UFBQNDQ0ZGRkoKCgvLy8iIiLWSdWYAAAAAXRSTlMAQObYZgAAAQVJREFUSMftlM0RgjAQhV+0ATYK6i1Xb+iMd0qgBEqgBEuwBOxU2QDKsjvojQPvkJ/ZL5sXkgWrFirK4MibYUdE3OR2nEpuKz1/q8CdNxNQgthZCXYVLjyoDQftaKuniHHWRnPh2GCUetR2/9HsMAXyUT4/3UHwtQT2AggSCGKeSAsFnxBIOuAggdh3AKTL7pDuCyABcMb0aQP7aM4AnAbc/wHwA5D2wDHTTe56gIIOUA/4YYV2e1sg713PXdZJAuncdZMAGkAukU9OAn40O849+0ornPwT93rphWF0mgAbauUrEOthlX8Zu7P5A6kZyKCJy75hhw1Mgr9RAUvX7A3csGqZegEdniCx30c3agAAAABJRU5ErkJggg==")](https://onebot.dev/)
[![QQ Bot](https://img.shields.io/badge/QQ-Bot-lightgrey?style=social&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMTIuODIgMTMwLjg5Ij48ZyBkYXRhLW5hbWU9IuWbvuWxgiAyIj48ZyBkYXRhLW5hbWU9IuWbvuWxgiAxIj48cGF0aCBkPSJNNTUuNjMgMTMwLjhjLTcgMC0xMy45LjA4LTIwLjg2IDAtMTkuMTUtLjI1LTMxLjcxLTExLjQtMzQuMjItMzAuMy00LjA3LTMwLjY2IDE0LjkzLTU5LjIgNDQuODMtNjYuNjQgMi0uNTEgNS4yMS0uMzEgNS4yMS0xLjYzIDAtMi4xMy4xNC0yLjEzLjE0LTUuNTcgMC0uODktMS4zLTEuNDYtMi4yMi0yLjMxLTYuNzMtNi4yMy03LjY3LTEzLjQxLTEtMjAuMTggNS40LTUuNTIgMTEuODctNS40IDE3LjgtLjU5IDYuNDkgNS4yNiA2LjMxIDEzLjA4LS44NiAyMS0uNjguNzQtMS43OCAxLjYtMS43OCAyLjY3djQuMjFjMCAxLjM1IDIuMiAxLjYyIDQuNzkgMi4zNSAzMS4wOSA4LjY1IDQ4LjE3IDM0LjEzIDQ1IDY2LjM3LTEuNzYgMTguMTUtMTQuNTYgMzAuMjMtMzIuNyAzMC42My04LjAyLjE5LTE2LjA3LS4wMS0yNC4xMy0uMDF6IiBmaWxsPSIjMDI5OWZlIi8+PHBhdGggZD0iTTMxLjQ2IDExOC4zOGMtMTAuNS0uNjktMTYuOC02Ljg2LTE4LjM4LTE3LjI3LTMtMTkuNDIgMi43OC0zNS44NiAxOC40Ni00Ny44MyAxNC4xNi0xMC44IDI5Ljg3LTEyIDQ1LjM4LTMuMTkgMTcuMjUgOS44NCAyNC41OSAyNS44MSAyNCA0NS4yOS0uNDkgMTUuOS04LjQyIDIzLjE0LTI0LjM4IDIzLjUtNi41OS4xNC0xMy4xOSAwLTE5Ljc5IDAiIGZpbGw9IiNmZWZlZmUiLz48cGF0aCBkPSJNNDYuMDUgNzkuNThjLjA5IDUgLjIzIDkuODItNyA5Ljc3LTcuODItLjA2LTYuMS01LjY5LTYuMjQtMTAuMTktLjE1LTQuODItLjczLTEwIDYuNzMtOS44NHM2LjM3IDUuNTUgNi41MSAxMC4yNnoiIGZpbGw9IiMxMDlmZmUiLz48cGF0aCBkPSJNODAuMjcgNzkuMjdjLS41MyAzLjkxIDEuNzUgOS42NC01Ljg4IDEwLTcuNDcuMzctNi44MS00LjgyLTYuNjEtOS41LjItNC4zMi0xLjgzLTEwIDUuNzgtMTAuNDJzNi41OSA0Ljg5IDYuNzEgOS45MnoiIGZpbGw9IiMwODljZmUiLz48L2c+PC9nPjwvc3ZnPg==")](https://bot.q.qq.com/wiki/)
[![Telegram Bot](https://img.shields.io/badge/telegram-Bot-lightgrey?style=social&logo=telegram)](https://core.telegram.org/bots/api)
[![Feishu Bot](https://img.shields.io/badge/%E9%A3%9E%E4%B9%A6-Bot-lightgrey?style=social&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz48c3ZnIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDQ4IDQ4IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xNyAyOUMyMSAyOSAyNSAyNi45MzM5IDI4IDIzLjQwNjVDMzYgMTQgNDEuNDI0MiAxNi44MTY2IDQ0IDE3Ljk5OThDMzguNSAyMC45OTk4IDQwLjUgMjkuNjIzMyAzMyAzNS45OTk4QzI4LjM4MiAzOS45MjU5IDIzLjQ5NDUgNDEuMDE0IDE5IDQxQzEyLjUyMzEgNDAuOTc5OSA2Ljg2MjI2IDM3Ljc2MzcgNCAzNS40MDYzVjE2Ljk5OTgiIHN0cm9rZT0iIzMzMyIgc3Ryb2tlLXdpZHRoPSI0IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48cGF0aCBkPSJNNS42NDgwOCAxNS44NjY5QzUuMDIyMzEgMTQuOTU2NyAzLjc3NzE1IDE0LjcyNjEgMi44NjY5NCAxNS4zNTE5QzEuOTU2NzMgMTUuOTc3NyAxLjcyNjE1IDE3LjIyMjggMi4zNTE5MiAxOC4xMzMxTDUuNjQ4MDggMTUuODY2OVpNMzYuMDAyMSAzNS43MzA5QzM2Ljk1OCAzNS4xNzc0IDM3LjI4NDMgMzMuOTUzOSAzNi43MzA5IDMyLjk5NzlDMzYuMTc3NCAzMi4wNDIgMzQuOTUzOSAzMS43MTU3IDMzLjk5NzkgMzIuMjY5MUwzNi4wMDIxIDM1LjczMDlaTTIuMzUxOTIgMTguMTMzMUM1LjI0MzUgMjIuMzM5IDEwLjc5OTIgMjguMTQ0IDE2Ljg4NjUgMzIuMjIzOUMxOS45MzQ1IDM0LjI2NjcgMjMuMjE3IDM1Ljk0NiAyNi40NDkgMzYuNzMyNEMyOS42OTQ2IDM3LjUyMiAzMy4wNDUxIDM3LjQ0MjggMzYuMDAyMSAzNS43MzA5TDMzLjk5NzkgMzIuMjY5MUMzMi4yMDQ5IDMzLjMwNzIgMjkuOTkyOSAzMy40NzggMjcuMzk0NyAzMi44NDU4QzI0Ljc4MyAzMi4yMTAzIDIxLjk0MDUgMzAuNzk1OCAxOS4xMTM1IDI4LjkwMTFDMTMuNDUwOCAyNS4xMDYgOC4yNTY1IDE5LjY2MSA1LjY0ODA4IDE1Ljg2NjlMMi4zNTE5MiAxOC4xMzMxWiIgZmlsbD0iIzMzMyIvPjxwYXRoIGQ9Ik0zMy41OTQ1IDE3QzMyLjgzOTggMTQuNzAyNyAzMC44NTQ5IDkuOTQwNTQgMjcuNTk0NSA3SDExLjU5NDVDMTUuMjE3MSAxMC42NzU3IDIzIDE2IDI3IDI0IiBzdHJva2U9IiMzMzMiIHN0cm9rZS13aWR0aD0iNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PC9zdmc+" alt="feishu">
[![GitHub Bot](https://img.shields.io/badge/GitHub-Bot-181717?style=social&logo=github)](https://docs.github.com/en/developers/apps)
<!-- [![DingTalk Bot](https://img.shields.io/badge/%E9%92%89%E9%92%89-Bot-lightgrey?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAnFBMVEUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4jUzeAAAAM3RSTlMAQKSRaA+/f0YyFevh29R3cyklIfrlyrGsn41tVUs48c/HqJm9uZdhX1otGwkF9IN8V1CX0Q+IAAABY0lEQVRYw+3V2W7CMBAF0JuNQAhhX9OEfYdu9///rUVWpagE27Ef2gfO+0zGozsKnv6bMGzAhkNytIe5gDdzrwtTCwrbI8x4/NF668NAxgI3Q3UtFi3TyPwNQtPLUUmDd8YfqGLNe4v22XwEYb5zoOuF5baHq2UHtsKe5ivWfGAwrWu2mC34QM0PoCAuqZdOmiwV+5BLyMRtZ7dTSEcs48rzWfzwptMLyzpApka1SJ5FtR4kfCqNIBPEVDmqoqgwUYY5plQOlf6UEjNoOPnuKB6wzDyCrks///TDza8+PnR109WQdxLo8RKWq0PPnuXG0OXKQ6wWLFnCg75uYYbhmMIVVdQ709q33aHbGIj6Duz+2k1HQFX9VwqmY8xYsEJll2ahvhWgsjYLHFRXvIi2Qb0jzMQCzC3FAoydxCma88UCzE3JCWwkjCNYyMUCzHX4DiuTMawEwwhW6hnshPhjZzzJfAH0YacpbmRd7QAAAABJRU5ErkJggg==")](https://ding-doc.dingtalk.com/document#/org-dev-guide/elzz1p) -->
[![QQ Group](https://img.shields.io/badge/QQ%E7%BE%A4-768887710-orange?style=flat-square)](https://jq.qq.com/?_wv=1027&k=5OFifDh)
[![QQ Channel](https://img.shields.io/badge/QQ%E9%A2%91%E9%81%93-NoneBot-5492ff?style=flat-square)](https://qun.qq.com/qqweb/qunpro/share?_wv=3&_wwv=128&appChannel=share&inviteCode=7b4a3&appChannel=share&businessType=9&from=246610&biz=ka)
[![Telegram Channel](https://img.shields.io/badge/telegram-botuniverse-blue?style=flat-square)](https://t.me/botuniverse)
[![Discord Server](https://discordapp.com/api/guilds/847819937858584596/widget.png?style=shield)](https://discord.gg/VKtE6Gdc4h)

---

## Key Features

*   **Asynchronous by Design:** Efficiently handles a high volume of messages with Python's async capabilities.
*   **Developer-Friendly:**  Streamlines development with the NB-CLI scaffolding tool, focusing your efforts on bot logic.
*   **Reliable & Type-Safe:** Benefit from 100% type annotations and editor integration for robust code and reduced bugs.
*   **Community-Driven:** Leverage an active community with tens of thousands of users and extensive resources.
*   **Cross-Platform Support:** Build bots for multiple chat platforms with adaptable communication protocols.

## Platform Adapters

NoneBot 2 supports a wide array of platforms, with adapters available for various protocols:

| Protocol                                                                                                           | Status |                                  Notes                                   |
| :------------------------------------------------------------------------------------------------------------------ | :----: | :------------------------------------------------------------------------: |
| [OneBot](https://onebot.dev/) (QQ, Telegram, WeChat Official Accounts, KOOK, etc.)                                 |   ✅   |   Supports multiple platforms listed in the [OneBot ecosystem](https://onebot.dev/ecosystem.html)     |
| [Telegram](https://core.telegram.org/bots/api)                                                                     |   ✅   |                                                                            |
| [Feishu](https://open.feishu.cn/document/home/index)                                                                |   ✅   |                                                                            |
| [GitHub](https://docs.github.com/en/apps) (GitHub App & OAuth APP)                                                 |   ✅   |                                                                            |
| [QQ](https://bot.q.qq.com/wiki/)                                                                                    |   ✅   |                     Official QQ interface is frequently updated                      |
| Console                                                                                                          |   ✅   |                                 Console interaction                                  |
| [Red](https://chrononeko.github.io/QQNTRedProtocol/) (QQNT Protocol)                                                 |   ✅   |                                                                            |
| [Satori](https://satori.js.org/zh-CN)  (Onebot, TG, Feishu, WeChat Official Accounts, Koishi, etc.)                   |   ✅   |                                                                            |
| [Discord](https://discord.com/developers/docs/intro)                                                                 |   ✅   |                                 Discord Bot Protocol                                  |
| [DoDo](https://open.imdodo.com/)                                                                                     |   ✅   |                                  DoDo Bot Protocol                                   |
| [Kritor](https://github.com/KarinJS/kritor) (OnebotX) (QQNT robot interface standard)                                  |   ✅   |                                                                            |
| [Mirai](https://docs.mirai.mamoe.net/mirai-api-http/)                                                                |   ✅   |                                    QQ Protocol                                     |
| [Milky](https://milky.ntqqrev.org/) (QQNT Robot Application Interface Standard)                                       |   ✅   |                                                                            |
| [DingTalk](https://open.dingtalk.com/document/)                                                                    |   🤗   |           Seeking a maintainer (currently unavailable)            |
| [Kaiheila](https://developer.kookapp.cn/) (Community contribution)                                                    |   ↗️   |                                 Community Contributed                                 |
| Ntchat (WeChat Protocol, Community contribution)                                                                     |   ↗️   |                                 Community Contributed                                 |
| Minecraft (Community contribution)                                                                                  |   ↗️   |                                 Community Contributed                                 |
| BiliBili Live (Community contribution)                                                                             |   ↗️   |                                 Community Contributed                                 |
| Walle-Q (QQ Protocol, Community contribution)                                                                        |   ↗️   |                                 Community Contributed                                 |
| Villa (MiHoYo's Big Villa Bot Protocol, Official Shutdown)                                                          |   ❌   |                                                                            |
| [Rocket.Chat](https://developer.rocket.chat/) (Rocket.Chat Bot Protocol, Community contribution)                       |   ↗️   |                                 Community Contributed                                 |
| [Tailchat](https://tailchat.msgbyte.com/) (Tailchat Open Platform Bot Protocol, Community contribution)                 |   ↗️   |                                 Community Contributed                                 |
| Mail (Email Send/Receive Protocol, Community Contribution)                                                           |   ↗️   |                                 Community Contributed                                 |
| [Heybox](https://github.com/QingFengOpen/HeychatDoc) (Heybox Robot Protocol, Community Contribution)                                |   ↗️   |                                 Community Contributed                                 |
| WeChat Official Account ([WeChat Official Account Protocol, Community Contribution](https://developers.weixin.qq.com/doc/))                                |   ↗️   |                                 Community Contributed                                 |
| Gewechat (Gewechat WeChat Protocol, Gewechat no longer maintained and available)                                                           |   ❌   |                                                                            |
| [EFChat](https://irinu-live.melon.fish/efc-help/) (Hengwu Chat Platform Protocol, Community Contribution)                                |   ↗️   |                                 Community Contributed                                 |

### Supported Web Frameworks
Integrates with several web frameworks for flexible configuration.

| Framework                                                                         | Type     |
| :-------------------------------------------------------------------------------- | :-------: |
| [FastAPI](https://fastapi.tiangolo.com/)                                        |  Server   |
| [Quart](https://quart.palletsprojects.com/en/latest/) (Asynchronous Flask)        |  Server   |
| [aiohttp](https://docs.aiohttp.org/en/stable/)                                 |  Client   |
| [httpx](https://www.python-httpx.org/)                                          |  Client   |
| [websockets](https://websockets.readthedocs.io/en/stable/)                      |  Client   |

For more information: [Documentation](https://nonebot.dev/docs/)

---

## What NoneBot 2 Isn't

NoneBot 2 is a framework that facilitates communication with various platforms and handles events, not a specific platform's implementation. Questions about specific features on a platform should be directed to its documentation or the adapter developers.

It's also not a direct replacement for NoneBot 1. Both are actively maintained.  Choose NoneBot 2 if you want new features or broader platform support.

## Get Started

Get started by consulting the [Documentation](https://nonebot.dev/).

**Quick Installation Guide:**

1.  Install [pipx](https://pypa.github.io/pipx/)
    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2.  Install the scaffolding tool
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

### Troubleshooting & Help

*   [FAQ (Frequently Asked Questions)](https://faq.nonebot.dev/)
*   [Discussion Forum](https://discussions.nonebot.dev/)

### Tutorials, Projects, & Sharing

*   [awesome-nonebot](https://github.com/nonebot/awesome-nonebot)

### Plugins

Extend NoneBot 2's functionality with a rich ecosystem of plugins:

*   [NoneBot-Plugin-Docs](https://github.com/nonebot/nonebot2/tree/master/packages/nonebot-plugin-docs): Integrate offline documentation directly into your project.
    To install, navigate to your project directory and run:
    ```bash
    nb plugin install nonebot_plugin_docs
    ```
    Alternatively, try this mirror: [Documentation Mirror (China)](https://nb2.baka.icu)

*   Browse the [Plugin Store](https://nonebot.dev/store/plugins) for more available plugins.

## License

NoneBot is open-sourced under the MIT license.

```text
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Contributing

See the [Contribution Guide](./CONTRIBUTING.md).

## Acknowledgements

### Sponsors

[GitHub](https://github.com/)
<br/>
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
<br/>
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

<a href="https://www.jetbrains.com/">
  <img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.svg" height="80" alt="JetBrains" >
</a>

Financial support is also provided by the sponsors below:
<a href="https://assets.nonebot.dev/sponsors.svg">
  <img src="https://assets.nonebot.dev/sponsors.svg" alt="sponsors" />
</a>

### Contributors

Thanks to all the contributors!
<a href="https://github.com/nonebot/nonebot2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nonebot/nonebot2&max=1000" alt="contributors" />
</a>