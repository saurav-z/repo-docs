# NoneBot: The Powerful, Cross-Platform Python Asynchronous Chatbot Framework

**Build versatile and responsive chatbots with NoneBot, a modern, asynchronous Python framework, for a seamless experience across multiple platforms.** ([See the original repo](https://github.com/nonebot/nonebot2))

[![License](https://img.shields.io/github/license/nonebot/nonebot2)](https://raw.githubusercontent.com/nonebot/nonebot2/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/nonebot2?logo=python&logoColor=edb641)](https://pypi.python.org/pypi/nonebot2)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue?logo=python&logoColor=edb641)](https://www.python.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=python&logoColor=edb641)](https://github.com/psf/black)
[![Type Checking: Pyright](https://img.shields.io/badge/types-pyright-797952.svg?logo=python&logoColor=edb641)](https://github.com/Microsoft/pyright)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code Coverage](https://codecov.io/gh/nonebot/nonebot2/branch/master/graph/badge.svg?token=2P0G0VS7N4)](https://codecov.io/gh/nonebot/nonebot2)
[![Website Deploy](https://github.com/nonebot/nonebot2/actions/workflows/website-deploy.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/website-deploy.yml)
[![Pre-commit](https://results.pre-commit.ci/badge/github/nonebot/nonebot2/master.svg)](https://results.pre-commit.ci/latest/github/nonebot/nonebot2/master)
[![Pyright](https://github.com/nonebot/nonebot2/actions/workflows/pyright.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/pyright.yml)
[![Ruff](https://github.com/nonebot/nonebot2/actions/workflows/ruff.yml/badge.svg?branch=master&event=push)](https://github.com/nonebot/nonebot2/actions/workflows/ruff.yml)
[![OneBot v11](https://img.shields.io/badge/OneBot-v11-black?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABABAMAAABYR2ztAAAAIVBMVEUAAAAAAAADAwMHBwceHh4UFBQNDQ0ZGRkoKCgvLy8iIiLWSdWYAAAAAXRSTlMAQObYZgAAAQVJREFUSMftlM0RgjAQhV+0ATYK6i1Xb+iMd0qgBEqgBEuwBOxU2QDKsjvojQPvkJ/ZL5sXkgWrFirK4MibYUdE3OR2nEpuKz1/q8CdNxNQgthZCXYVLjyoDQftaKuniHHWRnPh2GCUetR2/9HsMAXyUT4/3UHwtQT2AggSCGKeSAsFnxBIOuAggdh3AKTL7pDuCyABcMb0aQP7aM4AnAbc/wHwA5D2wDHTTe56gIIOUA/4YYV2e1sg713PXdZJAuncdZMAGkAukU9OAn40O849+0ornPwT93rphWF0mgAbauUrEOthlX8Zu7P5A6kZyKCJy75hhw1Mgr9RAUvX7A3csGqZegEdniCx30c3agAAAABJRU5ErkJggg==")](https://onebot.dev/)
[![OneBot v12](https://img.shields.io/badge/OneBot-v12-black?style=social&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABABAMAAABYR2ztAAAAIVBMVEUAAAAAAAADAwMHBwceHh4UFBQNDQ0ZGRkoKCgvLy8iIiLWSdWYAAAAAXRSTlMAQObYZgAAAQVJREFUSMftlM0RgjAQhV+0ATYK6i1Xb+iMd0qgBEqgBEuwBOxU2QDKsjvojQPvkJ/ZL5sXkgWrFirK4MibYUdE3OR2nEpuKz1/q8CdNxNQgthZCXYVLjyoDQftaKuniHHWRnPh2GCUetR2/9HsMAXyUT4/3UHwtQT2AggSCGKeSAsFnxBIOuAggdh3AKTL7pDuCyABcMb0aQP7aM4AnAbc/wHwA5D2wDHTTe56gIIOUA/4YYV2e1sg713PXdZJAuncdZMAGkAukU9OAn40O849+0ornPwT93rphWF0mgAbauUrEOthlX8Zu7P5A6kZyKCJy75hhw1Mgr9RAUvX7A3csGqZegEdniCx30c3agAAAABJRU5ErkJggg==")](https://onebot.dev/)
[![QQ Bot](https://img.shields.io/badge/QQ-Bot-lightgrey?style=social&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMTIuODIgMTMwLjg5Ij48ZyBkYXRhLW5hbWU9IuWbvuWxgiAyIj48ZyBkYXRhLW5hbWU9IuWbvuWxgiAxIj48cGF0aCBkPSJNNTUuNjMgMTMwLjhjLTcgMC0xMy45LjA4LTIwLjg2IDAtMTkuMTUtLjI1LTMxLjcxLTExLjQtMzQuMjItMzAuMy00LjA3LTMwLjY2IDE0LjkzLTU5LjIgNDQuODMtNjYuNjQgMi0uNTEgNS4yMS0uMzEgNS4yMS0xLjYzIDAtMi4xMy4xNC0yLjEzLjE0LTUuNTcgMC0uODktMS4zLTEuNDYtMi4yMi0yLjMxLTYuNzMtNi4yMy03LjY3LTEzLjQxLTEtMjAuMTggNS40LTUuNTIgMTEuODctNS40IDE3LjgtLjU5IDYuNDkgNS4yNiA2LjMxIDEzLjA4LS44NiAyMS0uNjguNzQtMS43OCAxLjYtMS43OCAyLjY3djQuMjFjMCAxLjM1IDIuMiAxLjYyIDQuNzkgMi4zNSAzMS4wOSA4LjY1IDQ4LjE3IDM0LjEzIDQ1IDY2LjM3LTEuNzYgMTguMTUtMTQuNTYgMzAuMjMtMzIuNyAzMC42My04LjAyLjE5LTE2LjA3LS4wMS0yNC4xMy0uMDF6IiBmaWxsPSIjMDI5OWZlIi8+PHBhdGggZD0iTTMxLjQ2IDExOC4zOGMtMTAuNS0uNjktMTYuOC02Ljg2LTE4LjM4LTE3LjI3LTMtMTkuNDIgMi43OC0zNS44NiAxOC40Ni00Ny44MyAxNC4xNi0xMC44IDI5Ljg3LTEyIDQ1LjM4LTMuMTkgMTcuMjUgOS44NCAyNC41OSAyNS44MSAyNCA0NS4yOS0uNDkgMTUuOS04LjQyIDIzLjE0LTI0LjM4IDIzLjUtNi41OS4xNC0xMy4xOSAwLTE5Ljc5IDAiIGZpbGw9IiNmZWZlZmUiLz48cGF0aCBkPSJNNDYuMDUgNzkuNThjLjA5IDUgLjIzIDkuODItNyA5Ljc3LTcuODItLjA2LTYuMS01LjY5LTYuMjQtMTAuMTktLjE1LTQuODItLjczLTEwIDYuNzMtOS44NHM2LjM3IDUuNTUgNi41MSAxMC4yNnoiIGZpbGw9IiMxMDlmZmUiLz48cGF0aCBkPSJNODAuMjcgNzkuMjdjLS41MyAzLjkxIDEuNzUgOS42NC01Ljg4IDEwLTcuNDcuMzctNi44MS00LjgyLTYuNjEtOS41LjItNC4zMi0xLjgzLTEwIDUuNzgtMTAuNDJzNi41OSA0Ljg5IDYuNzEgOS45MnoiIGZpbGw9IiMwODljZmUiLz48L2c+PC9nPjwvc3ZnPg==")](https://bot.q.qq.com/wiki/)
[![Telegram Bot](https://img.shields.io/badge/telegram-Bot-lightgrey?style=social&logo=telegram)](https://core.telegram.org/bots/api)
[![Feishu Bot](https://img.shields.io/badge/%E9%A3%9E%E4%B9%A6-Bot-lightgrey?style=social&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz48c3ZnIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDQ4IDQ4IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xNyAyOUMyMSAyOSAyNSAyNi45MzM5IDI4IDIzLjQwNjVDMzYgMTQgNDEuNDI0MiAxNi44MTY2IDQ0IDE3Ljk5OThDMzguNSAyMC45OTk4IDQwLjUgMjkuNjIzMyAzMyAzNS45OTk4QzI4LjM4MiAzOS45MjU5IDIzLjQ5NDUgNDEuMDE0IDE5IDQxQzEyLjUyMzEgNDAuOTc5OSA2Ljg2MjI2IDM3Ljc2MzcgNCAzNS40MDYzVjE2Ljk5OTgiIHN0cm9rZT0iIzMzMyIgc3Ryb2tlLXdpZHRoPSI0IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48cGF0aCBkPSJNNS42NDgwOCAxNS44NjY5QzUuMDIyMzEgMTQuOTU2NyAzLjc3NzE1IDE0LjcyNjEgMi44NjY5NCAxNS4zNTE5QzEuOTU2NzMgMTUuOTc3NyAxLjcyNjE1IDE3LjIyMjggMi4zNTE5MiAxOC4xMzMxTDUuNjQ4MDggMTUuODY2OVpNMzYuMDAyMSAzNS43MzA5QzM2Ljk1OCAzNS4xNzc0IDM3LjI4NDMgMzMuOTUzOSAzNi43MzA5IDMyLjk5NzlDMzYuMTc3NCAzMi4wNDIgMzQuOTUzOSAzMS43MTU3IDMzLjk5NzkgMzIuMjY5MUwzNi4wMDIxIDM1LjczMDlaTTIuMzUxOTIgMTguMTMzMUM1LjI0MzUgMjIuMzM5IDEwLjc5OTIgMjguMTQ0IDE2Ljg4NjUgMzIuMjIzOUMxOS45MzQ1IDM0LjI2NjcgMjMuMjE3IDM1Ljk0NiAyNi40NDkgMzYuNzMyNEMyOS42OTQ2IDM3LjUyMiAzMy4wNDUxIDM3LjQ0MjggMzYuMDAyMSAzNS43MzA5TDMzLjk5NzkgMzIuMjY5MUMzMi4yMDQ5IDMzLjMwNzIgMjkuOTkyOSAzMy40NzggMjcuMzk0NyAzMi44NDU4QzI0Ljc4MyAzMi4yMTAzIDIxLjk0MDUgMzAuNzk1OCAxOS4xMTM1IDI4LjkwMTFDMTMuNDUwOCAyNS4xMDYgOC4yNTY1IDE5LjY2MSA1LjY0ODA4IDE1Ljg2NjlMMi4zNTE5MiAxOC4xMzMxWiIgZmlsbD0iIzMzMyIvPjxwYXRoIGQ9Ik0zMy41OTQ1IDE3QzMyLjgzOTggMTQuNzAyNyAzMC44NTQ5IDkuOTQwNTQgMjcuNTk0NSA3SDExLjU5NDVDMTUuMjE3MSAxMC42NzU3IDIzIDE2IDI3IDI0IiBzdHJva2U9IiMzMzMiIHN0cm9rZS13aWR0aD0iNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PC9zdmc+" alt="feishu">
[![GitHub Bot](https://img.shields.io/badge/GitHub-Bot-181717?style=social&logo=github)](https://docs.github.com/en/developers/apps)

<p align="center">
  <a href="https://nonebot.dev/">Documentation</a>
  ·
  <a href="https://nonebot.dev/docs/quick-start">Quick Start</a>
  ·
  <a href="#plugins">Plugins</a>
</p>

<p align="center">
  <a href="https://asciinema.org/a/569440">
    <img src="https://nonebot.dev/img/setup.svg" alt="setup" >
  </a>
</p>

## Key Features

*   **Asynchronous Architecture:** Built on Python's async capabilities for efficient handling of high message volumes.
*   **Ease of Development:** Streamlined code with the NB-CLI scaffolding tool, allowing developers to focus on business logic.
*   **Reliable and Type-Safe:** 100% type-annotated code with editor support to catch bugs early.
*   **Active Community:**  Benefit from a large and active community with thousands of users. ([Community Resources](#community-resources))
*   **Cross-Platform Support:** A single framework supporting multiple chat platforms with customizable communication protocols.

## Supported Protocols

NoneBot supports a wide range of platforms through adapters:

| Protocol                                                                                                              | Status | Description                                                                                                    |
| :-------------------------------------------------------------------------------------------------------------------: | :----: | :-------------------------------------------------------------------------------------------------------------: |
|               OneBot ([Repo](https://github.com/nonebot/adapter-onebot), [Spec](https://onebot.dev/))               |   ✅   | Supports QQ, Telegram, WeChat Official Accounts, KOOK and more ([Platform list](https://onebot.dev/ecosystem.html)) |
|    Telegram ([Repo](https://github.com/nonebot/adapter-telegram), [Spec](https://core.telegram.org/bots/api))    |   ✅   |                                                                                                               |
|     Feishu ([Repo](https://github.com/nonebot/adapter-feishu), [Spec](https://open.feishu.cn/document/home/index))    |   ✅   |                                                                                                               |
|      GitHub ([Repo](https://github.com/nonebot/adapter-github), [Spec](https://docs.github.com/en/apps))      |   ✅   |                               GitHub APP & OAuth APP                                                              |
|             QQ ([Repo](https://github.com/nonebot/adapter-qq), [Spec](https://bot.q.qq.com/wiki/))            |   ✅   |                             QQ official interface changes frequently                                               |
|                          Console ([Repo](https://github.com/nonebot/adapter-console))                          |   ✅   |                                          Console interaction                                                      |
|     Red ([Repo](https://github.com/nonebot/adapter-red), [Spec](https://chrononeko.github.io/QQNTRedProtocol/))     |   ✅   |                                          QQNT protocol                                                               |
|            Satori ([Repo](https://github.com/nonebot/adapter-satori), [Spec](https://satori.js.org/zh-CN))            |   ✅   |                  Supports Onebot, Telegram, Feishu, WeChat Official Accounts, Koishi and more                  |
|   Discord ([Repo](https://github.com/nonebot/adapter-discord), [Spec](https://discord.com/developers/docs/intro))    |   ✅   |                                         Discord Bot protocol                                                      |
|             DoDo ([Repo](https://github.com/nonebot/adapter-dodo), [Spec](https://open.imdodo.com/))             |   ✅   |                                         DoDo Bot protocol                                                        |
|       Kritor ([Repo](https://github.com/nonebot/adapter-kritor), [Spec](https://github.com/KarinJS/kritor))        |   ✅   |               Kritor (OnebotX) protocol, QQNT Bot interface standard                                                 |
|    Mirai ([Repo](https://github.com/nonebot/adapter-mirai), [Spec](https://docs.mirai.mamoe.net/mirai-api-http/))    |   ✅   |                                         QQ protocol                                                            |
|    Milky ([Repo](https://github.com/nonebot/adapter-milky), [Spec](https://milky.ntqqrev.org/))                      |   ✅   |                          QQNT Bot application interface standard                                                   |
|        DingTalk ([Repo](https://github.com/nonebot/adapter-ding), [Spec](https://open.dingtalk.com/document/))       |   🤗   |                            Seeking Maintainer (currently unavailable)                                              |
|   Kaiheila ([Repo](https://github.com/Tian-que/nonebot-adapter-kaiheila), [Spec](https://developer.kookapp.cn/))   |   ↗️   |                                         Community-contributed                                                    |
|    Ntchat ([Repo](https://github.com/JustUndertaker/adapter-ntchat))    |   ↗️   |                                         WeChat protocol, Community-contributed                                                   |
|    MineCraft ([Repo](https://github.com/17TheWord/nonebot-adapter-minecraft))   |   ↗️   |                                         Community-contributed                                                    |
|    BiliBili Live ([Repo](https://github.com/wwweww/adapter-bilibili))   |   ↗️   |                                         Community-contributed                                                   |
|   Walle-Q ([Repo](https://github.com/onebot-walle/nonebot_adapter_walleq))   |   ↗️   |                                          QQ protocol, Community-contributed                                                  |
|   Villa ([Repo](https://github.com/CMHopeSunshine/nonebot-adapter-villa))  |   ❌   |                         Mihoyo's Villa Bot protocol, officially offline                                             |
|   Rocket.Chat ([Repo](https://github.com/IUnlimit/nonebot-adapter-rocketchat), [Spec](https://developer.rocket.chat/))  |   ↗️   |                        Rocket.Chat Bot protocol, Community-contributed                                                 |
|   Tailchat ([Repo](https://github.com/eya46/nonebot-adapter-tailchat), [Spec](https://tailchat.msgbyte.com/))   |   ↗️   |                      Tailchat open platform Bot protocol, Community-contributed                   |
|    Mail ([Repo](https://github.com/mobyw/nonebot-adapter-mail))   |   ↗️   |                                        Email sending and receiving protocol, Community-contributed                                             |
|    Heybox ([Repo](https://github.com/lclbm/adapter-heybox), [Spec](https://github.com/QingFengOpen/HeychatDoc))   |   ↗️   |                       Heybox Bot protocol, Community-contributed                             |
|   WeChat Official Accounts ([Repo](https://github.com/YangRucheng/nonebot-adapter-wxmp), [Spec](https://developers.weixin.qq.com/doc/))   |   ↗️   |                       WeChat Official Accounts protocol, Community-contributed                             |
|   Gewechat ([Repo](https://github.com/Shine-Light/nonebot-adapter-gewechat), [Spec](https://github.com/Devo919/Gewechat))  |   ❌   |                         Gewechat WeChat protocol, no longer maintained                                              |
|   EFChat ([Repo](https://github.com/molanp/nonebot_adapter_efchat), [Spec](https://irinu-live.melon.fish/efc-help/))   |   ↗️   |                            EF Chat protocol, Community-contributed                                                        |


## Web Framework Support

NoneBot seamlessly integrates with various web frameworks for flexibility:

| Framework                                                   | Type   |
| :----------------------------------------------------------: | :----: |
|       [FastAPI](https://fastapi.tiangolo.com/)       | Server |
| [Quart](https://quart.palletsprojects.com/en/latest/) (Async Flask)  | Server |
|    [aiohttp](https://docs.aiohttp.org/en/stable/)    | Client |
|    [httpx](https://www.python-httpx.org/)    | Client |
|  [websockets](https://websockets.readthedocs.io/en/stable/)  | Client |

## Getting Started

Quick installation guide:

1.  Install [pipx](https://pypa.github.io/pipx/)

    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2.  Install the scaffolding tool

    ```bash
    pipx install nb-cli
    ```

3.  Create a project

    ```bash
    nb create
    ```

4.  Run the project

    ```bash
    nb run
    ```

## Community Resources

### Common Issues

*   [FAQ](https://faq.nonebot.dev/)
*   [Discussions](https://discussions.nonebot.dev/)

### Tutorials, Projects & Experiences

*   [awesome-nonebot](https://github.com/nonebot/awesome-nonebot)

## Plugins

Enhance your NoneBot experience with a rich ecosystem of official and third-party plugins:

*   [NoneBot-Plugin-Docs](https://github.com/nonebot/nonebot2/tree/master/packages/nonebot-plugin-docs): Offline documentation for your project (No more broken documentation!)

    Run this inside your project directory:

    ```bash
    nb plugin install nonebot_plugin_docs
    ```
    or try the mirror
    - [Doc Mirror (China)](https://nb2.baka.icu)

*   Find more plugins at the [Store](https://nonebot.dev/store/plugins)

## License

NoneBot is open-source under the [MIT License](https://raw.githubusercontent.com/nonebot/nonebot2/master/LICENSE).

## Contribution

See the [Contribution Guide](./CONTRIBUTING.md) for information.

## Acknowledgements

### Sponsors

Thanks to the following sponsors for their support of the NoneBot project:

<!-- GitHub Sponsors -->
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

<!-- Open Collective or other financial support -->
<a href="https://assets.nonebot.dev/sponsors.svg">
  <img src="https://assets.nonebot.dev/sponsors.svg" alt="sponsors" />
</a>

### Contributors

Thanks to the many contributors:

<a href="https://github.com/nonebot/nonebot2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nonebot/nonebot2&max=1000" alt="contributors" />
</a>