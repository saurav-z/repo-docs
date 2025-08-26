# OpenHands: Build Software Faster with AI-Powered Agents

**OpenHands empowers developers to build software faster and more efficiently by leveraging AI-driven agents.**  ([Original Repository](https://github.com/All-Hands-AI/OpenHands))

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de) |
[Español](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es) |
[Français](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr) |
[日本語](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja) |
[한국어](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko) |
[Português](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt) |
[Русский](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru) |
[中文](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh)

<hr>

OpenHands (formerly OpenDevin) offers a revolutionary platform for AI-powered software development, enabling agents that can perform complex coding tasks with ease.

## Key Features

*   **AI-Powered Agents:** Utilize AI agents that can perform tasks like code modification, running commands, web browsing, API calls, and more.
*   **CLI & GUI Support:** Run OpenHands using a user-friendly CLI or a graphical user interface.
*   **Docker Support:** Easily deploy and run OpenHands with Docker for simplified setup.
*   **Cloud Access:** Get started quickly with [OpenHands Cloud](https://app.all-hands.dev), with free credits for new users.
*   **Community-Driven:** Benefit from a vibrant and active community via Slack, Discord, and GitHub.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The easiest way to experience OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The recommended method to run OpenHands locally is via the CLI launcher with [uv](https://docs.astral.sh/uv/). This provides better isolation.

**Install uv** (if not already installed):

Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for platform-specific instructions.

**Launch OpenHands**:

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access OpenHands via [http://localhost:3000](http://localhost:3000) for GUI mode!

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

Run OpenHands with Docker:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.54
```

</details>

> **Note**: If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

> [!WARNING]
> For public networks, see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) to secure your deployment.

### Getting Started

Choose an LLM provider and add your API key after opening the application. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended. Explore [many options](https://docs.all-hands.dev/usage/llms).

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for requirements and further information.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user local use. It is not suitable for multi-tenant deployments.

If you want to run OpenHands in a multi-tenant environment, check out the source-available, commercially-licensed
[OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud)

You can:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact with it via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run OpenHands in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Run it on tagged issues with [a github action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for comprehensive setup instructions.

For source code modifications, refer to [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

For troubleshooting, consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation
  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Discover project details and tips in our [documentation](https://docs.all-hands.dev/usage/getting-started). It covers LLM providers, troubleshooting, and advanced configurations.

## 🤝 How to Join the Community

Join the OpenHands community and contribute:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Discord](https://discord.gg/ESHStjSjD4)
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues)

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for more.

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

OpenHands relies on the contributions of many and other open-source projects.

See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses used.

## 📚 Cite

```
@inproceedings{
  wang2025openhands,
  title={OpenHands: An Open Platform for {AI} Software Developers as Generalist Agents},
  author={Xingyao Wang and Boxuan Li and Yufan Song and Frank F. Xu and Xiangru Tang and Mingchen Zhuge and Jiayi Pan and Yueqi Song and Bowen Li and Jaskirat Singh and Hoang H. Tran and Fuqiang Li and Ren Ma and Mingzhang Zheng and Bill Qian and Yanjun Shao and Niklas Muennighoff and Yizhe Zhang and Binyuan Hui and Junyang Lin and Robert Brennan and Hao Peng and Heng Ji and Graham Neubig},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=OJd3ayDDoF}
}
```