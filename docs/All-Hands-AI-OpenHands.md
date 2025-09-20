# OpenHands: AI-Powered Software Development Agent

**OpenHands empowers you to build software faster by leveraging the power of AI to automate development tasks.**  [Explore the OpenHands project on GitHub](https://github.com/All-Hands-AI/OpenHands).

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://dub.sh/openhands)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

---

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de) |
[Español](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es) |
[français](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr) |
[日本語](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja) |
[한국어](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko) |
[Português](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt) |
[Русский](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru) |
[中文](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh)

---

## Key Features

*   **AI-Powered Automation:** Automate software development tasks, from code modification to API calls.
*   **Web Browsing & API Integration:**  OpenHands agents can browse the web and interact with APIs to gather information and complete tasks.
*   **Code Snippet Retrieval:** Access and utilize code snippets from resources like Stack Overflow.
*   **Flexible Deployment:**  Run OpenHands locally, in the cloud (OpenHands Cloud), or integrate it into your existing workflows.
*   **Community-Driven:** Benefit from a vibrant and active community, and contribute to the project's growth.

## Getting Started

OpenHands empowers AI agents to act as developers, able to modify code, run commands, and more.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev) or [sign up for OpenHands Cloud](https://app.all-hands.dev).

> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The easiest way to start using OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), with $20 in free credits for new users.

## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The most straightforward method to run OpenHands locally is with the CLI launcher and [uv](https://docs.astral.sh/uv/).  This provides superior isolation from your project's virtual environment and is essential for OpenHands' default MCP servers.

**Install uv** (if you haven't already):

Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the most up-to-date installation instructions for your platform.

**Launch OpenHands**:

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Find OpenHands running at [http://localhost:3000](http://localhost:3000) (for GUI mode)!

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

Alternatively, run OpenHands with Docker:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.57
```

</details>

> **Note**: If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

> [!WARNING]
> On a public network? Secure your deployment by following the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Configuration

When you launch the application, you'll be prompted to choose an LLM provider and add an API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but multiple [LLM options](https://docs.all-hands.dev/usage/llms) are available.

Consult the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for comprehensive details and system requirements.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user local workstation environments.  It is unsuitable for multi-tenant deployments lacking built-in authentication, isolation, or scalability.
>
> For multi-tenant environments, consider the source-available, commercially-licensed [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud).

You can connect OpenHands to your local filesystem ([connecting to your filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)), interact with it via a friendly CLI ([CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode)), run OpenHands in a scriptable headless mode ([headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)), or on tagged issues with a GitHub action ([GitHub action](https://docs.all-hands.dev/usage/how-to/github-action)).

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more information and instructions.

For source code modifications, refer to [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Experiencing issues? The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## 📖 Documentation

For in-depth information, usage tips, and advanced configurations, explore our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started).

## 🤝 Join the Community

OpenHands thrives on community involvement! We primarily communicate on Slack; you can also connect with us on Discord and GitHub:

*   [Join our Slack workspace](https://dub.sh/openhands) - Discuss research, architecture, and future development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - Engage in general discussions, ask questions, and provide feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Review existing issues or submit your ideas.

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the maintainer's meeting at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License, except the `enterprise/` folder. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is built by many contributors, and we deeply appreciate their contributions! We're also grateful to other open-source projects.

For a list of open-source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

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