# OpenHands: Revolutionize Software Development with AI (Formerly OpenDevin)

**OpenHands empowers AI agents to act as developers, enabling you to write less code and achieve more.**  [Check out the original repository](https://github.com/All-Hands-AI/OpenHands).

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Arxiv Paper](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)
[DE](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de) | [ES](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es) | [FR](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr) | [JA](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja) | [KO](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko) | [PT](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt) | [RU](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru) | [ZH](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh)

<hr>

OpenHands, formerly known as OpenDevin, offers a cutting-edge platform for AI-powered software development agents.  These agents mimic human developers, automating tasks and accelerating your workflow.

## Key Features:

*   **AI-Powered Development:** Agents can modify code, execute commands, browse the web, and utilize APIs.
*   **Code Integration:**  Seamlessly incorporates code snippets from sources like Stack Overflow.
*   **OpenHands Cloud:**  Get started quickly with the cloud-based platform, including free credits for new users.
*   **Local Deployment Options:** Run OpenHands locally using the CLI launcher (recommended) or Docker.
*   **Extensive Documentation:** Comprehensive resources for setup, usage, troubleshooting, and advanced configurations.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The easiest way to experience OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), offering $20 in free credits for new users.

## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The CLI launcher, using [uv](https://docs.astral.sh/uv/), provides better isolation and is required for default MCP servers.

**Install uv** (if not already installed):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for instructions.

**Launch OpenHands**:

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access OpenHands at [http://localhost:3000](http://localhost:3000) (GUI mode).

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

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

> **Note:** Migrate your conversation history if you used OpenHands before version 0.44: `mv ~/.openhands-state ~/.openhands`.

> [!WARNING]
> Secure your Docker deployment on public networks with the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Getting Started

Choose an LLM provider and add your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.  Explore [multiple LLM options](https://docs.all-hands.dev/usage/llms).

Consult the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and further information.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user local workstations, not multi-tenant deployments. It lacks built-in authentication, isolation, or scalability.

If you need multi-tenant capabilities, explore the source-available, commercially-licensed [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud).

Additional options:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact with the [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run OpenHands in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Use it as a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for detailed instructions.

## 📖 Documentation

<a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Comprehensive documentation on using OpenHands can be found at [documentation](https://docs.all-hands.dev/usage/getting-started). Find information on LLM providers, troubleshooting, and advanced configuration.

## 🤝 Join the Community

OpenHands thrives on community involvement. Connect with us on Slack, Discord, or GitHub:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA): Discuss research, architecture, and development.
*   [Discord](https://discord.gg/ESHStjSjD4):  General discussions, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues): Review ongoing issues and contribute ideas.

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for community details and contribution guidelines.

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License.  See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is a collaborative project, and we extend our gratitude to all contributors and the open-source projects we build upon.

For a list of open-source projects and their licenses, see [CREDITS.md](./CREDITS.md).

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