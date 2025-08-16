<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands"><img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200"></a>
  <h1>OpenHands: The AI-Powered Platform for Smarter Software Development</h1>
  <p><em>Reduce code and increase output with OpenHands, your AI-powered development assistant.</em></p>
</div>

<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>

  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>
  <hr>
</div>

## What is OpenHands?

OpenHands is a cutting-edge AI platform designed to revolutionize software development. Empowered by AI agents, OpenHands streamlines coding tasks, allowing developers to build more, faster, with less code.  Find the original repository on [GitHub](https://github.com/All-Hands-AI/OpenHands).

## Key Features

*   **AI-Powered Agents:** OpenHands utilizes intelligent agents capable of performing a wide range of development tasks.
*   **Code Modification:** Seamlessly modify existing code with AI assistance.
*   **Command Execution:** Execute commands directly within the development environment.
*   **Web Browsing & API Integration:** Access web resources and integrate with APIs to enhance functionality.
*   **Code Snippet Integration:** Leverage code snippets from StackOverflow and other resources.

## Get Started

### OpenHands Cloud

The simplest way to experience OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

### Running OpenHands Locally

Choose from the following options to run OpenHands on your local machine:

#### Option 1: CLI Launcher (Recommended)

The CLI launcher, using [uv](https://docs.astral.sh/uv/), is the recommended method for local operation.

**Install uv (if you haven't already):**  Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for instructions.

**Launch OpenHands:**

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access OpenHands in GUI mode at [http://localhost:3000](http://localhost:3000).

#### Option 2: Docker

<details>
<summary>Expand Docker Command</summary>

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.53
```

</details>

> **Note:** If using OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

> [!WARNING]
> Secure your deployment on a public network by consulting our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

#### Getting Started

Select an LLM provider and enter your API key upon opening the application. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many [other options](https://docs.all-hands.dev/usage/llms) are available.  Visit the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for more details.

## Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use and is not suitable for multi-tenant environments without authentication, isolation, or scalability features.  Consider the source-available, commercially-licensed [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud) for multi-tenant deployments.

Explore these additional options:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Run in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Use a [GitHub action](https://docs.all-hands.dev/usage/how-to/github-action).

Refer to [Running OpenHands](https://docs.all-hands.dev/usage/installation) for further instructions.

Modify the OpenHands source by consulting [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md). For troubleshooting, see the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## Documentation

<a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Access detailed documentation at [docs.all-hands.dev/usage/getting-started] to learn about LLM providers, troubleshoot issues, and configure advanced features.

## Join the Community

Join the OpenHands community and contribute to its growth! We primarily communicate through Slack:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA): Discuss research, architecture, and development plans.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4): Participate in general discussions and provide feedback.
*   [View or submit GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues): Share your ideas and track ongoing issues.

Additional community information is available in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License. For details, see [`LICENSE`](./LICENSE).

## Acknowledgements

OpenHands is a community effort, and we're grateful for every contribution. We also appreciate the open-source projects upon which OpenHands is built.

See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses.

## Cite

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