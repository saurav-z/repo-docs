<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Supercharge Your Development with AI</h1>
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

## About OpenHands

OpenHands (formerly OpenDevin) is an open-source platform that empowers AI-driven software development, allowing you to build faster and more efficiently.  **This innovative platform allows AI agents to perform complex development tasks just like a human developer.**  [Learn more at our GitHub repository](https://github.com/All-Hands-AI/OpenHands).

## Key Features

*   **AI-Powered Agents:** Utilize intelligent agents that can modify code, execute commands, browse the web, and more.
*   **Code Modification & Execution:** Automate code changes, run commands, and integrate with APIs.
*   **Web Browsing & Information Retrieval:** Access and utilize information from the web to enhance your development process.
*   **Open Source & Community Driven:** Benefit from a collaborative community and the flexibility of open-source development.
*   **StackOverflow Integration:** Seamlessly integrates with StackOverflow for efficient code snippet retrieval.

## Getting Started

### OpenHands Cloud

The easiest way to experience OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), offering new users \$20 in free credits.

### Running OpenHands Locally

Choose from the CLI launcher (recommended) or Docker for local execution.

#### 1. CLI Launcher (Recommended)

1.  **Install uv:** (If you haven't already, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))
2.  **Launch OpenHands:**
    ```bash
    # Launch the GUI server
    uvx --python 3.12 --from openhands-ai openhands serve

    # Or launch the CLI
    uvx --python 3.12 --from openhands-ai openhands
    ```
    Access OpenHands at [http://localhost:3000](http://localhost:3000) in GUI mode!

#### 2. Docker

<details>
<summary>Click to expand Docker command</summary>

Run OpenHands with Docker:

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

> **Note**: If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your history.

> [!WARNING]
> On a public network? Secure your deployment with the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

#### Configuration

When running the application, select an LLM provider and input your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but other [options](https://docs.all-hands.dev/usage/llms) are available.

Find more system requirements and installation details in the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

## Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is for single-user, local workstation use. It is *not* designed for multi-tenant deployments.

Explore ways to run OpenHands:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Use [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting.
*   Run on tagged issues using a [GitHub action](https://docs.all-hands.dev/usage/how-to/github-action).

Get setup instructions at [Running OpenHands](https://docs.all-hands.dev/usage/installation).

For source code modifications, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

For troubleshooting help, see the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Access comprehensive documentation to learn more about the project: [documentation](https://docs.all-hands.dev/usage/getting-started). This includes information on:

*   Using various LLM providers
*   Troubleshooting resources
*   Advanced configuration options

## Join the Community

OpenHands is a community-driven project.  Join us to:

*   [Slack Workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - For research, architecture, and development discussions.
*   [Discord Server](https://discord.gg/ESHStjSjD4) - General discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Share ideas and contribute to our ongoing work.

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Progress

See the OpenHands monthly roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is built by a large community and relies on other open-source projects.
For a list of these projects, see [CREDITS.md](./CREDITS.md).

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