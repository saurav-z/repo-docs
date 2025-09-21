<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Supercharge Your Software Development with AI</h1>
</div>

<div align="center">
  <!-- Shields -->
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <!-- Community Links -->
  <a href="https://dub.sh/openhands"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <!-- Documentation and Research Links -->
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>
  <!-- Internationalization Links -->
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

OpenHands is an open-source platform, enabling AI agents to act as versatile software developers, empowering you to build software more efficiently.  Check out the [original repo](https://github.com/All-Hands-AI/OpenHands) for all the details.

**Key Features:**

*   **AI-Powered Development:** Leverage AI agents to modify code, run commands, and interact with the web.
*   **Versatile Capabilities:** OpenHands agents can copy code snippets, use APIs, and much more, mirroring human developer skills.
*   **Easy Cloud Access:** Quickly get started with OpenHands Cloud, offering free credits for new users.
*   **Flexible Deployment:** Run OpenHands locally using the CLI launcher or Docker.
*   **Community Driven:** Engage with a vibrant community through Slack, Discord, and GitHub.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud: Get Started Easily

The quickest way to experience OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), where new users receive $20 in free credits.

## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The CLI launcher, using [uv](https://docs.astral.sh/uv/), is the easiest way to run OpenHands locally, providing better isolation and required for OpenHands' default MCP servers.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands**:

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access OpenHands in GUI mode at [http://localhost:3000](http://localhost:3000)!

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

You can also run OpenHands directly with Docker:

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

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
> to secure your deployment by restricting network binding and implementing additional security measures.

### Getting Started

After launching, choose an LLM provider and add an API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works best.  Find more options at [https://docs.all-hands.dev/usage/llms](https://docs.all-hands.dev/usage/llms).

Refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and further details.

## 💡 Additional Ways to Run OpenHands

> [!WARNING]
> OpenHands is intended for single-user, local workstation use and is not appropriate for multi-tenant deployments without authentication, isolation, or scalability.
>
> Explore the source-available, commercially-licensed [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud) for multi-tenant environments.

You can also:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Run OpenHands in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Integrate OpenHands with a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action).

Find setup instructions and more information at [Running OpenHands](https://docs.all-hands.dev/usage/installation).

For source code modifications, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

If you experience any issues, consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

For comprehensive project information and usage tips, visit our [documentation](https://docs.all-hands.dev/usage/getting-started), which includes information on LLM providers, troubleshooting, and advanced configuration.

## 🤝 Join the OpenHands Community

Join our vibrant, community-driven project. Engage with us through:

*   [Slack](https://dub.sh/openhands) for research, architecture, and development discussions.
*   [Discord](https://discord.gg/ESHStjSjD4) for general discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) to contribute ideas and check out current projects.

Learn more about the community in [COMMUNITY.md](./COMMUNITY.md) and contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

Review the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Licensed under the MIT License, with the exception of the `enterprise/` folder. See [`LICENSE`](./LICENSE) for more details.

## 🙏 Acknowledgements

OpenHands thrives through the contributions of many individuals and relies on other open-source projects, as detailed in [CREDITS.md](./CREDITS.md).

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