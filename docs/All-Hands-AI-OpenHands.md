<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Empowering AI Software Development</h1>
  <p><b>Unlock the future of coding with OpenHands, an AI-powered platform designed to accelerate your development workflow.</b></p>
</div>

<div align="center">
  <!-- Badges -->
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://dub.sh/openhands"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
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

## Key Features

*   **AI-Powered Agents:** OpenHands leverages AI agents that can perform complex software development tasks.
*   **Code Modification & Execution:** Modify, run and debug code efficiently with AI assistance.
*   **Web Browsing & API Integration:** Access web resources, call APIs, and integrate external services.
*   **Code Snippet Integration:** Utilize code snippets from platforms like StackOverflow to expedite development.
*   **Cloud & Local Deployment Options:** Use OpenHands through the cloud or install it locally for more customization.

## Getting Started

OpenHands (formerly OpenDevin) is a platform designed to revolutionize software development, powered by AI agents capable of performing tasks a human developer can.

Get started with OpenHands at [docs.all-hands.dev](https://docs.all-hands.dev) and explore the possibilities.

> [!IMPORTANT]
> For early access to commercial features and to help shape our product roadmap, consider joining our Design Partner program by filling out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform).

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The easiest way to try OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

## 💻 Running OpenHands Locally

Choose from a few different methods to run OpenHands locally.

### Option 1: CLI Launcher (Recommended)

For the best local experience, use the CLI launcher with [uv](https://docs.astral.sh/uv/). This is also required for the default MCP servers.

**Install uv** (if not already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands**:
```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access OpenHands at [http://localhost:3000](http://localhost:3000) when in GUI mode.

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
    docker.all-hands-dev/all-hands-ai/openhands:0.57
```

</details>

> **Note**: If you used OpenHands before version 0.44, consider migrating your conversation history by running `mv ~/.openhands-state ~/.openhands`.

> [!WARNING]
> If you're on a public network, secure your deployment by reviewing the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Configuration

Upon launching the application, you'll be prompted to select an LLM provider and enter an API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many other options are available, found on the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for
system requirements and more information.

## 💡 Other ways to run OpenHands

> [!WARNING]
> OpenHands is designed for individual use on a local workstation and is not suited for multi-tenant environments.

You can connect OpenHands to your local filesystem, use the [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode), run in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode), or integrate it with [Github Actions](https://docs.all-hands.dev/usage/how-to/github-action).

More information and setup instructions are available in [Running OpenHands](https://docs.all-hands.dev/usage/installation).

For information on modifying the source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

If you have problems, consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

Explore the project further with the detailed [documentation](https://docs.all-hands.dev/usage/getting-started) for usage tips, LLM provider options, troubleshooting advice, and advanced configuration settings.

## 🤝 How to Join the Community

We welcome contributions! Connect with us on Slack, Discord, or GitHub:

-   [Join our Slack workspace](https://dub.sh/openhands) for discussions on research, architecture, and development.
-   [Join our Discord server](https://discord.gg/ESHStjSjD4) for community discussions, Q&A, and feedback.
-   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) for ideas, issues and contribution.

Learn more about the community in [COMMUNITY.md](./COMMUNITY.md) or details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

View the OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is licensed under the MIT License, excluding the `enterprise/` folder. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is built with the help of many contributors and relies on other open-source projects.

See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses.

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