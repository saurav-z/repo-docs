<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1 align="center">OpenHands: Unleash AI-Powered Software Development</h1>
</div>

<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stars">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
  </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
  </a>
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

OpenHands empowers developers with AI-driven agents that can write and debug code, automate tasks, and supercharge productivity.  [Explore the OpenHands Repo](https://github.com/All-Hands-AI/OpenHands)

## Key Features

*   **AI-Powered Development:** Leverage intelligent agents to automate coding tasks, debug, and more.
*   **Code Modification & Execution:** Modify code, run commands, and interact with various tools.
*   **Web Browsing & API Integration:** Agents can browse the web and call APIs to gather information and enhance development.
*   **Stack Overflow Integration:**  Easily incorporate code snippets from Stack Overflow to accelerate the development process.
*   **Cloud & Local Deployment:** Start quickly with OpenHands Cloud or run locally using CLI or Docker.
*   **Community-Driven:** Join an active community and contribute to the project's evolution.

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started

### ☁️ OpenHands Cloud

The simplest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

### 💻 Running OpenHands Locally

Choose from the following options:

#### Option 1: CLI Launcher (Recommended)

Use the CLI launcher with [uv](https://docs.astral.sh/uv/) for the best local experience.

**Install `uv`:** (if you haven't already)
See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands:**
```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access OpenHands via [http://localhost:3000](http://localhost:3000) in GUI mode.

#### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

Run OpenHands with Docker:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.55-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.55-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.55
```
</details>

> **Note:** Migrate your conversation history if upgrading from versions before 0.44: `mv ~/.openhands-state ~/.openhands`

> [!WARNING]
> For public networks, harden your Docker deployment using the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

#### Configuration

After launching, you'll be prompted to select an LLM provider and provide an API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but you have many options [here](https://docs.all-hands.dev/usage/llms).

Refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed system requirements and additional information.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use. Multi-tenant deployments are not recommended.  Consider the [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud) for multi-tenant environments.

Explore these options:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact through the [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Utilize the [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

See [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more details and setup instructions.

For source code modifications, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

If you encounter issues, consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

Dive deeper into the project and learn usage tips via the comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started), covering LLM providers, troubleshooting, and advanced configuration.

## 🤝 Join the OpenHands Community

OpenHands thrives on community contributions. Connect with us via:

*   [Slack](https://dub.sh/openhands) - For research, architecture, and development discussions.
*   [Discord](https://discord.gg/ESHStjSjD4) - For general discussions, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Explore open issues and contribute your ideas.

Find more about the community in [COMMUNITY.md](./COMMUNITY.md) and contribution guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress & Roadmap

View the OpenHands monthly roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Licensed under the MIT License, except for the `enterprise/` folder. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is built by a vibrant community, and we're deeply grateful to all contributors and the open-source projects that we build upon.

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