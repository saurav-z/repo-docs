<!-- Improved README.md -->
<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  </a>
  <h1>OpenHands: Revolutionize Software Development with AI</h1>
  <p><i>Code less, achieve more with the power of AI-driven software development agents.</i></p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/stargazers">
      <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
    </a>
    <a href="https://dub.sh/openhands">
      <img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community">
    </a>
    <a href="https://discord.gg/ESHStjSjD4">
      <img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md">
      <img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits">
    </a>
    <a href="https://docs.all-hands.dev/usage/getting-started">
      <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation">
    </a>
    <a href="https://arxiv.org/abs/2407.16741">
      <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv">
    </a>
    <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
      <img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score">
    </a>
  </p>

  <!-- Translation Links -->
  <p>
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>
  </p>
  <hr>
</div>

## About OpenHands

OpenHands (formerly OpenDevin) is a cutting-edge platform empowering software development through AI-driven agents, allowing developers to automate coding tasks and accelerate their workflow. These AI agents can perform a wide range of actions, much like a human developer: modify code, execute commands, browse the web, utilize APIs, and even extract code snippets from StackOverflow.

Explore the possibilities at [docs.all-hands.dev](https://docs.all-hands.dev) or get started today with [OpenHands Cloud](https://app.all-hands.dev).

> [!IMPORTANT]
> If you use OpenHands for your work, we invite you to join our Design Partner program.  Fill out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) to gain early access to commercial features and influence our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## Key Features

*   **AI-Powered Agents:** Utilize intelligent agents to automate and streamline software development tasks.
*   **Code Modification & Execution:** Effortlessly modify code, run commands, and interact with your projects.
*   **Web Browsing & API Integration:**  Leverage the web and APIs to enhance your development capabilities.
*   **Code Snippet Retrieval:** Quickly access and integrate code snippets from sources like StackOverflow.
*   **Local and Cloud Options:**  Use it locally for free or leverage the cloud version with $20 in free credits.
*   **Extensive Documentation**: Get help to install and use OpenHands.

## Getting Started

### ☁️ OpenHands Cloud

The easiest way to get started with OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

### 💻 Running OpenHands Locally

Choose from the two options below.

#### Option 1: CLI Launcher (Recommended)

Use the CLI launcher with [uv](https://docs.astral.sh/uv/) for the easiest local setup. This offers better isolation and is required for default MCP servers.

**Install uv** (if you haven't already):

Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for detailed instructions for your platform.

**Launch OpenHands**:
```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access OpenHands at [http://localhost:3000](http://localhost:3000) in GUI mode.

#### Option 2: Docker

<details>
<summary>Expand Docker Command</summary>

Run OpenHands directly with Docker:

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

> **Note**: If you used OpenHands before version 0.44, migrate your conversation history by running `mv ~/.openhands-state ~/.openhands`.

> [!WARNING]
> Ensure your deployment is secure by consulting the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) for network binding and additional security measures.

### Configuration

When the application opens, you'll need to select an LLM provider and add an API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works well. Check [many options](https://docs.all-hands.dev/usage/llms) for more LLM choices.

Consult the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and more information.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is intended for single-user, local workstation use.  Multi-tenant deployments are not supported, as there is no built-in authentication, isolation, or scalability.

Explore the following to customize your OpenHands experience:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact using a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Execute on tagged issues using a [Github Action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for detailed instructions.

If you're interested in modifying the source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Having trouble?  Consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

For project details, and tips, go to our [documentation](https://docs.all-hands.dev/usage/getting-started).

It contains valuable information on using various LLM providers, troubleshooting, and advanced configurations.

## 🤝 Join the Community

OpenHands is a community-driven project; contributions are highly valued. Get involved by joining our Slack or Discord or contribute on Github:

*   [Join our Slack workspace](https://dub.sh/openhands) - Discuss research, architecture, and future development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - For general discussion, questions, and feedback.
*   [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Check out the issues we're working on, or add your own ideas.

See details on community involvement in [COMMUNITY.md](./COMMUNITY.md) and contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1), updated monthly.

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License, excluding the `enterprise/` folder. Find more details in [`LICENSE`](./LICENSE).

## 🙏 Acknowledgements

OpenHands relies on a large number of contributors, whose contributions are very much appreciated. Additionally, we are grateful for the contributions of other open-source projects.

See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses used in OpenHands.

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