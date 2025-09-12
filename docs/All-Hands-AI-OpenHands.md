<!-- Improved & SEO-Optimized README for OpenHands -->

<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  </a>
  <br>
  <h1>OpenHands: Build Software Faster with AI</h1>
  <p><b>Unlock the power of AI for software development and accelerate your coding workflow with OpenHands.</b></p>
</div>

<div align="center">
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

*   **AI-Powered Agents:** OpenHands utilizes AI agents that can perform complex software development tasks.
*   **Code Modification & Execution:** Agents can modify code, run commands, and interact with your development environment.
*   **Web Browsing & API Integration:** Access the internet and call APIs to gather information and automate tasks.
*   **Stack Overflow Integration:**  Leverage the power of Stack Overflow by having agents find and utilize code snippets.
*   **Open Source and Customizable:** Build and extend OpenHands to meet your specific needs.

## Getting Started with OpenHands

OpenHands is a platform for AI-powered software development agents, offering a powerful toolkit to streamline your coding workflow.  Get started today with the options below.

### ☁️ OpenHands Cloud

The easiest way to begin is on [OpenHands Cloud](https://app.all-hands.dev), with $20 in free credits for new users.

### 💻 Running OpenHands Locally

Choose your preferred method to install and run OpenHands on your local machine:

#### Option 1: CLI Launcher (Recommended)

The CLI launcher provides isolation and is required for default MCP servers.

1.  **Install uv:** Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your platform.
2.  **Launch OpenHands:**

    ```bash
    # Launch the GUI server
    uvx --python 3.12 --from openhands-ai openhands serve

    # Or launch the CLI
    uvx --python 3.12 --from openhands-ai openhands
    ```
    Access OpenHands via the GUI at [http://localhost:3000](http://localhost:3000).

#### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

1.  **Pull the Docker image:**
    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.56-nikolaik
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -it --rm --pull=always \
        -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.56-nikolaik \
        -e LOG_ALL_EVENTS=true \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v ~/.openhands:/.openhands \
        -p 3000:3000 \
        --add-host host.docker.internal:host-gateway \
        --name openhands-app \
        docker.all-hands.dev/all-hands-ai/openhands:0.56
    ```
</details>

> **Note:** Migrate your history if needed: `mv ~/.openhands-state ~/.openhands`

> [!WARNING]
> See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) to secure your deployment on public networks.

### Configuration

1.  **LLM Provider:**  Choose an LLM provider and add your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.  Explore [other options](https://docs.all-hands.dev/usage/llms).
2.  **System Requirements:** Review the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and setup instructions.

## 💡 Additional Ways to Run

> [!WARNING]
> OpenHands is intended for single-user, local workstation use. Avoid multi-tenant deployments.

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Headless Mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for more details.

## 🛠️ Development

Find information about contributing to the source code in [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## 🐛 Troubleshooting

If you encounter issues, the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## 📚 Documentation

Explore our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started) to learn more.  You'll find:

*   LLM provider information
*   Troubleshooting resources
*   Advanced configuration options

## 🤝 Community

Join the OpenHands community and contribute!

*   [Join our Slack workspace](https://dub.sh/openhands) - Discuss research, architecture, and development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - For general discussion, questions, and feedback.
*   [View GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Track issues and contribute ideas.

Learn more about the community in [COMMUNITY.md](./COMMUNITY.md) or contribute through [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Project Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License (except for the `enterprise/` folder). See [`LICENSE`](./LICENSE).

## 🙏 Acknowledgements

Thanks to all contributors and the open-source projects used in OpenHands!  See [CREDITS.md](./CREDITS.md) for a list.

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