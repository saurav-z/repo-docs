<!-- Improved README with SEO Optimization -->

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: AI-Powered Software Development, Simplified</h1>
  <p><em>Automate software development tasks with an AI agent that understands code and executes your commands.</em></p>
  <a href="https://github.com/All-Hands-AI/OpenHands">
      <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stars">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="License">
  </a>
</div>

<div align="center">
  <a href="https://dub.sh/openhands">
      <img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community">
  </a>
  <a href="https://discord.gg/ESHStjSjD4">
      <img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md">
      <img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits">
  </a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started">
      <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation">
  </a>
  <a href="https://arxiv.org/abs/2407.16741">
      <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv">
  </a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
      <img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score">
  </a>
  <br/>

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

**OpenHands (formerly OpenDevin) empowers developers with AI agents capable of automating complex software development tasks, reducing coding time, and boosting productivity.**

*   **AI-Powered Code Modification:**  Modify code, run commands, and integrate with web APIs.
*   **Web Browsing & Information Retrieval:**  Access and utilize information from the internet.
*   **Seamless Integration:**  Copy code snippets from resources like StackOverflow.
*   **Flexible Deployment:** Run locally via CLI launcher or Docker, or use OpenHands Cloud.
*   **Community Driven:**  Join our Slack, Discord, or contribute via GitHub to shape the future of AI-assisted development.

[Get started with OpenHands](https://docs.all-hands.dev) or sign up for [OpenHands Cloud](https://app.all-hands.dev) to experience the future of software development.

> [!IMPORTANT]
> Interested in using OpenHands for work? Join our Design Partner program by filling out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> and get early access to commercial features.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

## 💻 Running OpenHands Locally

Choose from the following methods to run OpenHands on your local machine:

### Option 1: CLI Launcher (Recommended)

Use the CLI launcher with [uv](https://docs.astral.sh/uv/) for better isolation and default MCP server support.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for platform-specific instructions.

**Launch OpenHands**:

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access OpenHands in GUI mode at [http://localhost:3000](http://localhost:3000).

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

Run OpenHands with Docker using the following command:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.56-nikolaik

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

> **Note**:  If you are upgrading from a pre-0.44 version, migrate your conversation history by running `mv ~/.openhands-state ~/.openhands`.

> [!WARNING]
> Enhance security on public networks by consulting our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Getting Started

Upon opening the application, you'll be prompted to select an LLM provider and input an API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but other [LLM providers](https://docs.all-hands.dev/usage/llms) are also available.

Visit the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed system requirements and setup instructions.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use and isn't suitable for multi-tenant deployments due to a lack of built-in authentication, isolation, or scalability.

Explore alternative deployment options:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Run OpenHands in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Integrate with a GitHub Action using [a github action](https://docs.all-hands.dev/usage/how-to/github-action).

Refer to [Running OpenHands](https://docs.all-hands.dev/usage/installation) for comprehensive setup instructions.

If you want to contribute to the source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Encountering issues?  Find solutions in our [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

Comprehensive documentation, usage tips, LLM provider information, and advanced configuration options are available in our [documentation](https://docs.all-hands.dev/usage/getting-started).

## 🤝 How to Join the Community

Become a part of the OpenHands community!  We welcome contributions and engage primarily through Slack, Discord, and GitHub:

*   [Join our Slack workspace](https://dub.sh/openhands) - Discuss research, architecture, and future development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - General discussion, questions, and feedback.
*   [Read or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Contribute ideas and track progress.

Learn more about the community in [COMMUNITY.md](./COMMUNITY.md) and contributing guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is licensed under the MIT License, excluding the `enterprise/` folder.  See [`LICENSE`](./LICENSE) for complete details.

## 🙏 Acknowledgements

OpenHands is a community project, and we appreciate every contribution. We are grateful for the open-source projects that we build upon.

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