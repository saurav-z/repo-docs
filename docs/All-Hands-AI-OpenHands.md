# OpenHands: AI-Powered Software Development Agent (Code Less, Make More)

[<img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=social" alt="Stars">](https://github.com/All-Hands-AI/OpenHands)
[View on GitHub](https://github.com/All-Hands-AI/OpenHands)

**OpenHands empowers developers to build software faster by leveraging AI agents that can write, run, and debug code.**

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

![App screenshot](./docs/static/img/screenshot.png)

## Key Features

*   **AI-Powered Agents:**  Automate software development tasks with intelligent agents.
*   **Code Modification & Execution:**  Modify code, run commands, and debug errors seamlessly.
*   **Web Browsing and API Interaction:**  Agents can browse the web and call APIs to gather information and complete tasks.
*   **Integration with StackOverflow:** Leverage code snippets directly from StackOverflow to accelerate development.
*   **Flexible Deployment:** Run OpenHands locally, on the cloud, or integrated into your existing workflows.

## Get Started with OpenHands

### ☁️ OpenHands Cloud

The simplest way to start using OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

### 💻 Running OpenHands Locally

Choose from the following options to run OpenHands on your local machine:

#### Option 1: CLI Launcher (Recommended)

The CLI launcher (using `uv`) is the recommended method for local setup, providing better isolation and compatibility with OpenHands' MCP servers.

**Prerequisites:**
Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

**Launch OpenHands:**

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access the GUI via your browser at [http://localhost:3000](http://localhost:3000).

#### Option 2: Docker

<details>
<summary>Expand Docker Instructions</summary>

Run OpenHands with Docker:

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
    docker.all-hands-dev/all-hands-ai/openhands:0.56
```

</details>

> **Note:**  If you're migrating from a version prior to 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

> [!WARNING]
> For deployments on public networks, see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) to secure your setup.

#### Initial Configuration

After launching the application, you will need to choose an LLM provider and enter your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is the recommended option, but many other [LLM providers](https://docs.all-hands.dev/usage/llms) are supported.

Visit the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and detailed setup instructions.

## 💡 Additional Ways to Run OpenHands

> [!WARNING]
> OpenHands is intended for single-user, local workstation use and isn't suitable for multi-tenant environments without additional security and scalability considerations.
>
> Consider the source-available [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud) for multi-tenant deployments.

You can also:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact with the [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Use the [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting
*   Integrate with [GitHub Actions](https://docs.all-hands.dev/usage/how-to/github-action)

Find more information at [Running OpenHands](https://docs.all-hands.dev/usage/installation).

For contributing, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Troubleshooting help is available in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

Explore our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started) to learn more:

*   LLM provider setup
*   Troubleshooting resources
*   Advanced configuration options

## 🤝 Join the OpenHands Community

OpenHands thrives on community contributions!  Connect with us:

*   [Slack](https://dub.sh/openhands): Discuss research, architecture, and development.
*   [Discord](https://discord.gg/ESHStjSjD4): General discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues):  Contribute ideas and track progress.

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## 📈 Project Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is licensed under the [MIT License](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE), excluding the `enterprise/` folder.

## 🙏 Acknowledgements

We deeply appreciate the contributions of the OpenHands community and the open-source projects we build upon.  See [CREDITS.md](./CREDITS.md) for a list of dependencies and licenses.

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