<!-- Improved README.md for OpenHands -->
<a name="readme-top"></a>

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Your AI Copilot for Software Development</h1>
  <p><em>Supercharge your coding workflow and build more with OpenHands, an open-source platform for AI-powered software development.</em></p>

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
    <br/>
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

<!-- Summary Section -->
## About OpenHands

OpenHands is a cutting-edge platform designed to empower developers with the capabilities of AI. It allows AI agents to modify code, execute commands, browse the web, call APIs, and even utilize resources like StackOverflow, accelerating your development process.  [Explore the OpenHands GitHub Repository](https://github.com/All-Hands-AI/OpenHands).

<!-- Key Features Section -->
## Key Features

*   **AI-Powered Code Modification:** Automate code changes and refactoring tasks.
*   **Web Browsing & API Integration:**  Enable agents to gather information and interact with online services.
*   **Command Execution:** Execute terminal commands directly from the agent.
*   **Community-Driven:** Benefit from a vibrant and supportive open-source community.
*   **Flexible Deployment:** Run OpenHands locally or in the cloud.

<!-- Design Partner Program -->
> [!IMPORTANT]
>  Are you using OpenHands for your work?  We'd love to hear from you! Join our Design Partner program by filling out
>  [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
>  to get early access to commercial features and help shape our product roadmap.

<!-- Screenshot (Optional) -->
![App screenshot](./docs/static/img/screenshot.png)

<!-- Cloud Section -->
## ☁️ OpenHands Cloud

Get started quickly with OpenHands Cloud!  New users receive $20 in free credits.  Visit the [OpenHands Cloud](https://app.all-hands.dev) to learn more.

<!-- Local Installation Section -->
## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The easiest way to run OpenHands locally is using the CLI launcher with [uv](https://docs.astral.sh/uv/). This provides better isolation from your current project's virtual environment and is required for OpenHands' default MCP servers.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands**:
```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

You'll find OpenHands running at [http://localhost:3000](http://localhost:3000) (for GUI mode)!

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

When you open the application, you'll be asked to choose an LLM provider and add an API key.
[Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`)
works best, but you have [many options](https://docs.all-hands.dev/usage/llms).

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for
system requirements and more information.

<!-- Alternative Run Options -->
## 💡 Other ways to run OpenHands

> [!WARNING]
> OpenHands is meant to be run by a single user on their local workstation.
> It is not appropriate for multi-tenant deployments where multiple users share the same instance. There is no built-in authentication, isolation, or scalability.
>
> If you're interested in running OpenHands in a multi-tenant environment, check out the source-available, commercially-licensed
> [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud)

Explore more ways to run OpenHands:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Run in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Utilize a [GitHub action](https://docs.all-hands.dev/usage/how-to/github-action) for tagged issues.

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for setup instructions.

<!-- Development Section -->
If you want to contribute to OpenHands, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

<!-- Troubleshooting Section -->
Having issues? Consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) for assistance.

<!-- Documentation Section -->
## 📖 Documentation

Find comprehensive information, usage tips, and advanced configuration options in our [documentation](https://docs.all-hands.dev/usage/getting-started).

<!-- Community Section -->
## 🤝 Join the OpenHands Community

OpenHands thrives on community contributions! Connect with us on:

*   [Slack](https://dub.sh/openhands): Discuss research, architecture, and development.
*   [Discord](https://discord.gg/ESHStjSjD4): General discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues): Share ideas and track progress.

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

<!-- Progress Section -->
## 📈 Project Progress

Stay updated with the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1), updated at the end of each month.

<!-- Star History Chart -->
<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

<!-- License Section -->
## 📜 License

OpenHands is distributed under the [MIT License](./LICENSE).

<!-- Acknowledgements Section -->
## 🙏 Acknowledgements

OpenHands is a community effort, and we appreciate every contribution! We also rely on the work of other open-source projects.  See [CREDITS.md](./CREDITS.md) for a list of all project credits.

<!-- Citation Section -->
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