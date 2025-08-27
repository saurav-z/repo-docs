<!-- Improved README.md -->
<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  </a>
  <h1>OpenHands: Revolutionizing Software Development with AI</h1>
</div>

<div align="center">
  <!-- Badges -->
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>
  <br/>
  <!-- Translations -->
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

OpenHands is a groundbreaking platform that empowers AI agents to handle software development tasks, reducing the need for manual coding.

## Key Features:

*   **AI-Powered Automation:** Automates code modification, command execution, web browsing, and API calls.
*   **Intelligent Code Snippet Retrieval:**  Leverages resources like Stack Overflow for efficient coding.
*   **Flexible Deployment:**  Offers options for cloud, local CLI, Docker, and headless modes.
*   **Community-Driven:** Active Slack and Discord communities for support, discussion, and collaboration.
*   **Comprehensive Documentation:**  Detailed guides and resources to get you started.

## Getting Started

### OpenHands Cloud

The easiest way to start is on [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

### Running OpenHands Locally

Choose your preferred method for local setup:

#### Option 1: CLI Launcher (Recommended)

1.  **Install uv:** (Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)).
2.  **Launch OpenHands:**

```bash
uvx --python 3.12 --from openhands-ai openhands serve
# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access the GUI at [http://localhost:3000](http://localhost:3000).

#### Option 2: Docker

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.54
```

**Important Notes:**
*  If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate history.
*   For secure Docker deployment, see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

#### Configuration

1.  Open the application and select an LLM provider.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.
2.  Add your API key.
3.  Refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for requirements and detailed instructions.

## Other Ways to Run OpenHands
*   Connect to your local filesystem:  [Filesystem Guide](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   CLI Mode: [CLI Mode Guide](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Headless Mode: [Headless Mode Guide](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Github Action: [Github Action Guide](https://docs.all-hands.dev/usage/how-to/github-action)
**Important:** Avoid multi-tenant deployments without built-in security measures.

## Documentation

Explore the [OpenHands documentation](https://docs.all-hands.dev/usage/getting-started) for comprehensive guides, troubleshooting tips, and advanced configuration options.
  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

## Join the OpenHands Community

Connect with the OpenHands community for support and collaboration:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   [Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

We are deeply grateful to all contributors and the open-source projects we build upon. See [CREDITS.md](./CREDITS.md) for details.

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

<!-- Link Back to Original Repo -->
<p align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands">Go to the OpenHands GitHub Repository</a>
</p>