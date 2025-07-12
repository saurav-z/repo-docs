<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Supercharge Your Software Development with AI</h1>
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

OpenHands (formerly OpenDevin) is an AI-powered platform designed to revolutionize software development, allowing you to build and innovate more efficiently.  **[Explore the OpenHands project on GitHub](https://github.com/All-Hands-AI/OpenHands) and unlock the future of coding!**

**Key Features:**

*   **AI-Driven Automation:** OpenHands agents can perform complex software development tasks with minimal human intervention.
*   **Code Modification and Generation:**  Modify, generate, and refactor code directly within the platform.
*   **Web Browsing & API Integration:** Seamlessly browse the web and interact with APIs to gather information and integrate with external services.
*   **StackOverflow Integration:** Leverage the power of StackOverflow by incorporating code snippets directly into your projects.
*   **Local and Cloud Deployment:** Run OpenHands locally using Docker or experience the easiest setup with OpenHands Cloud.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud: Get Started Easily

The fastest way to begin using OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

## 💻 Running OpenHands Locally with Docker

You can also run OpenHands on your local system using Docker.  Find detailed instructions and system requirements in the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

> [!WARNING]
> For deployments on public networks, consult the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) for enhanced security measures.

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.48
```

> **Note:**  If you used OpenHands before version 0.44, migrate your conversation history with `mv ~/.openhands-state ~/.openhands`.

Access OpenHands at [http://localhost:3000](http://localhost:3000) after launching the application. You will be prompted to select an LLM provider and enter an API key; [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many other options are available.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use. It is not suitable for multi-tenant environments.

Explore these alternative methods for running OpenHands:

*   [Connect to Your Local Filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless Mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting
*   [CLI Mode](https://docs.all-hands.dev/usage/how-to/cli-mode) for command-line interaction
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action) for issue-based workflows

Refer to [Running OpenHands](https://docs.all-hands.dev/usage/installation) for comprehensive information and setup guidance.

To modify the OpenHands source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

If you encounter any issues, consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

For project details and usage tips, explore our detailed [documentation](https://docs.all-hands.dev/usage/getting-started).

Find resources on:

*   LLM provider integration
*   Troubleshooting
*   Advanced configuration options

## 🤝 Join the OpenHands Community

OpenHands is a community-driven project. Join us on these platforms:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for discussions on research, architecture, and development.
*   [Discord](https://discord.gg/ESHStjSjD4) for general discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) to contribute ideas and track progress.

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Project Roadmap

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands thrives thanks to its contributors and the open-source projects it builds upon.

Refer to [CREDITS.md](./CREDITS.md) for a list of used open-source projects and licenses.

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