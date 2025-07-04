<!-- Improved README.md -->
<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Code Less, Make More with AI-Powered Software Development</h1>
  <p><i>Supercharge your software development workflow with OpenHands, an open-source AI platform.</i></p>
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="GitHub stars">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
  </a>
  <br/>
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA">
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

## Introduction

OpenHands is an open-source platform empowering software development with AI, allowing you to code less and accomplish more.  This innovative platform allows AI agents to perform tasks just like human developers.

Key features:

*   **AI-Powered Development:** Leverage AI agents to modify code, run commands, browse the web, and interact with APIs.
*   **Web Browsing & API Integration:**  Agents can now access the web for information and call APIs.
*   **Community-Driven:**  Benefit from a vibrant community and contribute to the project's development.
*   **Open Source:** OpenHands is available for anyone to use, modify, and distribute under the MIT License.
*   **Flexible Deployment:** Run OpenHands locally with Docker or utilize the OpenHands Cloud for easy access.
*   **Documentation & Support:** Comprehensive documentation and community support via Slack, Discord, and GitHub.

Learn more and get started on the [OpenHands GitHub repository](https://github.com/All-Hands-AI/OpenHands).

> [!IMPORTANT]
> Interested in using OpenHands for your work?  Join our Design Partner program by filling out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) for early access and input on the product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

Get started quickly with OpenHands on [OpenHands Cloud](https://app.all-hands.dev), which offers $20 in free credits for new users.

## 💻 Running OpenHands Locally

You can run OpenHands locally using Docker. This allows for customization and control over your environment.

**Key Steps:**

1.  **System Requirements:** Ensure your system meets the necessary requirements.
2.  **Docker Installation:** Install and configure Docker on your machine.
3.  **Deployment:** Pull and run the OpenHands Docker image.

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed instructions and system requirements.

> [!WARNING]
> Protect your local deployment by following our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) on a public network.

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands-dev/all-hands-ai/runtime:0.48-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands-dev/all-hands-ai/openhands:0.48
```

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

Access OpenHands at [http://localhost:3000](http://localhost:3000). You'll be prompted to select an LLM provider and input your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.  Explore [various LLM options](https://docs.all-hands.dev/usage/llms) to find the best fit for your needs.

## 💡 Other Ways to Run OpenHands

Explore these additional options to tailor OpenHands to your workflow:

*   **Filesystem Integration:**  Connect OpenHands to your local filesystem.
*   **Headless Mode:** Run OpenHands in a scriptable, headless mode.
*   **CLI Mode:** Interact with OpenHands using a user-friendly CLI.
*   **GitHub Action:** Automate tasks using a GitHub action.

For detailed instructions and setup information, consult the [Running OpenHands](https://docs.all-hands.dev/usage/installation) documentation.

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Comprehensive documentation is available at [docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started).  It includes guides on:

*   LLM provider selection.
*   Troubleshooting common issues.
*   Advanced configuration options.

## 🤝 Join the Community

OpenHands thrives on community contributions.  Engage with us through these channels:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - Discuss research, architecture, and development.
*   [Discord](https://discord.gg/ESHStjSjD4) - General discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Report issues or propose ideas.

Learn more about community involvement in [COMMUNITY.md](./COMMUNITY.md) and contributing guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

Track the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1), updated monthly at the maintainer's meeting.

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

OpenHands is a collaborative effort, and we are grateful to all contributors.  We also acknowledge the open-source projects that OpenHands builds upon.

Find the list of open-source projects and licenses in our [CREDITS.md](./CREDITS.md) file.

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