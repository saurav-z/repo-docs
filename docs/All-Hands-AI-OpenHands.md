<!-- Improved README.md -->
<a name="readme-top"></a>

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1 align="center">OpenHands: AI-Powered Software Development Platform</h1>
  <p align="center">
    <i>Code less, create more with OpenHands, a platform that lets AI build software for you.</i>
  </p>

  <!-- Badges -->
  <p align="center">
    <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/stargazers">
      <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
    </a>
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
  </p>
  <hr>
</div>

## About OpenHands

OpenHands (formerly OpenDevin) is a cutting-edge platform designed to empower software developers with the capabilities of AI-driven agents. These agents can perform a wide range of tasks, simulating the actions of a human developer, from code modification to web browsing, API calls, and even code snippet integration from sources like StackOverflow.  **Explore the future of software development with OpenHands, where AI agents collaborate with you to build amazing things.**  Get started today by visiting the [OpenHands GitHub repository](https://github.com/All-Hands-AI/OpenHands).

### Key Features

*   **AI-Powered Agents:** Utilize intelligent agents to automate and accelerate software development tasks.
*   **Code Modification:**  Easily modify and update code with AI assistance.
*   **Web Browsing:** Allow agents to browse the web for information and resources.
*   **API Integration:** Seamlessly integrate with APIs to enhance application functionality.
*   **Code Snippet Integration:**  Leverage AI to incorporate code snippets from popular sources.
*   **OpenHands Cloud:** Get started easily with our cloud platform, plus a generous $20 in free credits.
*   **Local Deployment:**  Run OpenHands on your local system using Docker for complete control.
*   **Multi-language Support:**  Access documentation and resources in multiple languages.
*   **Community Focused:** Collaborate with a thriving community through Slack, Discord, and GitHub.

### Get Started

*   Explore the full documentation at [docs.all-hands.dev](https://docs.all-hands.dev) to learn more about features, setup, and configuration.
*   [Sign up for OpenHands Cloud](https://app.all-hands.dev) and receive $20 in free credits.

> [!IMPORTANT]
>
>  If you're using OpenHands for work, we invite you to join our Design Partner program to help shape the future of our product.  Fill out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) and receive early access to new commercial features, and have an opportunity to provide input to our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The easiest way to get started with OpenHands is through [OpenHands Cloud](https://app.all-hands.dev). New users receive $20 in free credits to get up and running quickly.

## 💻 Running OpenHands Locally

You can also run OpenHands locally using Docker. Follow the steps in the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and installation.

> [!WARNING]
> For secure local deployments on a public network, consult our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.49
```

> **Note**: If you used OpenHands before version 0.44, migrate your conversation history by running `mv ~/.openhands-state ~/.openhands`.

Access OpenHands at [http://localhost:3000](http://localhost:3000). You will be asked to choose an LLM provider and add your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) provides the best results, but [many other options](https://docs.all-hands.dev/usage/llms) are supported.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation deployments and is not suitable for multi-tenant environments due to the lack of built-in authentication, isolation, or scalability.
>
> For multi-tenant use, explore the source-available, commercially-licensed [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud).

Integrate OpenHands with your workflow in various ways:

*   Connect to your local filesystem: [Connecting to your filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Headless mode: [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   CLI Mode: [CLI Mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   GitHub Action: [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action).

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for detailed installation and setup guides.

To modify OpenHands source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

For troubleshooting assistance, see the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

For comprehensive information and tips on using OpenHands, consult the [documentation](https://docs.all-hands.dev/usage/getting-started).  Find resources on:
*   Utilizing various LLM providers.
*   Troubleshooting common issues.
*   Advanced configuration.

## 🤝 Join the Community

OpenHands thrives on community contributions!  The best way to connect is through Slack. We also welcome you on Discord and GitHub:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - Discuss research, architecture, and development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - General discussion, support, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Review current issues or suggest new ideas.

Learn more about community engagement in [COMMUNITY.md](./COMMUNITY.md) and contributing guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1), updated after each monthly maintainer meeting.

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

OpenHands is a collaborative project, and we are deeply grateful for all contributions. We also acknowledge and thank the open-source projects that support our work.

For a comprehensive list of open-source projects and licenses, see [CREDITS.md](./CREDITS.md).

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