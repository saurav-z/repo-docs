<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands"><img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200"></a>
  <h1 align="center">OpenHands: Revolutionize Software Development with AI</h1>
  <p align="center"><i>Code Less, Achieve More with the Power of AI Agents</i></p>
</div>

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
<br/>
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
<br/>
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de) |
[Español](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es) |
[Français](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr) |
[日本語](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja) |
[한국어](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko) |
[Português](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt) |
[Русский](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru) |
[中文](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh)

<hr>

OpenHands is a cutting-edge AI platform that empowers developers to build software faster and more efficiently.  These intelligent agents can perform complex tasks just like human developers, from writing and modifying code to browsing the web and even debugging – all with minimal manual intervention!  Explore the [OpenHands GitHub Repository](https://github.com/All-Hands-AI/OpenHands) to learn more.

**Key Features:**

*   **AI-Powered Code Generation and Modification:** Automate coding tasks, reduce development time, and improve code quality.
*   **Web Browsing and API Integration:**  OpenHands agents can access online resources and interact with APIs for comprehensive functionality.
*   **Intelligent Task Execution:** Execute commands, run tests, and manage the development lifecycle with ease.
*   **Code Retrieval from Stack Overflow:**  Leverage the power of existing code resources to accelerate development.
*   **Multi-Platform Deployment:** Deploy and run OpenHands in the cloud and locally with Docker.
*   **Community-Driven Development:**  Collaborate with a vibrant community through Slack, Discord, and GitHub.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) to get started.

> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud: Get Started Quickly

The easiest way to experience OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), offering new users \$20 in free credits.

## 💻 Running OpenHands Locally with Docker

You can run OpenHands on your local system using Docker. Follow these steps:

1.  **System Requirements:** Ensure you meet the necessary system requirements. Check the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed information.
2.  **Hardened Docker Installation (Recommended):** If running on a public network, secure your deployment by consulting the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation). This guide provides crucial security measures like restricting network binding.
3.  **Docker Run Command:** Execute the following Docker command to run OpenHands:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.51
```

> **Note:** If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

Access OpenHands at [http://localhost:3000](http://localhost:3000). You will be prompted to choose an LLM provider and add an API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works best. See [many options](https://docs.all-hands.dev/usage/llms).

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user local workstation use and is not intended for multi-tenant deployments without modifications for authentication, isolation, and scalability.

Explore other ways to run OpenHands:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Run OpenHands in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Interact with OpenHands through a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run OpenHands via [a GitHub action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for detailed instructions.

For source code modifications, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).
Troubleshooting resources are available in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

For more details on the project, LLM providers, troubleshooting, and advanced configuration options, consult our [documentation](https://docs.all-hands.dev/usage/getting-started).

## 🤝 Join the OpenHands Community

OpenHands thrives on community contributions. Join our community through:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - Discuss research, architecture, and development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - For general discussion, questions, and feedback.
*   [Read or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Explore existing issues and contribute your ideas.

See [COMMUNITY.md](./COMMUNITY.md) for community details or [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines.

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

We are grateful to all OpenHands contributors and the open-source projects we build upon. Our [CREDITS.md](./CREDITS.md) file lists the open-source projects and licenses.

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