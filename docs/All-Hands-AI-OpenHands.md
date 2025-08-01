<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: AI-Powered Software Development Agents</h1>
</div>

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![MIT License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
<br/>
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Join Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Project Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
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

OpenHands empowers developers with AI-driven agents that can write and manage code, effectively automating software development tasks.  

**Key Features:**

*   **AI-Powered Agents:** Automate coding tasks with intelligent agents.
*   **Code Modification:**  Modify code, debug, and refactor with ease.
*   **Web Browsing & API Integration:** Access information and interact with APIs.
*   **Code Snippet Integration:** Leverage existing code examples from StackOverflow.
*   **Local & Cloud Deployment:** Run OpenHands on your local machine with Docker, or on the cloud.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev) or get started on [OpenHands Cloud](https://app.all-hands.dev).

> [!IMPORTANT]
> Interested in using OpenHands for work? Join our Design Partner program by filling out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) to get early access to commercial features and provide feedback.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The easiest way to get started is on [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

## 💻 Running OpenHands Locally

You can also run OpenHands locally using Docker.  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and setup instructions.

> [!WARNING]
> When running on a public network, review the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) for security best practices.

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

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

OpenHands will be running at [http://localhost:3000](http://localhost:3000).

When you open the application, you will be asked to choose an LLM provider and add an API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works best, but you have [many options](https://docs.all-hands.dev/usage/llms).

## 💡 Other Deployment Options

> [!WARNING]
> OpenHands is designed for single-user, local workstation deployments and is not suitable for multi-tenant environments without additional security measures.

Explore these alternative ways to run OpenHands:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Run in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Interact via the [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Use a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action).

Refer to [Running OpenHands](https://docs.all-hands.dev/usage/installation) for details.

For contributing to the OpenHands source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).  For troubleshooting, see the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Access comprehensive project details, usage tips, and more in our [documentation](https://docs.all-hands.dev/usage/getting-started). This includes resources on LLM providers, troubleshooting, and advanced configuration.

## 🤝 Join the Community

OpenHands thrives on community contributions!  Connect with us on Slack, Discord, or GitHub:

*   [Join our Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - Discuss research, architecture, and development.
*   [Join our Discord](https://discord.gg/ESHStjSjD4) - A community-run server for discussions, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Explore and contribute to ongoing issues.

Read [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for further information.

## 📈 Progress

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

OpenHands is a community project, built by numerous contributors and built upon other open-source projects.

See [CREDITS.md](./CREDITS.md) for a list of the open-source projects and licenses used.

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