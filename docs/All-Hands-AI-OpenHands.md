# OpenHands: Supercharge Your Software Development with AI-Powered Agents

[OpenHands](https://github.com/All-Hands-AI/OpenHands) empowers developers to build software faster and more efficiently with AI-driven agents.

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

<!-- Keep these links. Translations will automatically update with the README. -->
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>

---

## Key Features

*   **AI-Powered Agents:** Utilize intelligent agents that can perform various developer tasks.
*   **Code Modification:** Modify existing codebases effortlessly.
*   **Command Execution:** Run commands within the development environment.
*   **Web Browsing & API Integration:** Agents can browse the web and interact with APIs.
*   **StackOverflow Integration:**  Access and use code snippets directly from StackOverflow.
*   **OpenHands Cloud:** Easily get started with the cloud version, including a free trial period.
*   **Local Deployment:** Run OpenHands on your local system using Docker.
*   **Flexible Usage:** Integrate with CLI, Headless Mode, and GitHub Actions for your workflow.

## What is OpenHands?

OpenHands is a pioneering platform for AI-powered software development, enabling developers to accomplish more with less code.  OpenHands agents can perform many tasks a human developer can, including modifying code, running commands, browsing the web, calling APIs, and even using code snippets from StackOverflow. Learn more at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) to get started.

> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

The easiest way to start with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

## 💻 Running OpenHands Locally

You can also run OpenHands locally using Docker; see the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
> to secure your deployment by restricting network binding and implementing additional security measures.

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

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

Access OpenHands at [http://localhost:3000](http://localhost:3000).  You'll need to choose an LLM provider and add an API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.  See [many options](https://docs.all-hands.dev/usage/llms).

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user local use and is not suitable for multi-tenant deployments.

You can [connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem), use [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode), interact via a [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode), or integrate it with a [GitHub action](https://docs.all-hands.dev/usage/how-to/github-action).

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for setup instructions.

For source code modification, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

For troubleshooting help, see the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Explore our [documentation](https://docs.all-hands.dev/usage/getting-started) for project details, LLM provider guidance, troubleshooting, and advanced configuration.

## 🤝 How to Join the Community

OpenHands thrives on community contributions.

-   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for research, architecture, and development discussions.
-   [Join our Discord server](https://discord.gg/ESHStjSjD4) for general discussions, questions, and feedback.
-   [Review and post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) to contribute ideas.

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands benefits from numerous contributors and open-source projects.  See [CREDITS.md](./CREDITS.md) for details.

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