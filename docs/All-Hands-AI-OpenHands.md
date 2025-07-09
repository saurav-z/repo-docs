<!-- Improved & SEO-Optimized README -->
<a name="readme-top"></a>

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1 align="center">OpenHands: AI-Powered Software Development for Faster Results</h1>
  <p align="center">
    <em>Code less, achieve more with OpenHands, your AI-powered software development agent.</em>
  </p>
</div>

<div align="center">
  <!-- Badges and Links -->
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

## What is OpenHands?

OpenHands is a cutting-edge platform designed to revolutionize software development, leveraging the power of AI to streamline your workflow. Explore the original repository on [GitHub](https://github.com/All-Hands-AI/OpenHands).

## Key Features

*   **AI-Powered Code Modification:** Modify and generate code with ease using intelligent AI agents.
*   **Web Browsing & API Integration:** Access the web, call APIs, and integrate external services directly into your development process.
*   **Contextual Code Retrieval:**  Copy code snippets and solutions from Stack Overflow and other resources.
*   **Cloud & Local Deployment:**  Easily deploy and run OpenHands through the cloud or on your local machine.
*   **Community Driven:** Benefit from a thriving community and contribute to the project's growth.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

Get started quickly with OpenHands Cloud, offering $20 in free credits for new users. This is the easiest way to experience the power of OpenHands.
[Get Started with OpenHands Cloud](https://app.all-hands.dev).

## 💻 Running OpenHands Locally

For local deployment, OpenHands is Docker-compatible. Detailed setup and system requirements can be found in the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

> [!WARNING]
> Ensure you review the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) if deploying on a public network to enhance security.

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
    docker.all-hands-dev/all-hands-ai/openhands:0.48
```

> **Note:** If upgrading from a version before 0.44, consider migrating your history using `mv ~/.openhands-state ~/.openhands`.

Access OpenHands at [http://localhost:3000](http://localhost:3000). Select an LLM provider and add an API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many other [LLM options](https://docs.all-hands.dev/usage/llms) are supported.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use and is not intended for multi-tenant deployments.

Explore various deployment methods, including:
*   Connecting to your [local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Running in a [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Interacting via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Utilizing a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action).

Find comprehensive information in the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

For source code modifications, review [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).
Troubleshooting assistance is available in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Access detailed information and tips in our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started).

## 🤝 Join the Community

OpenHands thrives on community contributions. Connect with us on Slack, Discord, and GitHub:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) to discuss research and development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) for community discussions, Q&A, and feedback.
*   [Review or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) to contribute ideas.

See [COMMUNITY.md](./COMMUNITY.md) for community details and [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines.

## 📈 Progress

Stay updated with the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is a community-driven project. We appreciate all contributors and acknowledge the open-source projects we build upon.

See [CREDITS.md](./CREDITS.md) for a complete list of open-source projects and licenses.

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