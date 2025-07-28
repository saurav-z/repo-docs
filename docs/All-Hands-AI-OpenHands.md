<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1 align="center">OpenHands: Code Less, Build More with AI-Powered Software Agents</h1>
</div>

<div align="center">
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

## OpenHands: Supercharge Your Software Development with AI

OpenHands (formerly OpenDevin) is a cutting-edge platform providing AI-powered software development agents to automate tasks and boost productivity.  Visit the [original repository](https://github.com/All-Hands-AI/OpenHands) for more information.

**Key Features:**

*   **AI-Powered Automation:** Leverage AI agents to perform complex coding tasks.
*   **Code Modification & Execution:** Modify code, run commands, and execute scripts seamlessly.
*   **Web Browsing & API Integration:** Access information online and interact with APIs.
*   **Stack Overflow Integration:**  Conveniently integrate code snippets from Stack Overflow.
*   **Local & Cloud Deployment:** Run OpenHands on your local machine or in the cloud.
*   **Community Driven:** Actively engage with the community on Slack and Discord.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

Get started quickly with OpenHands on [OpenHands Cloud](https://app.all-hands.dev), and receive $20 in free credits for new users.

## 💻 Running OpenHands Locally

Run OpenHands on your local system using Docker for development and testing. See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed system requirements.

> [!WARNING]
> On a public network? Secure your deployment by reviewing the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.50-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.50-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.50
```

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

Access OpenHands at [http://localhost:3000](http://localhost:3000) once the application is running.

Choose your preferred LLM provider and add your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is highly recommended, but you have [many options](https://docs.all-hands.dev/usage/llms).

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is best suited for single-user local deployments. Multi-tenant deployments require separate considerations for security and scalability.

Explore different deployment options:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Run OpenHands in a [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Utilize a [GitHub action](https://docs.all-hands.dev/usage/how-to/github-action) for automated tasks.

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for detailed setup instructions.

For source code modifications, refer to [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Troubleshooting? The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) is available.

## 📖 Documentation

<a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Comprehensive documentation can be found at [docs.all-hands.dev/usage/getting-started]. Explore resources on LLM providers, troubleshooting, and advanced configuration.

## 🤝 Join the OpenHands Community

OpenHands thrives on community contributions! Connect with us on Slack, Discord, and GitHub:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for discussions on research, architecture, and development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) for community-driven discussions.
*   [Read or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) to contribute ideas and report issues.

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

## 📈 Progress

Stay updated on the project roadmap at [All-Hands-AI/OpenHands/projects/1](https://github.com/orgs/All-Hands-AI/projects/1), updated monthly.

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is available under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

The project is built with contributions from many individuals, and we are thankful for their work and the open-source projects that make OpenHands possible.

A list of open-source projects and licenses used in OpenHands is available in [CREDITS.md](./CREDITS.md).

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