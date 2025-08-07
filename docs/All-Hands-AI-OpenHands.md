<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Build Software Faster with AI-Powered Agents</h1>
</div>

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![MIT License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
<br/>
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
<br/>
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Arxiv Paper](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
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

<hr>

OpenHands empowers developers with AI agents that can code, debug, and deploy, letting you create software faster.

## Key Features

*   **AI-Powered Agents:** OpenHands agents automate software development tasks.
*   **Code Modification:** Modify existing codebases directly.
*   **Command Execution:** Run commands and manage your development environment.
*   **Web Browsing & API Interaction:** Access web resources and utilize APIs seamlessly.
*   **Code Snippet Integration:** Efficiently use and adapt code from external sources like Stack Overflow.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev) or get started on [OpenHands Cloud](https://app.all-hands.dev).

> [!IMPORTANT]
> Join our Design Partner program!
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to get early access and shape our product's direction.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

Get started with OpenHands quickly on [OpenHands Cloud](https://app.all-hands.dev), with $20 in free credits for new users.

## 💻 Running OpenHands Locally

You can also run OpenHands locally using Docker.  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for details.

> [!WARNING]
> Secure your local deployment with our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

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
    docker.all-hands-dev/all-hands-ai/openhands:0.51
```

> **Note**: Migrate your conversation history if needed: `mv ~/.openhands-state ~/.openhands`

Access OpenHands at [http://localhost:3000](http://localhost:3000).  You'll need an LLM provider and API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is for single-user, local workstations only.  It's not designed for multi-tenant environments.

Explore these alternative deployment options:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

See [Running OpenHands](https://docs.all-hands.dev/usage/installation) for setup details.

If you want to contribute to the source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

For troubleshooting, consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Comprehensive documentation is available at [docs.all-hands.dev/usage/getting-started].

## 🤝 How to Join the Community

Join the OpenHands community via:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Discord](https://discord.gg/ESHStjSjD4)
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues)

For details on contributing, refer to [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

View the monthly roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE).

## 🙏 Acknowledgements

OpenHands is built by a community.  See [CREDITS.md](./CREDITS.md) for a list of project dependencies and licenses.

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