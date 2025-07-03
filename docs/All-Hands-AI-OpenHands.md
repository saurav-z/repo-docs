<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Code Smarter, Not Harder with AI-Powered Software Development</h1>
  <p><em>Unlock the power of AI to automate coding tasks and accelerate your development workflow.</em></p>
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

OpenHands (formerly OpenDevin) is a groundbreaking platform that empowers developers with AI-driven automation.  Visit the [OpenHands GitHub Repository](https://github.com/All-Hands-AI/OpenHands) for the latest updates and contributions.

**Key Features:**

*   **Automated Code Modification:**  Effortlessly modify code, saving time and reducing manual effort.
*   **Web Browsing & API Integration:**  Access information and call APIs to enhance your development process.
*   **StackOverflow Integration:**  Leverage the power of StackOverflow directly within your workflow, streamlining your coding.
*   **Flexible Deployment Options:** Run OpenHands on the cloud or locally using Docker for maximum versatility.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) to get started.

> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

Get started with OpenHands quickly and easily on [OpenHands Cloud](https://app.all-hands.dev), and receive $20 in free credits.

## 💻 Running OpenHands Locally

Run OpenHands locally using Docker for a fully customizable experience. Find detailed instructions in the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

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
    docker.all-hands-dev/all-hands-ai/openhands:0.48
```

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

Access OpenHands at [http://localhost:3000](http://localhost:3000) after the Docker container is up.

Configure your LLM provider and API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many [other options are available](https://docs.all-hands.dev/usage/llms).

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation environments.  It is not suitable for multi-tenant deployments.

Explore advanced usage options:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Run in scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Run on tagged issues with a [GitHub action](https://docs.all-hands.dev/usage/how-to/github-action).

Refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed setup instructions.

For source code modifications, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

If you encounter any issues, consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

<a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Comprehensive documentation is available at [docs.all-hands.dev/usage/getting-started], covering LLM providers, troubleshooting, and advanced configurations.

## 🤝 Join the OpenHands Community

Contribute to OpenHands!  Connect with us through:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   [Review and contribute on GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues)

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

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

OpenHands is a community project, and we appreciate all contributions. See [CREDITS.md](./CREDITS.md) for a list of open-source dependencies.

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