<!-- README.md -->
<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: The AI-Powered Software Development Platform</h1>
  <p><i>Stop coding and start creating: OpenHands empowers you to build software faster with AI.</i></p>
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

<p><b><a href="https://github.com/All-Hands-AI/OpenHands">OpenHands</a></b> is a cutting-edge platform designed to revolutionize software development by leveraging the power of AI agents.</p>

## Key Features

*   🤖 **AI-Powered Agents:** OpenHands agents can perform complex tasks like code modification, running commands, web browsing, API calls, and more.
*   💻 **Local and Cloud Deployment:**  Run OpenHands locally with Docker or easily get started on the cloud with free credits.
*   🌐 **Web Integration:** Seamlessly browse the web and integrate with online resources.
*   🛠️ **CLI & Headless Mode:** Interact through a friendly CLI or utilize headless mode for scripting and automation.
*   🤝 **Active Community:** Join our Slack and Discord communities to connect with fellow developers and contribute to the project.

## Getting Started

Explore OpenHands through these options:

*   **OpenHands Cloud:** The easiest way to get started, with $20 in free credits for new users: [OpenHands Cloud](https://app.all-hands.dev).
*   **Local Installation (Docker):**  Run OpenHands on your local machine. See our [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.
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
    OpenHands will be accessible at [http://localhost:3000](http://localhost:3000).  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) is recommended, but other LLMs are supported.

## Advanced Usage

*   **Connect to Your Filesystem:**  Learn how to connect OpenHands to your local file system:  [Connecting to Your Filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   **Headless Mode:** Run OpenHands in a scriptable, headless mode:  [Headless Mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   **CLI Mode:**  Interact with OpenHands using a friendly CLI: [CLI Mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   **GitHub Action:** Integrate OpenHands with GitHub Actions:  [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

## Security

> [!WARNING]
> OpenHands is designed for single-user local workstations and is not intended for multi-tenant deployments. It lacks built-in authentication, isolation, or scalability. For multi-tenant environments, consider the [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud).
>
> If deploying on a public network, use our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) to secure your deployment.

## ☁️ OpenHands Cloud

Get started instantly on [OpenHands Cloud](https://app.all-hands.dev) with free credits.

## 💡 Other Ways to Run OpenHands

Explore the documentation to discover the different deployment options: [docs.all-hands.dev](https://docs.all-hands.dev)

## 📖 Documentation

For detailed information, including LLM provider setup, troubleshooting, and advanced configuration, explore our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started).

## 🤝 Join the Community

We welcome contributions! Connect with us via:

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   **GitHub Issues:** [Report issues or suggest features](https://github.com/All-Hands-AI/OpenHands/issues)
*   **CONTRIBUTING.md:**  Find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Project Progress

See our roadmap for ongoing development:  [OpenHands Project Roadmap](https://github.com/orgs/All-Hands-AI/projects/1)

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is a community-driven project. See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses used.

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