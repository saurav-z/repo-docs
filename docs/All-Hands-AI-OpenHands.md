<!-- Improved README for OpenHands -->
<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Supercharge Your Development with AI</h1>
  <p><em>Code less, build more with OpenHands, the AI-powered software development platform.</em></p>
</div>

<div align="center">
  <!-- Badges -->
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

## Key Features of OpenHands

*   **AI-Powered Agents:** Utilize AI agents to automate and accelerate software development tasks.
*   **Code Modification:** Modify existing code, refactor, and implement new features with ease.
*   **Web Browsing & API Integration:** Browse the web and integrate with APIs to gather information and extend functionality.
*   **Stack Overflow Integration:** Leverage the power of Stack Overflow to find and implement code snippets directly.
*   **Flexible Deployment:** Run OpenHands on the cloud, locally with Docker, or integrate via various methods.

[Visit the OpenHands GitHub Repository](https://github.com/All-Hands-AI/OpenHands) for the latest updates and to contribute.

## Getting Started with OpenHands

OpenHands empowers developers with AI agents capable of performing complex tasks, mimicking human developers, and streamlining the software development lifecycle. Whether you're modifying code, running commands, browsing the web, or integrating with APIs, OpenHands has you covered.

### ☁️ OpenHands Cloud

The quickest way to get started is with [OpenHands Cloud](https://app.all-hands.dev), which offers $20 in free credits for new users.

### 💻 Running OpenHands Locally

OpenHands can also be run locally using Docker:

1.  **Prerequisites:** Ensure you have Docker installed on your system.
2.  **Docker Pull:**
    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.47-nikolaik
    ```
3.  **Docker Run:**
    ```bash
    docker run -it --rm --pull=always \
        -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.47-nikolaik \
        -e LOG_ALL_EVENTS=true \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v ~/.openhands:/.openhands \
        -p 3000:3000 \
        --add-host host.docker.internal:host-gateway \
        --name openhands-app \
        docker.all-hands.dev/all-hands-ai/openhands:0.47
    ```
4.  **Access:** Open OpenHands in your browser at [http://localhost:3000](http://localhost:3000).

    **Note:** If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.
    **Important:** If you are on a public network, please see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)

5.  **Configuration:** Choose an LLM provider (e.g., [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api), `anthropic/claude-sonnet-4-20250514`) and provide your API key.

### 💡 Other Ways to Run OpenHands

*   **Filesystem Connection:** Connect to your local filesystem. ([Documentation](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem))
*   **Headless Mode:** Run OpenHands in a scriptable headless mode. ([Documentation](https://docs.all-hands.dev/usage/how-to/headless-mode))
*   **CLI Mode:** Interact via a friendly CLI. ([Documentation](https://docs.all-hands.dev/usage/how-to/cli-mode))
*   **GitHub Action:** Run it on tagged issues via a GitHub Action. ([Documentation](https://docs.all-hands.dev/usage/how-to/github-action))

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed instructions and system requirements.

> [!WARNING]
> OpenHands is designed for single-user, local workstation use. Multi-tenant deployments with shared instances are not recommended due to the lack of built-in authentication, isolation, and scalability.
>
> For multi-tenant environments, consider the [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud).

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Explore the comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started) to learn more about OpenHands, including:

*   LLM provider setup
*   Troubleshooting resources
*   Advanced configuration options

## 🤝 Join the Community

OpenHands thrives on community contributions. Connect with us:

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for research, architecture discussions, and development updates.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) for general discussion, questions, and feedback.
*   **GitHub Issues:** [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues) to share ideas and track progress.

Find more about the community in [COMMUNITY.md](./COMMUNITY.md) and details on how to contribute in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Project Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

OpenHands is built through the contributions of many individuals and the support of the open-source community. Thank you!

For a list of the open source projects used in OpenHands, refer to our [CREDITS.md](./CREDITS.md) file.

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