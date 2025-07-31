<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: AI-Powered Software Development Agents</h1>
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

OpenHands (formerly OpenDevin) empowers developers with AI-driven software agents that can handle complex tasks, making coding more efficient.  [Explore the project on GitHub](https://github.com/All-Hands-AI/OpenHands).

**Key Features:**

*   **AI-Powered Agents:** Automate coding tasks with intelligent agents capable of human-level actions.
*   **Code Modification & Execution:** Modify code, run commands, and manage your projects seamlessly.
*   **Web Browsing & API Integration:** Access the web and integrate with APIs to streamline your development workflow.
*   **Stack Overflow Integration:**  Leverage code snippets from Stack Overflow directly within your projects.
*   **Flexible Deployment:** Run OpenHands on the cloud via OpenHands Cloud or locally with Docker.

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started with OpenHands

### OpenHands Cloud

The easiest way to get started is on [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

### Running OpenHands Locally with Docker

1.  **Install Docker:** Ensure Docker is installed on your system.
2.  **Pull the Image:**

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.50-nikolaik
```

3.  **Run the Container:**

```bash
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

4.  **Access OpenHands:** Open your web browser and navigate to [http://localhost:3000](http://localhost:3000).

**Note:** If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

**Configuration:** Select an LLM provider and add your API key when prompted. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.

## Other Ways to Run OpenHands

*   **Filesystem Connection:** [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   **Headless Mode:** Run OpenHands in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   **CLI Mode:** Interact with OpenHands via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   **GitHub Action:** Run it on tagged issues with [a GitHub action](https://docs.all-hands.dev/usage/how-to/github-action).

**Important Considerations:** OpenHands is designed for single-user, local workstation use. It is not intended for multi-tenant deployments.

## Documentation and Resources

*   **Comprehensive Documentation:** Find detailed information and usage tips in our [documentation](https://docs.all-hands.dev/usage/getting-started).
*   **DeepWiki:** <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>
*   **Troubleshooting Guide:** Get help with common issues in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).
*   **Development:** Learn how to modify the source code in [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## Join the OpenHands Community

We encourage contributions and discussions!

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - for research, architecture, and future development.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) - for general discussion, questions, and feedback.
*   **GitHub Issues:** [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues) - for contributing ideas and reporting issues.
*   **Community and Contributions:** Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Progress

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is licensed under the [MIT License](LICENSE).

## Acknowledgements

OpenHands is a community project, and we appreciate all contributions! We also utilize and are thankful for the open-source projects we build on.

*   **Credits:** See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses used.

## Cite

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