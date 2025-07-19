<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Revolutionizing Software Development with AI</h1>
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

## Introduction

OpenHands is an open-source platform that empowers AI agents to perform software development tasks, allowing you to build more with less code.  [Check out the original repo](https://github.com/All-Hands-AI/OpenHands) for more information.

## Key Features

*   **AI-Powered Agents:** Leverage AI to automate code modification, command execution, web browsing, and API calls.
*   **Code Generation & Integration:**  Agents can copy code snippets from Stack Overflow and other online resources.
*   **Local & Cloud Deployment:** Run OpenHands locally using Docker or easily get started on OpenHands Cloud.
*   **Flexible Integration:** Connect to your local filesystem, use headless mode, interact via CLI, or integrate with GitHub Actions.
*   **Community Driven:**  A vibrant community of contributors.

## Getting Started

The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

### Running OpenHands Locally

You can also run OpenHands locally using Docker:

1.  **Pull the Docker Image:**
    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik
    ```

2.  **Run the Container:**
    ```bash
    docker run -it --rm --pull=always \
        -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik \
        -e LOG_ALL_EVENTS=true \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v ~/.openhands:/.openhands \
        -p 3000:3000 \
        --add-host host.docker.internal:host-gateway \
        --name openhands-app \
        docker.all-hands.dev/all-hands-ai/openhands:0.49
    ```

3.  **Access OpenHands:** Open your browser and navigate to [http://localhost:3000](http://localhost:3000).

**Important Notes:**

*   If you used OpenHands before version 0.44, migrate your conversation history: `mv ~/.openhands-state ~/.openhands`.
*   For increased security, especially on public networks, see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

## Configuration

When you open the application, you'll be asked to choose an LLM provider and add an API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works best, but many other options are available in the [documentation](https://docs.all-hands.dev/usage/llms).

## Additional Ways to Run OpenHands

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

##  Documentation

Comprehensive documentation is available to help you get started with OpenHands:
*   **[Documentation](https://docs.all-hands.dev/usage/getting-started)**:  Learn how to use different LLM providers, troubleshoot issues, and configure advanced options.
*   **[DeepWiki Documentation](https://deepwiki.com/All-Hands-AI/OpenHands)**:  Automatically generated documentation.

## Join the Community

OpenHands thrives on community contributions!  Connect with us:

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   **GitHub Issues:** [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)
*   **[COMMUNITY.md](./COMMUNITY.md)**: For more information about our community.
*   **[CONTRIBUTING.md](./CONTRIBUTING.md)**: Details on contributing to the project.

## Progress

Check out the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the maintainer's meeting at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is a collaborative project. We are grateful for the contributions of the community and the open-source projects we build upon.

*   For a list of the open-source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

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