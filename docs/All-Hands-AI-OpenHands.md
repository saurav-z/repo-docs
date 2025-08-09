<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: AI-Powered Software Development for Rapid Creation</h1>
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

**OpenHands is an open-source platform empowering developers with AI-driven tools to code more efficiently, offering a significant boost to productivity and innovation.**  [Check out the original repository](https://github.com/All-Hands-AI/OpenHands)!

## Key Features of OpenHands:

*   **AI-Powered Code Assistance:** Leverage the power of AI to modify code, run commands, and automate development tasks.
*   **Web Browsing & API Integration:** Seamlessly browse the web and integrate with APIs to access external resources and data.
*   **StackOverflow Integration:**  Automatically copy code snippets from Stack Overflow, accelerating your development process.
*   **Local & Cloud Deployment Options:** Run OpenHands locally with Docker or utilize OpenHands Cloud for an easy setup.
*   **Community-Driven:** Benefit from an active community and contribute to the project's growth.

## Getting Started with OpenHands

Get up and running with OpenHands using the following methods:

### OpenHands Cloud

The easiest way to get started is by using [OpenHands Cloud](https://app.all-hands.dev), which provides a user-friendly interface and comes with $20 in free credits for new users.

### Running OpenHands Locally

Run OpenHands on your local system using Docker.

1.  **Install Docker:** Ensure you have Docker installed and running on your system.
2.  **Pull the Image:** Pull the latest OpenHands image from the Docker registry:

    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik
    ```
3.  **Run the Container:** Execute the following Docker command to run OpenHands:

    ```bash
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
    Access OpenHands via your browser at [http://localhost:3000](http://localhost:3000).
4.  **Configure LLM:**  When prompted, select your preferred LLM provider and input the necessary API key.  Anthropic's Claude Sonnet 4 (`anthropic/claude-sonnet-4-20250514`) is recommended, but many [other options](https://docs.all-hands.dev/usage/llms) are available.

> **Note:** If you used OpenHands before version 0.44, you may need to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.
> **Important**: Consider the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) for securing your local deployment.

### Other Ways to Run OpenHands

Explore these additional setup options:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Run OpenHands in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting.
*   Interact via the [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Integrate it with a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action).

**Note:** OpenHands is optimized for single-user, local workstation use. Multi-tenant deployments are not recommended.

## Documentation

For detailed information and usage tips, consult the comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started), including:

*   LLM Provider configurations.
*   Troubleshooting resources.
*   Advanced configuration options.

## Join the OpenHands Community

Contribute and collaborate with other developers:

*   **Slack:** Join our [Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for discussions on research, architecture, and development.
*   **Discord:** Join our [Discord server](https://discord.gg/ESHStjSjD4) for general discussions and feedback.
*   **GitHub Issues:** Review and submit [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) to share ideas.

Refer to [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for community guidelines and contribution details.

## Project Roadmap & Progress

Track our monthly progress on the OpenHands [roadmap](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is licensed under the [MIT License](./LICENSE).

## Acknowledgements

We appreciate the contributions of all OpenHands contributors and the open-source projects we build upon.  See our [CREDITS.md](./CREDITS.md) file for a complete list.

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