<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1 align="center">OpenHands: AI-Powered Software Development - Code Smarter, Not Harder</h1>
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

OpenHands (formerly OpenDevin) is an open-source platform that empowers AI agents to autonomously develop software, revolutionizing the way we build and deploy code.  [Explore the OpenHands project on GitHub](https://github.com/All-Hands-AI/OpenHands).

## Key Features

*   **Autonomous Code Generation:**  OpenHands agents can write, modify, and debug code with minimal human intervention.
*   **Web Browsing & API Integration:** Agents can search the web for information and interact with APIs to gather data or access services.
*   **Code Familiarity:** The agents can use and learn from StackOverflow for code snippets to enhance efficiency.
*   **Local & Cloud Deployment:**  Run OpenHands locally via Docker or leverage OpenHands Cloud for ease of use and quick startup.
*   **Community-Driven:**  Benefit from a vibrant and supportive community that actively contributes to development and provides assistance.

## Getting Started with OpenHands

Ready to experience the future of software development?

*   **OpenHands Cloud:** The simplest way to start is with [OpenHands Cloud](https://app.all-hands.dev), offering $20 in free credits for new users.

*   **Local Setup (Docker):** Run OpenHands locally using Docker for complete control and customization.  Consult the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and detailed instructions.

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

    Access OpenHands at [http://localhost:3000](http://localhost:3000). You'll be prompted to configure an LLM provider; [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.

*   **Important Note:** If you've used OpenHands before version 0.44, migrate your conversation history with `mv ~/.openhands-state ~/.openhands`.

## More Ways to Run OpenHands

Explore advanced usage scenarios:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting and automation.
*   [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode) for a command-line interface.
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action) integration for automated tasks.

For detailed setup and usage instructions, visit the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

## Community and Support

Join the growing OpenHands community:

*   **Slack:**  Share ideas and collaborate on [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA).
*   **Discord:** General discussion and feedback on [Discord](https://discord.gg/ESHStjSjD4).
*   **GitHub Issues:** Contribute or raise issues at [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues).

Learn more about community in [COMMUNITY.md](./COMMUNITY.md) and contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Documentation

Discover in-depth documentation for OpenHands:

*   Comprehensive documentation at [docs.all-hands.dev](https://docs.all-hands.dev/usage/getting-started).
*   [DeepWiki](https://deepwiki.com/All-Hands-AI/OpenHands) Autogenerated Documentation.

## Project Roadmap

Stay updated on OpenHands' progress and future development plans:

*   View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

## Acknowledgements

OpenHands is a collaborative effort, and we appreciate all contributions. Our work builds upon other open-source projects.

*   See a list of the projects and licenses used in OpenHands, [CREDITS.md](./CREDITS.md)

## License

This project is licensed under the MIT License. For more details, refer to the [`LICENSE`](./LICENSE) file.

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