<!-- Improved README.md -->
<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  </a>
  <h1>OpenHands: The AI-Powered Software Development Agent That Codes for You</h1>
  <p><i>Code Less, Make More with AI-powered software development.</i></p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/stargazers">
      <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
    </a>
    <br/>
    <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA">
      <img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community">
    </a>
    <a href="https://discord.gg/ESHStjSjD4">
      <img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md">
      <img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits">
    </a>
    <br/>
    <a href="https://docs.all-hands.dev/usage/getting-started">
      <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation">
    </a>
    <a href="https://arxiv.org/abs/2407.16741">
      <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv">
    </a>
    <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
      <img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score">
    </a>
  </p>

  <!-- Translation Links (Keep these) -->
  <p>
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>
  </p>
  <hr>
</div>

OpenHands (formerly OpenDevin) is a cutting-edge platform offering AI-powered software development agents designed to automate and streamline coding tasks. Visit the [OpenHands GitHub Repository](https://github.com/All-Hands-AI/OpenHands) for more information.

## Key Features

*   **AI-Powered Automation:** Automate coding tasks with AI agents.
*   **Code Modification and Execution:** Modify code, run commands, and call APIs.
*   **Web Browsing and Information Retrieval:** Browse the web to gather information.
*   **Stack Overflow Integration:**  Utilize code snippets from Stack Overflow directly.
*   **Cloud and Local Deployment:** Run OpenHands via OpenHands Cloud or locally using Docker.
*   **Flexible Integration:**  Connect to your local filesystem and interact via CLI or Github Actions.

## Getting Started

Explore the power of AI-driven software development with OpenHands Cloud or set up a local deployment.

*   **OpenHands Cloud:** The easiest way to get started, with $20 in free credits for new users.  Visit [OpenHands Cloud](https://app.all-hands.dev).

*   **Local Deployment with Docker:**

    1.  **Install Docker:** Ensure you have Docker installed on your system.
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

    4.  **Access OpenHands:**  Open your web browser and navigate to [http://localhost:3000](http://localhost:3000).
    5.  **Configure:** Choose an LLM provider and add your API key.  Anthropic's Claude Sonnet 4 (`anthropic/claude-sonnet-4-20250514`) is recommended.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) to secure your deployment.

> **Note:** If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

## Other Deployment Options

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Run OpenHands in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Use a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run it on tagged issues with a [Github action](https://docs.all-hands.dev/usage/how-to/github-action)

## Important Considerations

> [!WARNING]
> OpenHands is not designed for multi-tenant deployments and is intended for single-user local workstation use due to the lack of built-in authentication, isolation, and scalability. For multi-tenant environments, explore the [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud), which is source-available and commercially licensed.

## Documentation & Resources

*   **Documentation:** Comprehensive guides and information are available at [docs.all-hands.dev](https://docs.all-hands.dev).
*   **DeepWiki Autogenerated Documentation**: Learn more about the project and its features via [Ask DeepWiki](https://deepwiki.com/All-Hands-AI/OpenHands).

## Community & Support

Join our vibrant community for discussions, support, and collaboration.

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   **GitHub Issues:** [Report issues or propose ideas](https://github.com/All-Hands-AI/OpenHands/issues)

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for further details on the community and how to contribute.

## Progress & Roadmap

*   Track project progress with the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License.  See [`LICENSE`](./LICENSE) for more information.

## Acknowledgements

OpenHands is a community-driven project. For a list of open source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

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