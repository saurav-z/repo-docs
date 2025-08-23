<!-- Improved README with SEO and Summarization -->
<a name="readme-top"></a>

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Automate Software Development with AI</h1>
  <p><em>Unlock unprecedented productivity by letting AI agents handle your software development tasks.</em></p>
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

**OpenHands** is an open-source platform empowering AI agents to perform a wide range of software development tasks, dramatically accelerating your workflow.  Find out more on the original repo: [https://github.com/All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands).

## Key Features

*   **AI-Powered Automation:**  Leverage AI agents to perform tasks just like human developers.
*   **Code Modification & Execution:**  Modify code, run commands, and streamline your development process.
*   **Web Browsing & API Integration:**  Enable your AI agents to browse the web and interact with APIs.
*   **Stack Overflow Integration:**  Access and utilize code snippets directly from Stack Overflow.
*   **Cloud & Local Options:** Available as OpenHands Cloud and locally installable.

## Getting Started

Explore the powerful capabilities of OpenHands through these options:

*   **OpenHands Cloud:** The simplest way to get started. Visit [OpenHands Cloud](https://app.all-hands.dev) and receive $20 in free credits for new users.

*   **Local Installation (CLI Launcher - Recommended)**
    1.  **Install uv:** Follow the instructions at the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
    2.  **Launch OpenHands:**
        ```bash
        # Launch the GUI server
        uvx --python 3.12 --from openhands-ai openhands serve

        # Or launch the CLI
        uvx --python 3.12 --from openhands-ai openhands
        ```
        You'll find OpenHands running at [http://localhost:3000](http://localhost:3000) (for GUI mode)!

*   **Local Installation (Docker)**
    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik

    docker run -it --rm --pull=always \
        -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik \
        -e LOG_ALL_EVENTS=true \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v ~/.openhands:/.openhands \
        -p 3000:3000 \
        --add-host host.docker.internal:host-gateway \
        --name openhands-app \
        docker.all-hands.dev/all-hands-ai/openhands:0.54
    ```

    > **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

    > [!WARNING]
    > On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
    > to secure your deployment by restricting network binding and implementing additional security measures.

*   **Configuration:** When you open the application, you'll be asked to choose an LLM provider and add an API key.
    [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`)
    works best, but you have [many options](https://docs.all-hands.dev/usage/llms).

    See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for
    system requirements and more information.

## Other Ways to Run OpenHands

*   **Filesystem Integration:**  Connect OpenHands to your local file system for direct access.
*   **CLI Mode:**  Interact with OpenHands through a user-friendly command-line interface.
*   **Headless Mode:** Run OpenHands in a scriptable, headless environment.
*   **GitHub Action:** Integrate OpenHands into your workflow with a GitHub action.

  Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more information and setup instructions.

If you want to modify the OpenHands source code, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Having issues? The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## Documentation

To learn more about the project, and for tips on using OpenHands,
check out our [documentation](https://docs.all-hands.dev/usage/getting-started).

There you'll find resources on how to use different LLM providers,
troubleshooting resources, and advanced configuration options.

## Community and Contribution

OpenHands is a community-driven project and welcomes contributions.

*   **Slack:**  Join our Slack workspace for research, architecture, and development discussions. ([Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA))
*   **Discord:**  Participate in general discussion and feedback on our Discord server. ([Join our Discord server](https://discord.gg/ESHStjSjD4))
*   **GitHub Issues:**  Review or submit issues on GitHub. ([Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues))

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## Progress & Roadmap

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the maintainer's meeting at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is built by numerous contributors and leverages other open-source projects. A list of those projects and their licenses can be found in [CREDITS.md](./CREDITS.md).

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