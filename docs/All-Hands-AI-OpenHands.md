[![OpenHands Logo](docs/static/img/logo.png)](https://github.com/All-Hands-AI/OpenHands)

# OpenHands: Revolutionize Software Development with AI

**OpenHands empowers developers to build more, code less, by providing AI-powered software development agents.**  ([Original Repo](https://github.com/All-Hands-AI/OpenHands))

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


## Key Features

*   **AI-Powered Agents:** OpenHands agents can perform a wide range of developer tasks.
*   **Code Modification:**  Modify code, fix bugs, and implement new features.
*   **Web Browsing & API Integration:** Browse the web and integrate with APIs to access information and services.
*   **Automated Tasks:**  Run commands and automate tasks to streamline your workflow.
*   **Open Source:**  Leverage the power of open-source technology and contribute to the community.
*   **StackOverflow Integration:** Copy code snippets directly from StackOverflow.

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started

OpenHands offers flexible deployment options to fit your needs:

*   **OpenHands Cloud:** The easiest way to start, with free credits for new users. [OpenHands Cloud](https://app.all-hands.dev)

*   **Local Installation:** Run OpenHands on your local machine using the CLI launcher or Docker.

    *   **CLI Launcher (Recommended):**  Utilizes [uv](https://docs.astral.sh/uv/) for environment isolation.

        ```bash
        # Launch the GUI server
        uvx --python 3.12 --from openhands-ai openhands serve

        # Or launch the CLI
        uvx --python 3.12 --from openhands-ai openhands
        ```

    *   **Docker:**

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


    *   **Configuration:** Select an LLM provider and provide your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) works best.

    *   **Further Info:**  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for more information and setup instructions.

## Other Ways to Run OpenHands

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Interact with a friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Run headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

## Documentation

To learn more about OpenHands:
*   [Documentation](https://docs.all-hands.dev/usage/getting-started)
*   [DeepWiki](https://deepwiki.com/All-Hands-AI/OpenHands)

## Join the Community

We welcome your contributions!

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)

See [COMMUNITY.md](./COMMUNITY.md) for details.

## Progress and Roadmap

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## Acknowledgements

OpenHands is built by a large number of contributors, and every contribution is greatly appreciated! We also build upon other open source projects, and we are deeply thankful for their work.

For a list of open source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

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