[![OpenHands Logo](docs/static/img/logo.png)](https://github.com/All-Hands-AI/OpenHands)

# OpenHands: AI-Powered Software Development Agents

**OpenHands empowers you to code less and achieve more by leveraging the power of AI to automate software development tasks.**

[Link to Original Repo: OpenHands on GitHub](https://github.com/All-Hands-AI/OpenHands)

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Arxiv Paper](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

<!-- Keep these links. Translations will automatically update with the README. -->
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>

---

## Key Features

*   **AI-Powered Agents:** OpenHands agents can perform a wide range of developer tasks autonomously.
*   **Code Modification & Execution:** Agents can modify code, run commands, and call APIs.
*   **Web Browsing & Information Retrieval:** OpenHands can browse the web to gather information and find solutions.
*   **Integration with Existing Tools:** Easily integrate with your current development workflow.
*   **Open Source & Customizable:** Leverage the flexibility of open source to adapt OpenHands to your needs.

## Get Started

OpenHands offers several ways to jump in:

*   **[OpenHands Cloud](https://app.all-hands.dev):** The easiest way to get started, with free credits for new users!

    ![App screenshot](./docs/static/img/screenshot.png)

*   **Local Deployment:**  Run OpenHands on your local machine using Docker.  See the [Installation Guide](https://docs.all-hands.dev/usage/installation) for details.

    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.47-nikolaik

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

    *   OpenHands will be available at [http://localhost:3000](http://localhost:3000).
    *   For optimal results, we recommend using [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api).

*   **Other Deployment Options:** Explore [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode), the [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode), and a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action).

> [!WARNING]
> OpenHands is designed for single-user, local use and is not intended for multi-tenant deployments.

## Resources

*   **Documentation:**  Explore comprehensive documentation at [docs.all-hands.dev](https://docs.all-hands.dev) for detailed usage instructions, LLM provider configuration, and troubleshooting tips.
*   **Community:**
    *   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA): Discuss research, architecture, and development.
    *   [Discord](https://discord.gg/ESHStjSjD4): Community-run server for general discussion and feedback.
    *   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues): Report issues and propose ideas.
*   **Development:** Review [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md) for contributing to the project.

## Roadmap and Progress

Track our monthly roadmap and progress [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is released under the [MIT License](./LICENSE).

## Acknowledgements

We are grateful to the large number of contributors and open source projects that made OpenHands possible. See [CREDITS.md](./CREDITS.md) for details.

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