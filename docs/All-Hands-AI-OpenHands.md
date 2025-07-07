# OpenHands: Code Less, Make More with AI-Powered Software Development

[OpenHands](https://github.com/All-Hands-AI/OpenHands) empowers developers to automate tasks and build software faster using AI agents.

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de) |
[Español](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es) |
[français](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr) |
[日本語](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja) |
[한국어](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko) |
[Português](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt) |
[Русский](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru) |
[中文](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh)

---

OpenHands is a revolutionary platform for AI-powered software development agents that can perform the same tasks as human developers, helping you build more, faster.

## Key Features:

*   **AI-Powered Automation:** Automate code modification, command execution, web browsing, API calls, and more.
*   **Code Snippet Integration:** Easily incorporate code snippets from platforms like StackOverflow.
*   **Flexible Deployment:** Run OpenHands on the cloud, locally with Docker, or integrate it via scripts and GitHub actions.
*   **Community-Driven:** Benefit from a vibrant and supportive community for collaboration, feedback, and open-source contributions.

![App screenshot](./docs/static/img/screenshot.png)

## Get Started with OpenHands:

*   **OpenHands Cloud:** The easiest way to get started is on [OpenHands Cloud](https://app.all-hands.dev), with $20 in free credits for new users.
*   **Local Setup with Docker:** Run OpenHands on your local system using Docker.  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for details.

    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik

    docker run -it --rm --pull=always \
        -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik \
        -e LOG_ALL_EVENTS=true \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v ~/.openhands:/.openhands \
        -p 3000:3000 \
        --add-host host.docker.internal:host-gateway \
        --name openhands-app \
        docker.all-hands-ai/openhands:0.48
    ```

    *Note: After version 0.44, you may need to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.*

    Access OpenHands at [http://localhost:3000](http://localhost:3000) after setup. Use [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) for best results.

## Advanced Usage:

*   **Connect to Filesystem:** [Learn how to connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   **Headless Mode:** Run OpenHands in a [scriptable headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   **CLI Mode:** Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   **GitHub Action:** Run on tagged issues with [a GitHub action](https://docs.all-hands.dev/usage/how-to/github-action).

> [!WARNING]
> OpenHands is designed for single-user local environments. It does not offer built-in features for multi-tenant deployments, including authentication, isolation, or scalability.

## 💡 Other Resources:

*   **Documentation:** Explore detailed documentation at [docs.all-hands.dev](https://docs.all-hands.dev) to learn more, including LLM provider options, troubleshooting, and advanced configurations.
*   **Troubleshooting:** Find solutions in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).
*   **Development:** Modify the source code by following the guidelines at [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## 🤝 Join the OpenHands Community

OpenHands thrives on community contributions; join us to collaborate and shape the future of AI-powered software development!

*   **Slack:** Engage in discussions, architecture, and future developments at [OpenHands Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA).
*   **Discord:** Join the community-run server for general discussion, questions, and feedback at [OpenHands Discord](https://discord.gg/ESHStjSjD4).
*   **GitHub Issues:** Share ideas and check out the issues we're working on at [OpenHands GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues).
*   **Contributing:** Find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Project Progress

*   Check out the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the maintainer's meeting at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

OpenHands is built with the help of many contributors and other open source projects. See [CREDITS.md](./CREDITS.md) for details.

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