# OpenHands: AI-Powered Software Development Agents (Code Less, Make More)

> **OpenHands empowers you to code less and build more by utilizing AI agents to automate software development tasks.**

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

[Deutsch](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de) |
[Español](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es) |
[français](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr) |
[日本語](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja) |
[한국어](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko) |
[Português](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt) |
[Русский](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru) |
[中文](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh)

<hr>

Welcome to OpenHands, a cutting-edge platform that leverages AI agents to transform the software development lifecycle.  OpenHands, formerly known as OpenDevin, is designed to drastically reduce coding time and effort.

<img src="./docs/static/img/screenshot.png" alt="OpenHands Screenshot" width="70%">

**Key Features:**

*   **AI-Powered Automation:** Automate code modification, command execution, web browsing, API calls, and more.
*   **Human-Like Capabilities:**  Mimics human developer actions, including using StackOverflow snippets.
*   **Cloud and Local Deployment:** Utilize OpenHands Cloud for ease of use or run it locally via Docker.
*   **Extensive Documentation:** Comprehensive documentation to get you started and troubleshoot issues.
*   **Active Community:** Join our Slack and Discord communities for support, discussions, and collaboration.

**Get Started:**

*   **OpenHands Cloud:** The easiest way to start with OpenHands; sign up and get \$20 in free credits [here](https://app.all-hands.dev).
*   **Local Installation (Docker):** Instructions for running OpenHands on your local machine [here](https://docs.all-hands.dev/usage/installation).

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
    docker.all-hands.dev/all-hands-ai/openhands:0.48
```

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

Access OpenHands at [http://localhost:3000](http://localhost:3000).  For optimal performance, use [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) with an API key. More LLM options are available [here](https://docs.all-hands.dev/usage/llms).

### Additional Ways to Use OpenHands

Explore different ways to integrate and use OpenHands based on your needs.

*   **Filesystem Connection:** Connect to your local filesystem [here](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   **Headless Mode:** Run OpenHands in a scriptable headless mode [here](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   **CLI Mode:** Interact with OpenHands using a friendly CLI [here](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   **GitHub Action:** Run OpenHands on tagged issues [here](https://docs.all-hands.dev/usage/how-to/github-action).

### Important Considerations

> **Warning:** OpenHands is best suited for single-user local development; it is not designed for multi-tenant environments without built-in authentication, isolation, or scalability.

For multi-tenant deployments, consider the [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud).

## 📚 Documentation

Comprehensive documentation is available at [docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started), including:
*   LLM provider setup.
*   Troubleshooting guides.
*   Advanced configuration options.

## 🤝 Community and Contribution

Join our vibrant community to collaborate and contribute:

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for research, architecture, and development discussions.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) for general discussion, questions, and feedback.
*   **GitHub Issues:** [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues) for issues and feature requests.

Learn more about contributing in [CONTRIBUTING.md](./CONTRIBUTING.md) and find details on the community in [COMMUNITY.md](./COMMUNITY.md).

## 📈 Project Progress

Stay updated with our monthly roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the [MIT License](./LICENSE).

## 🙏 Acknowledgements

OpenHands is built by a large number of contributors, and every contribution is greatly appreciated! We also build upon other open source projects, and we are deeply thankful for their work.

See [CREDITS.md](./CREDITS.md) for a list of open source projects and licenses used in OpenHands.

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

[Back to Top](#readme-top)