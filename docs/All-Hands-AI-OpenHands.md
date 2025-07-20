# OpenHands: AI-Powered Software Development Agent

OpenHands revolutionizes software development with AI, allowing you to write less code and achieve more, all while automating common tasks. **Check out the original repository on GitHub: [All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands)**

<div align="center">
  <img src="./docs/static/img/logo.png" alt="Logo" width="200">
</div>

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![MIT License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
<br/>
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Join Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Project Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
<br/>
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
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

<hr>

## Key Features

*   **AI-Powered Development:** Leverage AI agents to perform tasks like code modification, running commands, web browsing, and API calls.
*   **Versatile Capabilities:** OpenHands agents are designed to mimic human developers, including copying code snippets from Stack Overflow and more.
*   **Easy to Get Started:** Start with the cloud version or easily run OpenHands locally using Docker.
*   **Open Source & Community Driven:** Join a vibrant community and contribute to the project's ongoing development.

## Get Started

Explore OpenHands through the following resources:

*   **Documentation:** Comprehensive guides and tutorials can be found at [docs.all-hands.dev](https://docs.all-hands.dev).
*   **Cloud Access:** Easily start with OpenHands Cloud, with free credits for new users at [app.all-hands.dev](https://app.all-hands.dev).

> [!IMPORTANT]
> If you use OpenHands for work, join our Design Partner program by filling out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) to get early access to commercial features and help shape our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## Running OpenHands Locally (Docker)

Quickly set up OpenHands locally using Docker:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands-ai/openhands:0.49
```

Access OpenHands at [http://localhost:3000](http://localhost:3000).

## More Ways to Run OpenHands

*   **Connect to Filesystem:**  Connect OpenHands to your local file system for easy project access.
*   **Headless Mode:** Run OpenHands in a scriptable, headless mode for automation.
*   **CLI Mode:** Interact with OpenHands through a user-friendly command-line interface.
*   **GitHub Action:** Integrate OpenHands into your workflow using a GitHub action.

Refer to [Running OpenHands](https://docs.all-hands.dev/usage/installation) for detailed setup instructions.

## Documentation and Community

*   **Documentation:** Comprehensive guides and tutorials are available at [docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started).
*   **DeepWiki Documentation:** Get auto-generated documentation with [Ask DeepWiki](https://deepwiki.com/All-Hands-AI/OpenHands)
*   **Join the Community:**
    *   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
    *   [Discord](https://discord.gg/ESHStjSjD4)
    *   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues)
*   **Community Guidelines:** Find more community details in [COMMUNITY.md](./COMMUNITY.md) and contribution guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Progress

*   Track the project's progress and upcoming features via the [OpenHands monthly roadmap](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the [MIT License](./LICENSE).

## Acknowledgements

Special thanks to all contributors and the open-source projects that OpenHands is built upon. See [CREDITS.md](./CREDITS.md) for details.

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