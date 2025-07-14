# OpenHands: The AI-Powered Platform for Smarter Software Development

OpenHands empowers you to build software faster by leveraging AI agents to automate coding tasks.  [Check out the original repository](https://github.com/All-Hands-AI/OpenHands).

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

[Deutsch](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de) |
[Español](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es) |
[français](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr) |
[日本語](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja) |
[한국어](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko) |
[Português](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt) |
[Русский](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru) |
[中文](https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh)

---

## Key Features

*   **AI-Powered Agents:** Utilize AI agents to perform development tasks like code modification, web browsing, API calls, and more.
*   **Code Automation:**  Automate repetitive coding tasks and accelerate your development workflow.
*   **Web Integration:**  Agents can browse the web and access information to assist in coding.
*   **API Integration:** Seamlessly integrate with APIs to extend functionality.
*   **Stack Overflow Integration:**  Leverage code snippets from Stack Overflow directly within your development process.

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started

The easiest way to experience OpenHands is via the [OpenHands Cloud](https://app.all-hands.dev), which provides new users with $20 in free credits.

### Running OpenHands Locally

You can also run OpenHands locally using Docker:

1.  **Pull the Docker Image:**

    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik
    ```

2.  **Run the Docker Container:**

    ```bash
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

    Access OpenHands at [http://localhost:3000](http://localhost:3000).

    **Note:** If upgrading from a version prior to 0.44, consider migrating your conversation history: `mv ~/.openhands-state ~/.openhands`

3.  **Configure LLM:**
    Choose an LLM provider and enter your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.  Explore [other LLM options](https://docs.all-hands.dev/usage/llms).

## Other Deployment Options

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Run OpenHands in a headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [Interact with OpenHands via a CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Run OpenHands on tagged issues with a github action](https://docs.all-hands.dev/usage/how-to/github-action)

For detailed setup and usage instructions, see the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

## Community and Support

*   **Join our Slack workspace:**  Connect with the community for discussions on research, architecture, and future development. [Join Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   **Join our Discord server:** Participate in general discussions, ask questions, and provide feedback. [Join Discord](https://discord.gg/ESHStjSjD4)
*   **Explore GitHub Issues:** Review and contribute to ongoing projects. [Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)
*   See more about the community in [COMMUNITY.md](./COMMUNITY.md) or find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).
*   **Documentation:**  Access comprehensive documentation and usage tips at [https://docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started).  Also check out the auto-generated documentation from DeepWiki:  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

## Important Notes

> [!IMPORTANT]
>  For users using OpenHands for work,  consider joining the [Design Partner program](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) for early access to commercial features and product roadmap input.

> [!WARNING]
> OpenHands is intended for single-user, local workstation use and is not appropriate for multi-tenant deployments.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
> to secure your deployment by restricting network binding and implementing additional security measures.

## Progress

View the monthly OpenHands roadmap: [https://github.com/orgs/All-Hands-AI/projects/1](https://github.com/orgs/All-Hands-AI/projects/1)

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is built with the contributions of a large community and builds upon other open source projects.  See [CREDITS.md](./CREDITS.md) for a complete list.

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