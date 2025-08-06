<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: AI-Powered Software Development That Makes Coding Easier</h1>
</div>

<div align="center">
  <!-- Badges -->
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

OpenHands, formerly OpenDevin, is revolutionizing software development with its AI-powered platform, enabling developers to write less code and achieve more. Explore the [OpenHands Repository](https://github.com/All-Hands-AI/OpenHands) to get started.

## Key Features

*   **AI-Powered Agents:** OpenHands agents can perform tasks that a human developer can, including modifying code, running commands, and browsing the web.
*   **Code Modification & Execution:**  Agents can directly modify code, run commands, and integrate with web resources.
*   **Web Browsing & API Integration:** Access and leverage information from the internet and external APIs.
*   **Stack Overflow Integration:** Agents can even incorporate code snippets from Stack Overflow, streamlining development.
*   **Cloud & Local Deployment:**  Easily get started with OpenHands on [OpenHands Cloud](https://app.all-hands.dev) or run it locally using Docker.

> [!IMPORTANT]
> Interested in using OpenHands for work?  Join our Design Partner program by filling out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) to gain early access and provide input on our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started with OpenHands

### OpenHands Cloud

The easiest way to start using OpenHands is with [OpenHands Cloud](https://app.all-hands.dev), which offers $20 in free credits for new users.

### Running OpenHands Locally

You can also run OpenHands locally using Docker.  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and instructions.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
> to secure your deployment by restricting network binding and implementing additional security measures.

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands-ai/openhands:0.51
```

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

Access OpenHands at [http://localhost:3000](http://localhost:3000). You'll be prompted to select an LLM provider and enter an API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.  Explore [many other options](https://docs.all-hands.dev/usage/llms) to suit your needs.

### Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use and is not suitable for multi-tenant deployments.

Consider these alternatives:

*   [Connecting OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless Mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting.
*   [CLI Mode](https://docs.all-hands.dev/usage/how-to/cli-mode) for command-line interaction.
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action) for use on tagged issues.

Find more details and setup instructions in the [Running OpenHands](https://docs.all-hands.dev/usage/installation) documentation.

For modifying the OpenHands source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).
Troubleshooting resources are available in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## Documentation

<a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Consult our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started) to fully utilize OpenHands, including LLM provider options, troubleshooting assistance, and advanced configurations.

## Community & Contribution

OpenHands is a community-driven project.  Join us:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for research, architecture, and development discussions.
*   [Discord](https://discord.gg/ESHStjSjD4) for general discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) to contribute ideas and track progress.

More information about the community is in [COMMUNITY.md](./COMMUNITY.md) and details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the maintainer's meeting at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is distributed under the MIT License.  See [`LICENSE`](./LICENSE) for more information.

## Acknowledgements

OpenHands is built by a community of contributors. A list of open source projects and licenses used in OpenHands can be found in [CREDITS.md](./CREDITS.md).

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