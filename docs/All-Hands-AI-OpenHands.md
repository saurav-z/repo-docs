<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Your AI-Powered Software Development Copilot</h1>
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

<p align="center">
    <a href="https://github.com/All-Hands-AI/OpenHands">
        <img src="https://img.shields.io/badge/View%20on%20GitHub-OpenHands-blue?style=for-the-badge&logo=github" alt="View on GitHub">
    </a>
</p>

**OpenHands empowers developers with AI-driven agents that can write, debug, and deploy code, significantly accelerating the software development lifecycle.**

## Key Features

*   **AI-Powered Agents:** OpenHands utilizes AI agents that can perform tasks like code modification, command execution, web browsing, API calls, and more.
*   **Cloud & Local Deployment:** Easily get started with [OpenHands Cloud](https://app.all-hands.dev), or run it locally for enhanced flexibility.
*   **Flexible LLM Integration:** Supports a variety of LLM providers, with Anthropic's Claude Sonnet 4 recommended for optimal performance.
*   **Extensive Documentation:** Comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started) to help you get started and explore advanced features.
*   **Community Driven:**  Join the vibrant OpenHands community through [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA), [Discord](https://discord.gg/ESHStjSjD4), and [GitHub](https://github.com/All-Hands-AI/OpenHands/issues).

## Get Started

### OpenHands Cloud

The simplest way to try OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

### Running OpenHands Locally

Choose your preferred installation method:

#### 1.  CLI Launcher (Recommended)

Install [uv](https://docs.astral.sh/uv/) and launch OpenHands:

```bash
# Install uv (if you haven't already) - follow the instructions on the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

You can access the GUI at [http://localhost:3000](http://localhost:3000).

#### 2. Docker

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.55-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.55-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.55
```

> **Note**: If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

> [!WARNING]
> For Docker on public networks, see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) for security.

### Configuration

1.  Open the application and select an LLM provider, then add your API key.
2.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works best, but many [other options](https://docs.all-hands.dev/usage/llms) are supported.
3.  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed system requirements and setup instructions.

## Additional Run Methods

OpenHands can be used in several different ways:

*   Connect to your local filesystem.
*   Use the command-line interface.
*   Run OpenHands in a scriptable headless mode.
*   Use a GitHub action.

Refer to [Running OpenHands](https://docs.all-hands.dev/usage/installation) for details.

## Development

To contribute or modify the source code, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## Troubleshooting

Find solutions to common problems in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## Documentation

Dive deeper into OpenHands with comprehensive documentation: [https://docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started)

## Join the Community

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for research and development discussions.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) for community support and general discussions.
*   **GitHub:** [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues) to get involved.

For more information on community engagement, see [COMMUNITY.md](./COMMUNITY.md) or [CONTRIBUTING.md](./CONTRIBUTING.md).

## Progress

*   View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1)

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is distributed under the [MIT License](./LICENSE).

## Acknowledgements

OpenHands is a community project built with contributions from many individuals and the use of other open-source projects. See [CREDITS.md](./CREDITS.md) for more details.

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