<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: AI-Powered Software Development, Simplified</h1>
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

**OpenHands empowers you to develop software faster by leveraging the power of AI to automate complex coding tasks.**  

[Go to the OpenHands GitHub Repository](https://github.com/All-Hands-AI/OpenHands)

## Key Features

*   **AI-Powered Agents:** OpenHands agents can perform a wide range of developer tasks, including code modification, command execution, web browsing, and API interactions.
*   **Code Generation and Modification:**  Automate repetitive coding tasks and accelerate development with AI-driven code generation and modification capabilities.
*   **Web Browsing and Information Retrieval:**  Access information from the web, including StackOverflow, directly within your development workflow.
*   **Open Source and Community-Driven:** Benefit from a collaborative development environment with active community support.

## Getting Started

### OpenHands Cloud

The easiest way to experience OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which provides new users with $20 in free credits.

### Running OpenHands Locally

Choose your preferred method for running OpenHands locally:

#### Option 1: CLI Launcher (Recommended)

1.  **Install uv:** Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for instructions.
2.  **Launch OpenHands:**

    ```bash
    # Launch the GUI server
    uvx --python 3.12 --from openhands-ai openhands serve

    # Or launch the CLI
    uvx --python 3.12 --from openhands-ai openhands
    ```

    Access the GUI at [http://localhost:3000](http://localhost:3000).

#### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.52-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.52-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.52
```
</details>

> **Note:** Migrate your conversation history if you used a version before 0.44: `mv ~/.openhands-state ~/.openhands`

> [!WARNING]
> For secure deployment on a public network, consult the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)

### Configuration

*   When the application launches, select your preferred LLM provider and add your API key.
*   [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many [LLM options](https://docs.all-hands.dev/usage/llms) are supported.
*   Refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and detailed information.

## Other Run Options

> [!WARNING]
> OpenHands is intended for single-user, local workstation use only.  Multi-tenant deployments are not supported.

You can also:

*   Connect to your [local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact with a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Use a [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Run with a [github action](https://docs.all-hands.dev/usage/how-to/github-action)

## Resources

*   [**Documentation:**](https://docs.all-hands.dev/usage/getting-started) Comprehensive guides and tutorials to help you get started.
*   [**Troubleshooting:**](https://docs.all-hands.dev/usage/troubleshooting)  Resolve common issues and find solutions.
*   [**Development:**](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md) Contribute to the OpenHands project and modify the source code.

## Community and Support

Connect with the OpenHands community for support, discussions, and contributions:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   [Report issues and discuss ideas on Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)

Learn more about the community in [COMMUNITY.md](./COMMUNITY.md) and contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Development Roadmap

*   [OpenHands Roadmap](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly)

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands relies on contributions from many individuals and other open-source projects.

See [CREDITS.md](./CREDITS.md) for a complete list of the open-source projects and licenses used.

## Citation

```
@inproceedings{
  wang2025openhands,
  title={OpenHands: An Open Platform for {AI} Software Developers as Generalist Agents},
  author={Xingyao Wang and Boxuan Li and Yufan Song and Frank F. Xu and Xiangru Tang and Mingchen Zhuge and Jiayi Pan and Yueqi Song and Bowen Li and Jaskirat Singh and Hoang H. Tran and Fuqiang Li and Ren Ma and Mingzhang Zheng and Bill Qian and Yanjun Shao and Niklas Muennighoff and Yizhe Zhang and Binyuan Hui and Junyang Lin and Robert Brennan and Hao Peng and Heng Ji and Graham Neubig},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=OJd3ayDDoF}
}