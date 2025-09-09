<!-- Improved README.md -->

<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  </a>
  <h1>OpenHands: Code Smarter, Not Harder with AI-Powered Software Development</h1>
  <p>
    OpenHands empowers developers with AI agents capable of automating code modification, web browsing, API calls, and more. <a href="https://github.com/All-Hands-AI/OpenHands">Explore the project on GitHub</a>.
  </p>
</div>

<div align="center">
  <!-- Badges -->
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers">
    <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
  </a>
  <br/>
  <a href="https://dub.sh/openhands">
    <img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community">
  </a>
  <a href="https://discord.gg/ESHStjSjD4">
    <img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md">
    <img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits">
  </a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started">
    <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation">
  </a>
  <a href="https://arxiv.org/abs/2407.16741">
    <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv">
  </a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
    <img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score">
  </a>

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

## Key Features of OpenHands

*   **AI-Powered Agents:** Utilize intelligent agents to automate various software development tasks.
*   **Code Modification:**  Modify, generate and refactor code with ease.
*   **Web Browsing & API Integration:** Empowered agents that can browse the web and interact with APIs.
*   **Stack Overflow Integration:**  Quickly access and implement code snippets from Stack Overflow.
*   **Flexible Deployment:** Deploy and use OpenHands via Cloud, CLI, or Docker.
*   **Community Driven:** Open source and driven by an active community, fostering innovation and collaboration.
*   **Comprehensive Documentation:** Get started quickly with detailed documentation and guides.

## Getting Started with OpenHands

### OpenHands Cloud

The easiest way to start is with [OpenHands Cloud](https://app.all-hands.dev), which includes \$20 in free credits for new users.

### Running OpenHands Locally

#### Option 1: CLI Launcher (Recommended)

Use the CLI launcher with [uv](https://docs.astral.sh/uv/) for the best experience, including better isolation and required MCP server support.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands**:
```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```
You'll find OpenHands running at [http://localhost:3000](http://localhost:3000) (for GUI mode)!

#### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

You can also run OpenHands directly with Docker:

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

</details>

> **Note:** If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) to secure your deployment by restricting network binding and implementing additional security measures.

### Initial Setup

After launching the application, select your preferred LLM provider and add your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but several [other options](https://docs.all-hands.dev/usage/llms) are available.

Refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and more information.

## More Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstations and is not intended for multi-tenant environments due to the lack of built-in authentication, isolation, and scalability.
>
> Explore the [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud) if you're interested in a multi-tenant environment.

Explore various running options: [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem),  use the [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode), operate in a [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode), or [integrate with a GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action).

For more information and setup instructions, see [Running OpenHands](https://docs.all-hands.dev/usage/installation).

## Development & Contributing

For those interested in modifying the source code, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Having issues? The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## Documentation

Consult the [documentation](https://docs.all-hands.dev/usage/getting-started) for detailed information, LLM provider tips, troubleshooting, and advanced configuration options.

## Join the Community

OpenHands is a community-driven project, welcoming contributions and discussions.  Join us on:

*   [Slack](https://dub.sh/openhands) - Discuss research, architecture, and future development.
*   [Discord](https://discord.gg/ESHStjSjD4) - A community-run server for general discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Contribute ideas and check out the latest development.

See more about the community in [COMMUNITY.md](./COMMUNITY.md) or find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Progress

Stay updated on the project roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License (excluding the `enterprise/` folder). See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is built by a large community of contributors. See [CREDITS.md](./CREDITS.md) for a list of used open-source projects and licenses.

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
```