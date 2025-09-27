# OpenHands: The AI-Powered Software Development Platform

OpenHands empowers you to write less code and accomplish more, offering AI-driven software development capabilities. ([See the original repository](https://github.com/All-Hands-AI/OpenHands))

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![MIT License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://dub.sh/openhands)
[![Join Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Project Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
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

OpenHands (formerly OpenDevin) is a groundbreaking platform for AI-powered software development, designed to accelerate your workflow and boost productivity.

**Key Features:**

*   **AI-Driven Development:** Utilize AI agents to perform tasks just like a human developer.
*   **Code Modification:** Easily modify existing code.
*   **Command Execution:** Run commands and automate tasks.
*   **Web Browsing and API Calls:** Access web resources and interact with APIs.
*   **Code Snippet Integration:** Effortlessly incorporate code snippets from sources like StackOverflow.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) to get started.

> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## OpenHands Cloud

Get started effortlessly with OpenHands on [OpenHands Cloud](https://app.all-hands.dev), which offers $20 in free credits for new users.

## Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The recommended method is using the CLI launcher with [uv](https://docs.astral.sh/uv/) for optimized performance and isolation.

**Install uv**:

Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for platform-specific instructions.

**Launch OpenHands**:

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access the GUI at [http://localhost:3000](http://localhost:3000).

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

Run OpenHands with Docker:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.57
```

</details>

> **Note**: If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

> [!WARNING]
> On a public network? Review our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) to secure your deployment.

### Getting Started

Upon opening the application, select an LLM provider and enter your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but multiple [LLM options](https://docs.all-hands.dev/usage/llms) are available.

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed system requirements and information.

## Additional Run Options

> [!WARNING]
> OpenHands is intended for single-user, local workstation deployments.  Multi-tenant environments are not supported.

Explore options such as:

*   [Connecting to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for setup instructions.

For source code modifications, refer to [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).  For troubleshooting, see the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## Documentation

Comprehensive documentation is available at [https://docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started) providing resources on LLM providers, troubleshooting, and advanced configurations.

## Join the Community

Join the OpenHands community!

*   [Slack workspace](https://dub.sh/openhands) - Discuss research, architecture, and development.
*   [Discord server](https://discord.gg/ESHStjSjD4) - General discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Contribute your ideas and review current issues.

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.

## Progress & Roadmap

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License (except the `enterprise/` folder).  See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is a community-driven project; thank you to all contributors!  We're grateful to the open-source projects upon which we build.

See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses.

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