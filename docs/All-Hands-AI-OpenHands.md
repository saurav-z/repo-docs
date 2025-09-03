# OpenHands: Revolutionize Software Development with AI

> OpenHands empowers you to code less and build more by leveraging the power of AI-driven software development agents.

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
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

**OpenHands** is a cutting-edge platform enabling AI-powered software development agents. Designed to streamline the development process, OpenHands agents can perform a wide range of tasks, mirroring the capabilities of human developers.  [See the original repo](https://github.com/All-Hands-AI/OpenHands).

## Key Features

*   **Code Modification:**  Edit and refine code with AI assistance.
*   **Command Execution:** Run commands directly within the development environment.
*   **Web Browsing:** Access and gather information from the web.
*   **API Integration:** Seamlessly interact with various APIs.
*   **Code Snippet Retrieval:** Leverage code snippets from Stack Overflow and other resources.

## Getting Started

Get started with OpenHands Cloud or run it locally!

### OpenHands Cloud

The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which comes with $20 in free credits for new users.

### Running OpenHands Locally

Choose from the CLI Launcher or Docker to run OpenHands locally.

#### Option 1: CLI Launcher (Recommended)

Use the CLI launcher with [uv](https://docs.astral.sh/uv/) for easy local use.

**Install uv:**  Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

**Launch OpenHands:**
```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```
OpenHands GUI mode available at [http://localhost:3000](http://localhost:3000)!

#### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

Run OpenHands with Docker:

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

> **Note**: If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate conversation history.

> [!WARNING]
> For secure deployments on public networks, see our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Next Steps

1.  Open the application and select an LLM provider.
2.  Add your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) works best.

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for more information.

## Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use and is not suitable for multi-tenant deployments due to a lack of built-in authentication, isolation, or scalability.

Explore these options:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Use a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more information.

## Development

If you want to modify the OpenHands source code, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## Troubleshooting

Encountering issues? The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## Documentation

Find in-depth information, guides, and advanced configuration options in the [documentation](https://docs.all-hands.dev/usage/getting-started).

## Community

Join our community for discussions and support:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)

See more about the community in [COMMUNITY.md](./COMMUNITY.md) or find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is built by a large number of contributors, and every contribution is greatly appreciated! We also build upon other open source projects, and we are deeply thankful for their work.

For a list of open source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

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