# OpenHands: Revolutionizing Software Development with AI

**OpenHands is an open-source platform that empowers AI agents to automate and accelerate software development, allowing you to code less and achieve more.** ([See original repository](https://github.com/All-Hands-AI/OpenHands))

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Join Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

<hr>

OpenHands enables AI agents to perform complex software development tasks, mirroring the capabilities of human developers.

## Key Features

*   **AI-Powered Automation:** Automate coding, debugging, testing, and deployment tasks.
*   **Code Modification:** Agents can modify existing codebases effectively.
*   **Web Browsing & API Integration:** Access information and integrate with external services.
*   **Code Snippet Access:** Leverage code snippets from resources like Stack Overflow.
*   **OpenHands Cloud:** The easiest way to get started with $20 in free credits for new users.

## Getting Started

### ☁️ OpenHands Cloud

The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which comes with $20 in free credits for new users.

### 💻 Running OpenHands Locally

Choose your preferred method:

#### Option 1: CLI Launcher (Recommended)

1.  **Install uv:** See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
2.  **Launch OpenHands:**
    ```bash
    # Launch the GUI server
    uvx --python 3.12 --from openhands-ai openhands serve

    # Or launch the CLI
    uvx --python 3.12 --from openhands-ai openhands
    ```
    Access the GUI at [http://localhost:3000](http://localhost:3000) (GUI mode)!

#### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.54
```
</details>

> **Note:** Migrate your conversation history by running `mv ~/.openhands-state ~/.openhands` if you used OpenHands before version 0.44.

> [!WARNING]
> For public networks, refer to the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) for enhanced security.

### LLM Setup

After launching, select your preferred LLM provider and enter your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many options are available [here](https://docs.all-hands.dev/usage/llms).

For more information, check out the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
>  OpenHands is designed for single-user local deployments; multi-tenant use is not recommended.

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Interact via a friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Headless mode for scripting](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

See [Running OpenHands](https://docs.all-hands.dev/usage/installation) for instructions.

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Comprehensive documentation is available at [https://docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started), covering LLM providers, troubleshooting, and advanced configuration.

## 🤝 Join the Community

Contribute and connect with the OpenHands community:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA): Discuss research, architecture, and development.
*   [Discord](https://discord.gg/ESHStjSjD4): General discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues): Explore and contribute to ongoing projects.

Find community details in [COMMUNITY.md](./COMMUNITY.md) and contributing guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License.  See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is a collaborative project; thank you to all contributors!  We also utilize other open-source projects, and we are deeply thankful for their work.

See [CREDITS.md](./CREDITS.md) for a list of used open source projects and licenses.

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