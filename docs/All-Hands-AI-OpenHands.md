<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Empowering AI-Driven Software Development</h1>
</div>

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
<br/>
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://dub.sh/openhands)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
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

OpenHands is a cutting-edge AI platform that transforms software development by enabling intelligent agents to automate tasks and boost productivity.

Key Features:

*   **AI-Powered Development:** Leverage AI agents to automate coding, debugging, and more.
*   **Code Modification & Execution:** Modify code, run commands, and call APIs with ease.
*   **Web Browsing & Information Retrieval:** Access and utilize information from the web for informed decision-making.
*   **Community & Support:** Join our vibrant Slack and Discord communities for collaboration and support.
*   **Open Source & Extensible:**  Built on open source and designed for customization and expansion.

Learn more and get started with OpenHands at [the official repository](https://github.com/All-Hands-AI/OpenHands) and [docs.all-hands.dev](https://docs.all-hands.dev).

> [!IMPORTANT]
> Interested in using OpenHands for your work? Join our Design Partner program by filling out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) to gain early access to commercial features and contribute to our product roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started with OpenHands

The easiest way to get started is with OpenHands Cloud:

## ☁️ OpenHands Cloud

Get started with OpenHands quickly using [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

## 💻 Running OpenHands Locally

You can also run OpenHands locally using either the CLI launcher or Docker.

### Option 1: CLI Launcher (Recommended)

The CLI launcher provides isolation and is required for OpenHands' default MCP servers.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

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
> Secure your deployment on public networks by following the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Configuration

1.  **Choose an LLM provider:**  Anthropic's Claude Sonnet 4 (`anthropic/claude-sonnet-4-20250514`) is recommended.
2.  **Add your API key:**  Follow the prompts to add your key.

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for details.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user local workstations and is not intended for multi-tenant deployments.

Explore additional options:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Interact via the [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Run in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Use a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action).

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more.
For development, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).
Troubleshooting help is available in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

For detailed information on OpenHands, including LLM provider setups, troubleshooting, and advanced configuration, consult our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started).

## 🤝 Join the Community

OpenHands thrives on community contributions. Connect with us on:

*   [Slack](https://dub.sh/openhands): Research, architecture, and future development discussions.
*   [Discord](https://discord.gg/ESHStjSjD4): General discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues): Contribute ideas and track progress.

See [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

## 📈 Progress and Roadmap

Stay updated on the project's progress via the monthly OpenHands roadmap:  [Project Roadmap](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is licensed under the MIT License, except for the `enterprise/` folder.  See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

We are grateful for the contributions of many individuals and the use of other open-source projects.  See our [CREDITS.md](./CREDITS.md) file for a list of project acknowledgements and licenses.

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