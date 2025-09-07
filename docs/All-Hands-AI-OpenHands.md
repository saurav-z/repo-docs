<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1 align="center">OpenHands: Empowering AI-Driven Software Development</h1>
</div>

<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://dub.sh/openhands"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
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

OpenHands is an open-source platform that uses AI to automate software development tasks, allowing you to write less code and achieve more.

## Key Features of OpenHands

*   **AI-Powered Development:** Leverage AI agents to automate code modification, command execution, web browsing, and API calls.
*   **Code Snippet Integration:** Effortlessly incorporate code snippets from StackOverflow and other online resources.
*   **Local and Cloud Deployment:** Run OpenHands on your local machine or utilize the convenient OpenHands Cloud.
*   **Community-Driven:** Join a vibrant community for support, collaboration, and feature development.
*   **Comprehensive Documentation:** Access detailed documentation to guide you through setup, usage, and advanced configurations.
*   **Open Source & Customizable:**  Contribute to the project and tailor it to your specific needs.

**[Explore the OpenHands repository on GitHub](https://github.com/All-Hands-AI/OpenHands)**

## Getting Started with OpenHands

OpenHands offers flexibility in how you can get started.  Choose the method that best suits your needs:

### ☁️ OpenHands Cloud (Recommended)

The easiest way to get started is via [OpenHands Cloud](https://app.all-hands.dev), which provides new users with $20 in free credits.

### 💻 Running OpenHands Locally

You can run OpenHands on your local machine using a CLI launcher or Docker.

#### Option 1: CLI Launcher (Recommended)

The CLI launcher, combined with [uv](https://docs.astral.sh/uv/), provides a streamlined experience and better isolation.

**Install uv (if you haven't already):**

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Launch OpenHands:**

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

> **Note:** Migrate your conversation history if you are upgrading from an older version:  `mv ~/.openhands-state ~/.openhands`

> [!WARNING]
> See the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation) for securing your deployment on a public network.

### Configuration

*   **LLM Provider:**  Select your preferred Large Language Model (LLM) provider and provide your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) is a great starting point, but many other options are available.  See the [LLM options](https://docs.all-hands.dev/usage/llms) for more details.
*   **Refer to the documentation:** The [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide details system requirements and additional setup instructions.

## More Ways to Run OpenHands

> [!WARNING]
> OpenHands is primarily intended for single-user, local workstation usage; it's not designed for multi-tenant deployments.

Explore various deployment options:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting
*   Integrate with a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for further details.

## Contributing and Development

To contribute to OpenHands, consult [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## Troubleshooting

For assistance, see the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📚 Documentation

Find comprehensive documentation and usage guides at [docs.all-hands.dev/usage/getting-started](https://docs.all-hands.dev/usage/getting-started) to learn how to:

*   Use various LLM providers
*   Troubleshoot common issues
*   Configure advanced settings

## 🤝 Join the OpenHands Community

Join our collaborative community!

*   **Slack:** [Join our Slack workspace](https://dub.sh/openhands) to discuss research, architecture, and development.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) for general discussion and feedback.
*   **GitHub Issues:** [Read or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) to contribute ideas and track progress.

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Project Progress

Track the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License, with the exception of the `enterprise/` folder. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands thrives through the contributions of many individuals and relies on other open-source projects.

See our [CREDITS.md](./CREDITS.md) for a list of the open source projects used and their licenses.

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