<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands" target="_blank">
    <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  </a>
  <h1 align="center">OpenHands: Code Less, Build More with AI-Powered Software Development</h1>
  <p><i>Transform your software development workflow with OpenHands, the AI platform that empowers developers to build and innovate faster.</i></p>
</div>

<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors" target="_blank"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA" target="_blank"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4" target="_blank"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md" target="_blank"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started" target="_blank"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741" target="_blank"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0" target="_blank"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>

  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de" target="_blank">Deutsch</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es" target="_blank">Español</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr" target="_blank">français</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja" target="_blank">日本語</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko" target="_blank">한국어</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt" target="_blank">Português</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru" target="_blank">Русский</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh" target="_blank">中文</a>

  <hr>
</div>

## What is OpenHands?

OpenHands is an open-source platform designed to revolutionize software development using AI-powered agents. This platform allows developers to automate and accelerate various coding tasks, ultimately increasing productivity.  OpenHands agents can perform tasks similar to human developers, including code modification, command execution, web browsing, API calls, and code snippet integration.

<br/>

**Key Features:**

*   **AI-Powered Agents:** Utilize AI agents to automate and assist in the software development process.
*   **Code Modification:**  Easily modify code with the assistance of AI.
*   **Command Execution:**  Run commands directly within the OpenHands environment.
*   **Web Browsing:**  Enable agents to browse the web for information and resources.
*   **API Integration:**  Call APIs to expand functionality and integrate external services.
*   **Code Snippet Integration:**  Seamlessly incorporate code snippets from sources like StackOverflow.

Explore the possibilities and get started at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) for a quick start.

> [!IMPORTANT]
> Working with OpenHands? We'd love to hear from you!  Join our Design Partner program by filling out [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform) to get early access to commercial features and provide input on our roadmap.

![App screenshot](./docs/static/img/screenshot.png)

## ☁️ OpenHands Cloud

Experience the easiest way to get started with OpenHands on [OpenHands Cloud](https://app.all-hands.dev). New users receive $20 in free credits!

## 💻 Running OpenHands Locally

OpenHands can also be run locally using Docker.

*   Refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and detailed instructions.
*   For enhanced security, especially on public networks, consult the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands-dev/all-hands-ai/openhands:0.49
```

> **Note**:  If you used OpenHands before version 0.44, migrate your conversation history by running `mv ~/.openhands-state ~/.openhands`.

Access OpenHands at [http://localhost:3000](http://localhost:3000). You will be prompted to choose an LLM provider and enter an API key upon opening the application.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but multiple options are available at [https://docs.all-hands.dev/usage/llms](https://docs.all-hands.dev/usage/llms).

## 💡 Additional Deployment Options

> [!WARNING]
> OpenHands is optimized for single-user, local workstation use and lacks built-in authentication, isolation, and scalability suitable for multi-tenant deployments.

For multi-tenant deployments, consider the source-available, commercially licensed [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud).

Explore these additional ways to run OpenHands:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Run in a [scriptable headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run on tagged issues using a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

Find more information and setup instructions in the [Running OpenHands](https://docs.all-hands.dev/usage/installation) documentation.

For modifying the source code, review [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## 🐛 Troubleshooting

Encountering issues? Consult the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) for assistance.

## 📖 Documentation

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands" target="_blank"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Get started and learn more about OpenHands with our extensive [documentation](https://docs.all-hands.dev/usage/getting-started). It includes guidance on LLM providers, troubleshooting, and advanced configuration.

## 🤝 Join the Community

OpenHands thrives on community contributions!  Connect with us through:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA): Discuss research, architecture, and development.
*   [Discord](https://discord.gg/ESHStjSjD4):  Community-run server for general discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues): Contribute ideas and track ongoing issues.

Additional details on community involvement can be found in [COMMUNITY.md](./COMMUNITY.md) and contributions in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Project Progress

Stay updated with the OpenHands monthly roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1), updated monthly following the maintainer's meeting.

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date" target="_blank">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

OpenHands is licensed under the MIT License.  See [`LICENSE`](./LICENSE) for more details.

## 🙏 Acknowledgements

OpenHands is a collaborative effort.  We are grateful for all contributions and acknowledge the open-source projects we build upon.

For a list of all open-source projects used and their licenses, please consult [CREDITS.md](./CREDITS.md).

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