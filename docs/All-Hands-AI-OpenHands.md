<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: AI-Powered Software Development, Simplified</h1>
  <p><em>Code less, achieve more with the power of AI.</em></p>
</div>

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
<br/>
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
<br/>
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

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

## About OpenHands

OpenHands, formerly OpenDevin, is an open-source platform that leverages AI to empower software development. Automate tasks, accelerate workflows, and focus on innovation with AI agents that can perform a wide range of actions.  Visit the [OpenHands GitHub Repository](https://github.com/All-Hands-AI/OpenHands) to explore the code and contribute.

## Key Features

*   **AI-Powered Development:** Utilize AI agents to automate coding tasks, debug code, and more.
*   **Code Modification and Execution:**  Agents can modify code, run commands, and interact with your development environment.
*   **Web Browsing and API Integration:** Agents can browse the web, access APIs, and integrate external services.
*   **Stack Overflow Integration:**  Leverage the vast knowledge of Stack Overflow directly within your development workflow.
*   **Flexible Deployment:** Run OpenHands locally with Docker or utilize the cloud-based [OpenHands Cloud](https://app.all-hands.dev).

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started

### OpenHands Cloud

The easiest way to start using OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), which provides a user-friendly interface and comes with $20 in free credits for new users.

### Running Locally with Docker

OpenHands can also be run locally using Docker.

**Prerequisites:** Docker installed

**Installation:**

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.50-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.50-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.50
```

**Access:** Open OpenHands in your browser at [http://localhost:3000](http://localhost:3000)

**Configuration:**  When prompted, choose an LLM provider and add your API key.  Anthropic's Claude Sonnet 4 (`anthropic/claude-sonnet-4-20250514`) is recommended.  See [LLM options](https://docs.all-hands.dev/usage/llms) for alternatives.

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
> to secure your deployment by restricting network binding and implementing additional security measures.

### Other Deployment Options

OpenHands offers flexibility in how you can use it:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting
*   [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode) for command-line interaction
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action) for automated tasks

Find more setup and installation instructions at [Running OpenHands](https://docs.all-hands.dev/usage/installation).

## Documentation & Resources

*   **Comprehensive Documentation:** Explore detailed guides and tutorials at [docs.all-hands.dev](https://docs.all-hands.dev/usage/getting-started)
*   **Troubleshooting:** Find solutions to common issues in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).
*   **Development:** Learn how to modify the OpenHands source code in [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).
*   **DeepWiki:** [Autogenerated Documentation by DeepWiki](https://deepwiki.com/All-Hands-AI/OpenHands)

## Join the OpenHands Community

We welcome contributions and encourage community participation.

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) - Discuss research, architecture, and development.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) - General discussion, Q&A, and feedback.
*   **GitHub Issues:** [Read or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Explore current work and contribute ideas.
*   **Community Details:** Learn more in [COMMUNITY.md](./COMMUNITY.md) and find contribution guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Progress & Roadmap

Stay up-to-date on the project's progress and future plans:

*   **Monthly Roadmap:** View the OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is released under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is a community-driven project, and we are thankful to all contributors!  We also leverage other open source projects and are deeply grateful for their contributions.  See [CREDITS.md](./CREDITS.md) for a list of open-source dependencies and licenses.

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

## Design Partner Program

> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.