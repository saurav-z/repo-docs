<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Supercharge Your Development with AI-Powered Agents</h1>
  <p><i>Write less code and accomplish more with OpenHands, the open-source platform for AI-driven software development.</i></p>
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

## Introduction

OpenHands (formerly OpenDevin) is an open-source platform designed to revolutionize software development by utilizing AI agents. These agents can perform tasks like human developers: modifying code, running commands, browsing the web, calling APIs, and more. Reduce your coding time and boost your productivity with the power of AI.  [Visit the OpenHands repository on GitHub](https://github.com/All-Hands-AI/OpenHands) to get started!

## Key Features

*   **AI-Powered Agents:** Leverage intelligent agents to automate and streamline your development workflow.
*   **Code Modification & Execution:**  Modify code, run commands, and execute tasks directly within the platform.
*   **Web Browsing & API Integration:** Access information and integrate with external services through web browsing and API calls.
*   **Stack Overflow Integration:**  Seamlessly incorporate code snippets from Stack Overflow to accelerate development.
*   **Flexible Deployment:**  Run OpenHands on the cloud with [OpenHands Cloud](https://app.all-hands.dev), or locally with Docker.
*   **Open Source & Community Driven:** Benefit from a community-driven project with ongoing development and support.

## Getting Started

The easiest way to experience OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits. For local deployment, follow the instructions below:

### Run OpenHands Locally with Docker

1.  **Prerequisites:** Ensure you have Docker installed on your system.
2.  **Pull the Docker Image:**

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.48-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.48
```
3.  **Access OpenHands:** Open your web browser and navigate to [http://localhost:3000](http://localhost:3000).
4.  **Configure LLM:**  Choose your preferred LLM provider and add your API key. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.

### Other Run Options:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless Mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [CLI Mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Github Action](https://docs.all-hands.dev/usage/how-to/github-action)

## Important Notes

> [!WARNING]
> OpenHands is designed for single-user local deployments and is not suitable for multi-tenant environments.

## ☁️ OpenHands Cloud

Get started easily with [OpenHands Cloud](https://app.all-hands.dev), which comes with $20 in free credits for new users.

## Documentation

For detailed information, tutorials, and troubleshooting tips, explore the [OpenHands documentation](https://docs.all-hands.dev/usage/getting-started).

## Community

Join the OpenHands community and collaborate on the future of AI-driven development:

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   [Contribute on GitHub](https://github.com/All-Hands-AI/OpenHands/issues)

## Development

For contributions and source code modifications, please see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md)

## Roadmap

The monthly OpenHands roadmap is available [here](https://github.com/orgs/All-Hands-AI/projects/1).

## License

OpenHands is released under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is built by a large number of contributors. We are deeply thankful for their work.

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