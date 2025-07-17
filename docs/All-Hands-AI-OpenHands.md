<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: The AI-Powered Software Development Agent That Writes Code for You</h1>
  <p><i>Reduce coding time and increase productivity with OpenHands!</i></p>
  <a href="https://github.com/All-Hands-AI/OpenHands">
      <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stars">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
</div>

<div align="center">
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>
</div>

<!-- Keep these links. Translations will automatically update with the README. -->
<div align="center">
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

## What is OpenHands?

OpenHands is a cutting-edge AI platform for software development, empowering developers to automate and accelerate their coding workflows.  This open-source project allows you to leverage the power of AI to write, modify, and manage code with unprecedented ease.  **Explore the future of software development – visit the [OpenHands repository on GitHub](https://github.com/All-Hands-AI/OpenHands).**

## Key Features

*   **AI-Powered Code Generation:** Generate code snippets, functions, and entire programs with simple prompts.
*   **Code Modification:** Modify existing code based on natural language instructions.
*   **Web Browsing and API Integration:** Access information online and integrate with external APIs to enhance functionality.
*   **Local and Cloud Deployment:**  Run OpenHands on your local machine or leverage the OpenHands Cloud for ease of use.
*   **Community Driven:** Open-source and actively maintained by a vibrant community.
*   **Full Stack Capabilities:** OpenHands can handle a wide variety of development tasks, as any human developer can.

![OpenHands Interface Screenshot](./docs/static/img/screenshot.png)

## Getting Started

### OpenHands Cloud

The easiest way to start with OpenHands is by using [OpenHands Cloud](https://app.all-hands.dev). New users receive $20 in free credits.

### Local Installation (with Docker)

Follow these steps to run OpenHands locally using Docker:

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
    docker.all-hands.dev/all-hands-ai/openhands:0.49
```

OpenHands will be accessible at [http://localhost:3000](http://localhost:3000).  You'll be prompted to select an LLM provider and enter an API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works well, but many [other LLMs](https://docs.all-hands.dev/usage/llms) are supported.

> **Note:** If you used OpenHands before version 0.44, migrate your history with `mv ~/.openhands-state ~/.openhands`.

### Other Run Modes

Explore additional deployment options:

*   Connect to your [local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Run in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting.
*   Use the [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Integrate with [GitHub Actions](https://docs.all-hands.dev/usage/how-to/github-action).

See the [Running OpenHands documentation](https://docs.all-hands.dev/usage/installation) for detailed instructions.

## Documentation and Support

Comprehensive documentation is available to guide you through OpenHands:

*   [Documentation](https://docs.all-hands.dev/usage/getting-started): Learn about using different LLM providers, troubleshooting, and advanced configuration.
*   [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting): Resolve any issues you may encounter.
*   [DeepWiki Documentation](https://deepwiki.com/All-Hands-AI/OpenHands): Automatically generated documentation by DeepWiki.

## Community

OpenHands is a community-driven project.  We welcome your participation:

*   **Slack:** [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA) for discussions.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) for general discussion and feedback.
*   **GitHub Issues:** [Read or post issues](https://github.com/All-Hands-AI/OpenHands/issues) to contribute.

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Roadmap and Progress

*   See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).
*   Track project progress and growth over time with the Star History chart:

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License.  See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

Special thanks to the contributors and the open-source projects used in OpenHands. See [CREDITS.md](./CREDITS.md) for details.

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