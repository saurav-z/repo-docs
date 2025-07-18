<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1 align="center">OpenHands: Your AI-Powered Software Development Copilot</h1>
</div>

<div align="center">
  <!-- Badges for project info -->
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <!-- Community links -->
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <!-- Documentation & Evaluation links -->
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>
  <br>
  <!-- Translation links -->
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

OpenHands is an open-source platform that empowers developers with AI-driven software development agents, allowing you to write less code and accomplish more.  [Check out the original repo](https://github.com/All-Hands-AI/OpenHands).

## Key Features

*   **AI-Powered Code Generation & Modification:** OpenHands agents can modify code, run commands, and browse the web to automate tasks.
*   **Web Browsing & API Interaction:**  Agents can browse the internet and interact with APIs, expanding the scope of what can be automated.
*   **Stack Overflow Integration:**  OpenHands agents can access and utilize code snippets from Stack Overflow, speeding up development.
*   **Cloud and Local Deployment:** Deploy and run OpenHands on the cloud via the OpenHands Cloud or locally using Docker.
*   **Multiple Integration Options:** Integrate OpenHands with your workflow using a friendly CLI, headless mode, and Github actions.

![OpenHands Application Screenshot](./docs/static/img/screenshot.png)

## Get Started

### OpenHands Cloud

The easiest way to start using OpenHands is with [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

### Run Locally with Docker

You can also run OpenHands on your local machine using Docker.  Follow these steps:

1.  **Pull the Docker Image:**
    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.49-nikolaik
    ```
2.  **Run the Container:**
    ```bash
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
3.  **Access the Application:** Open your web browser and navigate to [http://localhost:3000](http://localhost:3000).

   **Note:** If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

4.  **Configure your LLM:**  Select your preferred LLM provider and add your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended, but many [other options](https://docs.all-hands.dev/usage/llms) are supported.

> [!WARNING]
> **Important:** OpenHands is designed for single-user, local development. It is not intended for multi-tenant deployments.

####  Hardened Docker Installation

To secure your deployment by restricting network binding and implementing additional security measures, see our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

## Other Ways to Run OpenHands

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Run OpenHands in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Interact with OpenHands via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run it on tagged issues with [a github action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more detailed instructions.

## Documentation & Support

### Documentation

For comprehensive information, usage tips, and configuration options, explore the [OpenHands documentation](https://docs.all-hands.dev/usage/getting-started).

### DeepWiki

  <a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

### Troubleshooting

If you encounter issues, the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## Community & Contributing

OpenHands thrives on community contributions.  Join us on:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA):  Discuss research, architecture, and future development.
*   [Discord](https://discord.gg/ESHStjSjD4): General discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues): Report issues, share ideas, and track development.

Find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Progress & Roadmap

Track the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## Acknowledgements

OpenHands is built by a collaborative community and utilizes various open-source projects.  See [CREDITS.md](./CREDITS.md) for a complete list.

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