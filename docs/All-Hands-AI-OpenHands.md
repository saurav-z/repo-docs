<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Supercharge Your Software Development with AI</h1>
  <p><b>Automate coding tasks, boost productivity, and build software faster with OpenHands.</b></p>
</div>

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
<br/>
[![Join Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://dub.sh/openhands)
[![Join Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Project Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
<br/>
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
<a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>

<hr>

**OpenHands** is an open-source platform empowering AI-driven software development, allowing developers to build software more efficiently. Visit the [OpenHands GitHub Repository](https://github.com/All-Hands-AI/OpenHands) for the source code and more information.

## Key Features

*   **AI-Powered Agents:** Automate coding tasks with intelligent agents capable of modifying code, running commands, browsing the web, and more.
*   **Versatile Capabilities:** OpenHands agents can perform a wide range of actions, including calling APIs and even copying code snippets from StackOverflow.
*   **Flexible Deployment:** Run OpenHands locally, in the cloud, or integrate it with your existing workflows.
*   **Community-Driven:** Benefit from a vibrant and supportive community through Slack, Discord, and GitHub.

## Getting Started with OpenHands

The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

![OpenHands Screenshot](./docs/static/img/screenshot.png)

### Running OpenHands Locally

Choose your preferred method for running OpenHands locally:

#### 1.  CLI Launcher (Recommended)

   *   **Install uv:** Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your platform.
   *   **Launch OpenHands:**

     ```bash
     # Launch the GUI server
     uvx --python 3.12 --from openhands-ai openhands serve

     # Or launch the CLI
     uvx --python 3.12 --from openhands-ai openhands
     ```
     You can access the GUI at [http://localhost:3000](http://localhost:3000).

#### 2. Docker

   <details>
   <summary>Click to expand Docker command</summary>

   ```bash
   docker pull docker.all-hands.dev/all-hands-ai/runtime:0.56-nikolaik

   docker run -it --rm --pull=always \
       -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.56-nikolaik \
       -e LOG_ALL_EVENTS=true \
       -v /var/run/docker.sock:/var/run/docker.sock \
       -v ~/.openhands:/.openhands \
       -p 3000:3000 \
       --add-host host.docker.internal:host-gateway \
       --name openhands-app \
       docker.all-hands.dev/all-hands-ai/openhands:0.56
   ```

   </details>

   > **Note:** If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands`.

   > [!WARNING]
   > For a secure deployment on a public network, see our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Configuration

1.  **LLM Provider:** Choose your preferred LLM provider. [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.
2.  **API Key:** Add your API key.
3.  **Explore Options:**  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and more information.

## Other Ways to Run OpenHands

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Use it with a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

Visit the [Running OpenHands](https://docs.all-hands.dev/usage/installation) page for setup instructions.

## For Developers

*   Modify the source code: Check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).
*   Troubleshooting: The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help resolve any issues.

## Documentation

Learn more about OpenHands with comprehensive resources, including how to use different LLM providers, troubleshooting guides, and advanced configuration options. [Check out the documentation](https://docs.all-hands.dev/usage/getting-started).

## Join the Community

Connect with the OpenHands community on these platforms:

*   [Join our Slack workspace](https://dub.sh/openhands) - Discuss research, architecture, and future development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - Ask questions, provide feedback, and engage in general discussion.
*   [Read or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Explore ongoing issues and share your ideas.

For more community information, see [COMMUNITY.md](./COMMUNITY.md) and for contributing guidelines, see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Progress

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the MIT License, except for the `enterprise/` folder. See [`LICENSE`](./LICENSE) for details.

## Acknowledgements

OpenHands is a community-driven project. See our [CREDITS.md](./CREDITS.md) for a list of open source projects and licenses used in OpenHands.

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