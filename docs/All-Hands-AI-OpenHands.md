<!-- Improved README - SEO Optimized -->

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Supercharge Your Development with AI</h1>
  <p><b>Unlock the power of AI-driven software development, coding less and achieving more.</b></p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/stargazers">
      <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stars">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
    </a>
    <br/>
    <a href="https://dub.sh/openhands">
      <img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community">
    </a>
    <a href="https://discord.gg/ESHStjSjD4">
      <img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md">
      <img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits">
    </a>
    <br/>
    <a href="https://docs.all-hands.dev/usage/getting-started">
      <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Documentation">
    </a>
    <a href="https://arxiv.org/abs/2407.16741">
      <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Arxiv Paper">
    </a>
    <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
      <img src="https://img.shields.io/badge/Benchmark%20Score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Benchmark Score">
    </a>
  </p>
  <hr>
</div>

**[View the source code on GitHub](https://github.com/All-Hands-AI/OpenHands)**

OpenHands is a cutting-edge platform for AI-powered software development, empowering you to build and innovate faster.  OpenHands agents are designed to perform complex software development tasks.

## Key Features

*   **AI-Powered Development:** Leverage the power of AI to automate tasks and accelerate your workflow.
*   **Code Modification and Execution:**  Modify existing code, run commands, and manage your projects with ease.
*   **Web Browsing and API Interaction:** Access information and integrate with external services through web browsing and API calls.
*   **Seamless Integration:** Copy code snippets from StackOverflow, integrate with your favorite tools, and get up and running quickly.
*   **Multi-Platform Support:** Run OpenHands on your local machine using the CLI Launcher or Docker.
*   **Open Source & Community Driven:**  Benefit from a vibrant community of developers, researchers, and contributors.

## Getting Started

The easiest way to try OpenHands is through OpenHands Cloud.  You'll get $20 in free credits for new users.

### OpenHands Cloud

[Sign up for OpenHands Cloud](https://app.all-hands.dev) to start using OpenHands immediately.

### Running OpenHands Locally

#### 1. CLI Launcher (Recommended)

   1.  **Install `uv`:**
       See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.
   2.  **Launch OpenHands:**
       ```bash
       # Launch the GUI server
       uvx --python 3.12 --from openhands-ai openhands serve

       # Or launch the CLI
       uvx --python 3.12 --from openhands-ai openhands
       ```
       Access the GUI at [http://localhost:3000](http://localhost:3000).

#### 2. Docker

<details>
<summary>Expand Docker Commands</summary>

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

> **Note:** If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

> [!WARNING]
> For hardened Docker installation on public networks, see the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

#### Configuration

1.  Choose an LLM provider and enter the API key.
2.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.

Visit the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for more information.

## Additional Ways to Run OpenHands

> [!WARNING]
> OpenHands is intended for single-user, local workstation use.  It lacks multi-tenant features like authentication, isolation, and scalability.

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact via the [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run in [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Integrate via a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

Consult the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for detailed instructions.  For source code modifications, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## Documentation and Support

*   **Documentation:**  Explore the comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started) for detailed usage instructions, LLM provider setup, troubleshooting, and advanced configuration.
*   **Troubleshooting:**  Refer to the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) for assistance.

## Join the Community

Connect with the OpenHands community:

*   [Join our Slack workspace](https://dub.sh/openhands) - For discussions on research, architecture, and development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - For general discussion, questions, and feedback.
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Share ideas and contribute to project development.

See [COMMUNITY.md](./COMMUNITY.md) for details. Learn about contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Progress

View the monthly roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

Distributed under the [MIT License](./LICENSE).

## Acknowledgements

OpenHands is a community effort built on the shoulders of many contributors and open-source projects.  See [CREDITS.md](./CREDITS.md) for the list of open source projects and licenses used.

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