[![OpenHands Logo](docs/static/img/logo.png)](https://github.com/All-Hands-AI/OpenHands)

# OpenHands: AI-Powered Software Development Agents

**OpenHands empowers you to build more, code less, by harnessing the power of AI for software development.**

[View on GitHub](https://github.com/All-Hands-AI/OpenHands) | [Join our Slack](https://dub.sh/openhands) | [Join our Discord](https://discord.gg/ESHStjSjD4) | [Documentation](https://docs.all-hands.dev/usage/getting-started)

**Key Features:**

*   ✅ **AI-Powered Code Editing:**  Automate code modifications, debugging, and refactoring tasks.
*   ✅ **Web Browsing & API Integration:** Access information online and integrate with external APIs for comprehensive development capabilities.
*   ✅ **Local & Cloud Deployment:** Run OpenHands locally or utilize the cloud platform for easy access and collaboration.
*   ✅ **Community Driven:** Benefit from a vibrant community, actively contributing to the project's growth and improvement.
*   ✅ **Open Source:** Benefit from a permissive MIT license, allowing for flexible use and modification.

## What is OpenHands?

OpenHands is a platform for building AI software developers, allowing them to modify code, run commands, browse the web, and call APIs.  Think of it as an AI-powered assistant that can perform tasks traditionally done by human developers, including the ability to copy code snippets.

## Getting Started

The easiest way to get started with OpenHands is through [OpenHands Cloud](https://app.all-hands.dev), offering new users $20 in free credits.

### Running OpenHands Locally

Choose your preferred method for local setup:

1.  **CLI Launcher (Recommended):** Uses `uv` for isolated environments.
    *   **Install `uv`**: Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
    *   **Launch OpenHands:**
        ```bash
        uvx --python 3.12 --from openhands-ai openhands serve
        # or
        uvx --python 3.12 --from openhands-ai openhands
        ```
        Access the GUI at [http://localhost:3000](http://localhost:3000).
2.  **Docker:**
    *   Pull the image:
        ```bash
        docker pull docker.all-hands.dev/all-hands-ai/runtime:0.56-nikolaik
        ```
    *   Run the container (see details in the original README).
        > **Important:**  For production use, review the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### LLM Setup

After launching, choose your LLM provider and add your API key. Anthropic's Claude Sonnet 4 (`anthropic/claude-sonnet-4-20250514`) works well.  See the [LLM options](https://docs.all-hands.dev/usage/llms).
For more details, refer to the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

## Other Deployment Options & Advanced Features

OpenHands offers flexibility:

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Interact through a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Run in a [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode).
*   Use a [GitHub action](https://docs.all-hands.dev/usage/how-to/github-action).

## Join the OpenHands Community

Connect with the OpenHands community and contribute:

*   **Slack:** [Join our Slack workspace](https://dub.sh/openhands) - for discussions about research, architecture, and development.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) -  for general discussion, questions, and feedback.
*   **GitHub Issues:** [Read or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) -  to see active development or propose ideas.

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Progress

Stay updated with the [OpenHands monthly roadmap](https://github.com/orgs/All-Hands-AI/projects/1)

## License and Acknowledgements

*   **License:**  MIT License ([LICENSE](./LICENSE)).
*   **Acknowledgements:**  See [CREDITS.md](./CREDITS.md) for a list of open-source dependencies and contributions.

## Cite Us

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