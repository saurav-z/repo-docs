# OpenHands: AI-Powered Software Development - Code Less, Achieve More

[![Contributors](https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/stargazers)
[![License](https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue)](https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge)](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/ESHStjSjD4)
[![Credits](https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white)](https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md)
[![Documentation](https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge)](https://docs.all-hands.dev/usage/getting-started)
[![Paper on Arxiv](https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2407.16741)
[![Benchmark Score](https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge)](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0)

OpenHands is an open-source platform that empowers AI agents to revolutionize software development.

**[Explore the OpenHands Repository](https://github.com/All-Hands-AI/OpenHands)**

<hr>

OpenHands (formerly OpenDevin) provides a powerful platform for AI-driven software development, enabling AI agents to perform complex tasks with minimal human intervention. With OpenHands, you can significantly accelerate your development workflow and unlock new possibilities in software creation.

## Key Features

*   **AI-Powered Agents:** Leverage intelligent agents capable of modifying code, running commands, browsing the web, and calling APIs.
*   **Full Stack Development:** Agents can perform anything a human developer can, including generating, debugging and managing code.
*   **Flexible Deployment:** Run OpenHands locally, in the cloud, or integrate it into your existing development environment.
*   **Community-Driven:** Benefit from a vibrant community with open-source contributors and active support.
*   **Extensive Documentation:** Access comprehensive documentation and resources to get started and troubleshoot issues.

## Getting Started

The easiest way to experience OpenHands is through the [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

## Running OpenHands Locally

Choose your preferred method to run OpenHands:

### CLI Launcher (Recommended)

1.  **Install `uv`:** Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your platform.

2.  **Launch OpenHands:**
    ```bash
    # Launch the GUI server
    uvx --python 3.12 --from openhands-ai openhands serve

    # Or launch the CLI
    uvx --python 3.12 --from openhands-ai openhands
    ```

### Docker

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.54
```

After starting, configure OpenHands by selecting an LLM provider and providing the API key.  Anthropic's Claude Sonnet 4 (`anthropic/claude-sonnet-4-20250514`) is the recommended option, but other [LLM options](https://docs.all-hands.dev/usage/llms) are also available.

## 💡 Other Ways to Run OpenHands

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Interact via a friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Run in a scriptable headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [Run on tagged issues with a GitHub action](https://docs.all-hands.dev/usage/how-to/github-action)

## Documentation

*   Explore the comprehensive [OpenHands documentation](https://docs.all-hands.dev/usage/getting-started) for detailed guides, troubleshooting tips, and advanced configuration options.
*   Ask DeepWiki:  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/All-Hands-AI/OpenHands)

## 🤝 Join the Community

Connect with the OpenHands community through:

*   [Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Discord](https://discord.gg/ESHStjSjD4)
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues)

Learn more in [COMMUNITY.md](./COMMUNITY.md) and [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

View the OpenHands monthly roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## 🙏 Acknowledgements

OpenHands is built by a large number of contributors, and we are deeply thankful for their work. For a list of open source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

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