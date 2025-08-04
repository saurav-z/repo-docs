<!-- Improved README.md -->

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: The AI-Powered Software Development Agent That Writes Code For You</h1>
  <p><em>Code Less, Make More with OpenHands.</em></p>
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
  </a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
  </a>
  <br/>
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA">
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
    <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation">
  </a>
  <a href="https://arxiv.org/abs/2407.16741">
    <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv">
  </a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
    <img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score">
  </a>
  <br/>

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

## What is OpenHands?

OpenHands (formerly OpenDevin) is a cutting-edge platform designed to empower software developers with the capabilities of AI, enabling you to code more efficiently and effectively.  Visit the [OpenHands GitHub Repo](https://github.com/All-Hands-AI/OpenHands) for the source code.

## Key Features

*   **AI-Powered Agent:** OpenHands agents can perform tasks similar to a human developer, automating code modifications, running commands, browsing the web, and more.
*   **Web Browsing & API Integration:**  Access and interact with web resources and APIs directly from within your development workflow.
*   **Code Snippet Integration:**  Seamlessly incorporate code snippets from Stack Overflow and other sources.
*   **Cloud & Local Deployment:** Easily get started with OpenHands Cloud or run it locally using Docker.
*   **Open Source:**  Built on open source principles, fostering community-driven development and innovation.
*   **Extensive Documentation:**  Comprehensive documentation at [docs.all-hands.dev](https://docs.all-hands.dev) to help you get the most out of OpenHands.

![App screenshot](./docs/static/img/screenshot.png)

## Getting Started

### OpenHands Cloud

The quickest way to get started is through [OpenHands Cloud](https://app.all-hands.dev), where new users receive $20 in free credits.

### Running OpenHands Locally

You can also run OpenHands on your local system using Docker.

1.  **Prerequisites:** Ensure you have Docker installed.
2.  **Pull the Image:**

    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik
    ```

3.  **Run the Container:**

    ```bash
    docker run -it --rm --pull=always \
        -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.51-nikolaik \
        -e LOG_ALL_EVENTS=true \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v ~/.openhands:/.openhands \
        -p 3000:3000 \
        --add-host host.docker.internal:host-gateway \
        --name openhands-app \
        docker.all-hands.dev/all-hands-ai/openhands:0.51
    ```

4.  **Access OpenHands:** Open your web browser and navigate to [http://localhost:3000](http://localhost:3000).

5.  **Configure:** Choose an LLM provider and provide an API key.  Anthropic's [Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works best.  See the [LLM options](https://docs.all-hands.dev/usage/llms) documentation for more choices.

> **Note:** If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

### Additional Run Options

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [CLI mode](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

See [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more detailed installation and setup instructions.

##  Important Considerations

> [!WARNING]
> OpenHands is intended for single-user, local workstation use and is not designed for multi-tenant deployments without additional security and authentication measures.

> [!WARNING]
> For secure deployments on public networks, consult our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

## Community and Resources

*   **Documentation:**  Find comprehensive documentation at [docs.all-hands.dev](https://docs.all-hands.dev).
*   **Slack:** Join our Slack workspace for discussions on research, architecture, and development: [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA).
*   **Discord:** Join our community-run Discord server for general discussion, questions, and feedback: [Join our Discord server](https://discord.gg/ESHStjSjD4).
*   **GitHub Issues:**  Contribute ideas and track progress by viewing or posting issues: [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues)
*   **Community:** See more about the community in [COMMUNITY.md](./COMMUNITY.md) or find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Development

If you want to modify the OpenHands source code, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

## Troubleshooting

If you experience any issues, refer to the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## Project Roadmap

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the maintainer's meeting at the end of each month).

## Progress

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

OpenHands is distributed under the MIT License.  See [`LICENSE`](./LICENSE) for complete details.

## Acknowledgements

OpenHands is a community-driven project, and we appreciate all contributions!  We also leverage various open-source projects. A list of these is provided in [`CREDITS.md`](./CREDITS.md).

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