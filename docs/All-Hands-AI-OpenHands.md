<!-- Improved README - Optimized for SEO -->
<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Build Software Faster with AI-Powered Agents</h1>
  <p><em>Revolutionize software development by leveraging AI agents to write, debug, and deploy code with ease.</em></p>
  <p>
    <a href="https://github.com/All-Hands-AI/OpenHands">
      <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stars">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License">
    </a>
  </p>
  <p>
    <a href="https://dub.sh/openhands">
      <img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join Slack">
    </a>
    <a href="https://discord.gg/ESHStjSjD4">
      <img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join Discord">
    </a>
    <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md">
      <img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits">
    </a>
  </p>
  <p>
    <a href="https://docs.all-hands.dev/usage/getting-started">
      <img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Documentation">
    </a>
    <a href="https://arxiv.org/abs/2407.16741">
      <img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="ArXiv Paper">
    </a>
    <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0">
      <img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Benchmark Score">
    </a>
  </p>
  <!-- Keep these links. Translations will automatically update with the README. -->
  <p>
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
    <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>
  </p>
  <hr>
</div>

<!-- Description -->
## About OpenHands: Your AI-Powered Software Development Partner

OpenHands (formerly OpenDevin) is a cutting-edge platform designed to empower developers with AI-driven software development agents.  These agents can perform a wide range of tasks, just like human developers, including code modification, command execution, web browsing, API interactions, and more.  Reduce coding time and accelerate your development workflow with the power of AI.

<!-- Key Features -->
## Key Features of OpenHands

*   **AI-Powered Code Generation and Modification:**  Generate, understand, and modify code with ease.
*   **Automated Task Execution:**  Run commands, manage dependencies, and automate development processes.
*   **Web Access and Integration:**  Browse the web, access APIs, and integrate with external services.
*   **Easy to Use:** User-friendly interface
*   **Community Driven:** Open source and welcoming community.
*   **Open Source & Extensible:**  Customize and extend OpenHands to fit your specific needs.

<!-- Getting Started -->
## Getting Started with OpenHands

The easiest way to get started is on [OpenHands Cloud](https://app.all-hands.dev), which offers new users $20 in free credits.

### Running OpenHands Locally

Choose the option that best suits your needs:

**1.  CLI Launcher (Recommended)**

    *   **Install uv:** Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
    *   **Launch OpenHands:**

    ```bash
    # Launch the GUI server
    uvx --python 3.12 --from openhands-ai openhands serve

    # Or launch the CLI
    uvx --python 3.12 --from openhands-ai openhands
    ```
    Access the GUI at [http://localhost:3000](http://localhost:3000).

**2.  Docker**

    *   Pull the Docker image:
    ```bash
    docker pull docker.all-hands.dev/all-hands-ai/runtime:0.56-nikolaik
    ```

    *   Run the container:
    ```bash
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

    > **Note**:  If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands`.

    > [!WARNING]
    > For secure deployment on a public network, see our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Configuration

When you open the application, you'll be prompted to select an LLM provider and add an API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works best. See [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more.

<!-- Other Ways to Run -->
## More Ways to Run OpenHands

> [!WARNING]
> OpenHands is for single-user, local workstation use only. Avoid multi-tenant deployments.

Explore these alternative methods:

*   [Connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [Interact via the CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Run in headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [Integrate with a GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

Find detailed instructions at [Running OpenHands](https://docs.all-hands.dev/usage/installation).

<!-- Development -->
## Development

To contribute to the OpenHands project, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

<!-- Troubleshooting -->
## Troubleshooting

Encountering issues?  The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

<!-- Documentation -->
## Documentation

For in-depth information, including LLM provider details, troubleshooting, and advanced configuration, explore our comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started).

<!-- Community -->
## Join the OpenHands Community

Join a vibrant community of developers:

*   [Join our Slack workspace](https://dub.sh/openhands) - Discuss research and development.
*   [Join our Discord server](https://discord.gg/ESHStjSjD4) - Get support, ask questions, and share feedback.
*   [Browse and contribute to GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues)

Learn more about community participation in [COMMUNITY.md](./COMMUNITY.md) and contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

<!-- Progress -->
## Project Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

<!-- License -->
## License

OpenHands is distributed under the MIT License (except the `enterprise/` folder). See [`LICENSE`](./LICENSE) for full details.

<!-- Acknowledgements -->
## Acknowledgements

OpenHands is built with the contributions of many developers and leverages various open-source projects. See [CREDITS.md](./CREDITS.md) for a complete list.

<!-- Citation -->
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
```

Key improvements and why:

*   **Clearer Headline and Subheadline:**  Uses SEO keywords like "AI-Powered," "Software Development," and "Faster" to attract relevant searches.  The subheadline adds a concise, value-driven description.
*   **Concise Hook:** Provides a single sentence describing the core value proposition of the project.
*   **Strategic Keyword Use:**  Incorporates relevant keywords throughout the README.
*   **Organized Structure with Headings:**  Uses clear headings to improve readability and make it easier for users and search engines to understand the content.
*   **Bulleted Key Features:**  Highlights the core functionalities of the project in an easy-to-scan format.
*   **Direct Links:**  Includes clear, direct links to documentation, getting started guides, and community resources, improving user engagement and SEO.
*   **Community and Contribution Focus:**  Emphasizes community involvement and how to contribute.
*   **Code Blocks and Formatting:** Uses code blocks and markdown formatting to make the instructions easy to follow.
*   **SEO-Friendly Alt Tags:** Uses descriptive alt tags for all images.
*   **Simplified Docker Instructions:** Streamlined and clarified the docker instructions.
*   **Included warnings, and getting started steps**: Adding more clarity to getting started.
*   **Clearer Project Overview**: A section dedicated to a project overview.
*   **Removed Redundant Information**: Removed redundant sections to make the document more streamlined.

This improved README is more informative, user-friendly, and SEO-optimized, helping users find and understand the project more effectively.