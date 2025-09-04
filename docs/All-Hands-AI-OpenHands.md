<!-- Improved README - Optimized for SEO -->
<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Unleash AI-Powered Software Development</h1>
  <p><em>Revolutionize software development with AI agents that write, run, and debug code.</em></p>
</div>

<div align="center">
  <!-- Badges for project information -->
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <!-- Community and resource links -->
  <a href="https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://discord.gg/ESHStjSjD4"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>

  <!-- Language Translations - Keeping these for automatic updates -->
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

OpenHands is a cutting-edge platform that empowers software development through the use of advanced AI agents, previously known as OpenDevin.  These intelligent agents automate and streamline tasks, enabling developers to code more efficiently.  

**[Explore the OpenHands Repository on GitHub](https://github.com/All-Hands-AI/OpenHands)**

## Key Features:

*   **AI-Powered Development:** Leverage AI agents to perform tasks like code modification, command execution, web browsing, and API calls.
*   **Simplified Workflow:**  Agents can copy code snippets from StackOverflow, speeding up development.
*   **Flexible Deployment:** Run OpenHands locally, in the cloud, or integrate it into your development workflow.
*   **Community Driven:** Benefit from an active and supportive community via Slack and Discord.
*   **Open Source:**  OpenHands is licensed under the MIT License and welcomes community contributions.

## Getting Started

The easiest way to get started is to create an account on [OpenHands Cloud](https://app.all-hands.dev) and receive $20 in free credits.

If you prefer to run OpenHands locally, here's how:

### Option 1: CLI Launcher (Recommended)

Use the CLI launcher with [uv](https://docs.astral.sh/uv/) for a clean and isolated environment.

**Install uv:**

Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your platform.

**Launch OpenHands:**

```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

Access the GUI at [http://localhost:3000](http://localhost:3000).

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.55-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.55-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.55
```

</details>

> **Note:** Migrate your conversation history from older versions.

> [!WARNING]
> Secure your Docker deployment.  See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

**First time users:**  Choose an LLM provider and add your API key.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) works best.  See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for more.

## Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is meant to be run locally by a single user.  See [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud) for multi-tenant environments.

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   Run in a [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   Use a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action)

## Documentation & Resources

*   **[Documentation](https://docs.all-hands.dev/usage/getting-started):**  Find detailed guides, troubleshooting tips, and advanced configuration options.
*   **[Development Guide](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md):** Learn how to contribute to the OpenHands source code.
*   **[Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting):** Get help resolving any issues you encounter.

## Join the OpenHands Community

Connect with other developers and contribute to the project:

*   **[Slack](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA):** Discuss research, architecture, and future development.
*   **[Discord](https://discord.gg/ESHStjSjD4):** Get general support, ask questions, and provide feedback.
*   **[GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues):**  Report issues and contribute to feature development.

Find more about community participation in [COMMUNITY.md](./COMMUNITY.md) and contributing details in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Project Progress

*   **[Roadmap](https://github.com/orgs/All-Hands-AI/projects/1):** Stay updated on the monthly roadmap.

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

*   Distributed under the [MIT License](./LICENSE).

## Acknowledgements

*   Special thanks to our contributors and the open-source projects upon which OpenHands is built.
*   See our [CREDITS.md](./CREDITS.md) file for a complete list.

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

Key improvements and SEO optimization:

*   **Clear Title and Hook:**  The title is more descriptive and includes relevant keywords ("AI-Powered Software Development"). The opening sentence immediately grabs attention.
*   **Keyword Optimization:**  Includes keywords like "AI agents," "software development," "automate," "code," etc., throughout the text.
*   **Structured Content:**  Uses clear headings, bullet points, and concise paragraphs for readability and SEO benefits.
*   **Concise Summarization:** The text is more concise and avoids unnecessary phrases.
*   **Stronger Calls to Action:** Encourages users to explore the repository, join the community, and use the cloud platform.
*   **Comprehensive Documentation:** Provides links to the documentation and other resources.
*   **SEO-Friendly Alt Text:** Uses descriptive alt text for all images.
*   **Reduced Redundancy:** Removed redundant information.
*   **Cleaned Up Instructions:** Made the installation instructions clearer and more direct.
*   **Focus on Benefits:** Highlights the benefits of using OpenHands.