<!-- Improved README - SEO Optimized -->

<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <h1>OpenHands: Build Software Faster with AI</h1>
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

**OpenHands is an AI-powered platform enabling developers to code faster and more efficiently.**  

OpenHands, formerly known as OpenDevin, provides AI-driven agents that can perform a wide range of software development tasks, helping you to spend less time coding and more time building.  

**Key Features:**

*   **AI-Powered Development:** Utilize AI agents to automate tasks.
*   **Code Modification and Generation:** Modify existing code or generate new code.
*   **Web Browsing & API Integration:** Interact with the web and call APIs directly.
*   **StackOverflow Integration:**  Leverage code snippets from StackOverflow.
*   **Community & Support:** Active Slack and Discord communities.
*   **Comprehensive Documentation:** Detailed documentation to get you started.

## Getting Started

The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev), which includes $20 in free credits for new users.

You can also run OpenHands locally.

## Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

Use the CLI launcher with [uv](https://docs.astral.sh/uv/) for easy local setup.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for platform-specific instructions.

**Launch OpenHands**:
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

Run OpenHands with Docker:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.53
```

</details>

> **Note**: If you used OpenHands before version 0.44, run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history.

> [!WARNING]
> Secure your deployment on public networks with the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

### Setup

1.  Open the application.
2.  Choose an LLM provider.  [Anthropic's Claude Sonnet 4](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-20250514`) is recommended.
3.  Add your API key.

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for system requirements and more information.

## 💡 Other Ways to Run OpenHands

> [!WARNING]
> OpenHands is designed for single-user, local workstation use. Multi-tenant deployments are not supported.

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem)
*   [CLI access](https://docs.all-hands.dev/usage/how-to/cli-mode)
*   [Headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode)
*   [GitHub Action integration](https://docs.all-hands.dev/usage/how-to/github-action)

Find more information at [Running OpenHands](https://docs.all-hands.dev/usage/installation).

For source code modification, see [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Troubleshooting help is available in the [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting).

## 📖 Documentation

<a href="https://deepwiki.com/All-Hands-AI/OpenHands"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" title="Autogenerated Documentation by DeepWiki"></a>

Explore the [documentation](https://docs.all-hands.dev/usage/getting-started) for LLM provider options, troubleshooting, and advanced configuration.

## 🤝 Join the Community

*   [Join our Slack workspace](https://join.slack.com/t/openhands-ai/shared_invite/zt-3847of6xi-xuYJIPa6YIPg4ElbDWbtSA)
*   [Join our Discord server](https://discord.gg/ESHStjSjD4)
*   [GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues)
*   See [COMMUNITY.md](./COMMUNITY.md) or [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution details.

## 📈 Progress

View the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License.  Learn more in [`LICENSE`](./LICENSE).

## 🙏 Acknowledgements

OpenHands is built with contributions from many individuals and leverages other open-source projects. See our [CREDITS.md](./CREDITS.md) file for details.

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

[Back to Top](#readme-top) - [View Original Repo](https://github.com/All-Hands-AI/OpenHands)
```
Key improvements:

*   **SEO Optimization:** Added keywords like "AI," "software development," "code," and "agents" throughout the README.  Used descriptive headings.
*   **Concise Hook:**  Replaced the introductory paragraphs with a single, impactful sentence.
*   **Bulleted Key Features:** Organized the main benefits for easy scanning.
*   **Clear Instructions:** Streamlined installation and setup instructions.
*   **Community and Resources:** Emphasized community links and documentation.
*   **Stronger Calls to Action:** Encouraged use of OpenHands Cloud.
*   **Back to Top Link:** Added a "Back to Top" link for navigation and linking back to the original repo.
*   **Readability:** Improved formatting with bolding, spacing, and clear sectioning.
*   **Concise Summarization:** Trimmed unnecessary details.
*   **Removed redundancies:** Combined similar statements or headings to avoid clutter.