<!-- Improved README with SEO and Structure -->
<div align="center">
  <img src="./docs/static/img/logo.png" alt="OpenHands Logo" width="200">
  <br/>
  <a href="https://github.com/All-Hands-AI/OpenHands">
    <img src="https://img.shields.io/badge/View_on_GitHub-OpenHands-blue?style=for-the-badge&logo=github" alt="View on GitHub">
  </a>
  <br/>
  <h1 align="center">OpenHands: Empowering AI Software Development</h1>
</div>

<hr>

**OpenHands revolutionizes software development, allowing AI agents to write and manage code, freeing you to focus on innovation.**

OpenHands is a powerful open-source platform designed to enable AI-powered software development agents. These agents can perform tasks like a human developer, including code modification, command execution, web browsing, API calls, and more.

## Key Features

*   **AI-Powered Code Generation & Modification:** Automate code writing, editing, and debugging with AI assistance.
*   **Web Browsing & API Integration:** Access information and connect with external services directly within your development workflow.
*   **Local & Cloud Deployment:** Run OpenHands locally or leverage the convenience of OpenHands Cloud.
*   **Open Source & Customizable:** Built on open-source principles, allowing for flexibility and community contributions.
*   **Collaboration & Community:** Join a vibrant community through Slack, Discord, and GitHub to share knowledge and contribute to OpenHands.

## Getting Started

### OpenHands Cloud

The easiest way to get started is with [OpenHands Cloud](https://app.all-hands.dev), which offers $20 in free credits for new users.

### Running OpenHands Locally

Choose an option for running OpenHands locally:

**Option 1: CLI Launcher (Recommended)**

1.  **Install uv (if you haven't already):**  Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2.  **Launch OpenHands:**
    ```bash
    # Launch the GUI server
    uvx --python 3.12 --from openhands-ai openhands serve

    # Or launch the CLI
    uvx --python 3.12 --from openhands-ai openhands
    ```

    Access OpenHands via the GUI at [http://localhost:3000](http://localhost:3000).

**Option 2: Docker**

1.  **Run Docker command:**

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

**Important Notes:**

*   If you used OpenHands before version 0.44, consider migrating your conversation history: `mv ~/.openhands-state ~/.openhands`.
*   For secure deployments on public networks, consult the [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation).

**Initial Setup:**

Select an LLM provider (Anthropic's Claude Sonnet 4 is recommended: `anthropic/claude-sonnet-4-20250514`) and add your API key.  Explore [many options](https://docs.all-hands.dev/usage/llms).

For system requirements and more details, see the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide.

## Other Ways to Run OpenHands

*   [Connect to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem).
*   Interact via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode).
*   Run in a [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode) for scripting.
*   Use a [GitHub Action](https://docs.all-hands.dev/usage/how-to/github-action) with tagged issues.

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for setup instructions.

## Documentation

*   Comprehensive [documentation](https://docs.all-hands.dev/usage/getting-started) provides in-depth usage guides, LLM provider details, troubleshooting, and advanced configuration options.

## Community & Support

OpenHands thrives on community contributions! Join us and get involved:

*   **Slack:** [Join our Slack workspace](https://dub.sh/openhands) for discussions on research, architecture, and development.
*   **Discord:** [Join our Discord server](https://discord.gg/ESHStjSjD4) for general discussions and feedback.
*   **GitHub Issues:** [Read or post GitHub Issues](https://github.com/All-Hands-AI/OpenHands/issues) for ideas and bug reports.
*   More community details: [COMMUNITY.md](./COMMUNITY.md).
*   Contributing details: [CONTRIBUTING.md](./CONTRIBUTING.md).

## Progress & Roadmap

*   Track project progress:  [OpenHands roadmap](https://github.com/orgs/All-Hands-AI/projects/1) (updated monthly).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## License

*   Distributed under the MIT License.  See [`LICENSE`](./LICENSE) for details.  The `enterprise/` folder has its own license.

## Acknowledgements

*   We are deeply thankful for the contributions of the community and the open-source projects we rely upon. See [CREDITS.md](./CREDITS.md) for a list of open-source projects and licenses.

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