# AutoGPT: Automate Complex Workflows with AI Agents

**Unleash the power of AI and automate your tasks with AutoGPT, a cutting-edge platform for building, deploying, and managing intelligent agents.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents using a low-code, intuitive interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease, connecting modular blocks for action-based agents.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Utilize a library of ready-to-use agents for immediate automation.
*   **Agent Interaction:** Run and interact with your own or pre-configured agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights to improve automation processes.

## Getting Started: Self-Hosting

AutoGPT offers a self-hosting option, empowering you to take control of your AI agent deployment.

*   **[Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)**:  Follow the detailed guide for setting up and running AutoGPT.

### Quick Setup (Recommended)

Use the one-line script to get started quickly:

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```
*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

### System Requirements

Ensure your system meets these requirements before setup:

#### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: Minimum 8GB, 16GB recommended
- Storage: At least 10GB of free space

#### Software Requirements
- Operating Systems:
  - Linux (Ubuntu 20.04 or newer recommended)
  - macOS (10.15 or newer)
  - Windows 10/11 with WSL2
- Required Software (with minimum versions):
  - Docker Engine (20.10.0 or newer)
  - Docker Compose (2.0.0 or newer)
  - Git (2.30 or newer)
  - Node.js (16.x or newer)
  - npm (8.x or newer)
  - VSCode (1.60 or newer) or any modern code editor

#### Network Requirements
- Stable internet connection
- Access to required ports (will be configured in Docker)
- Ability to make outbound HTTPS connections

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user-friendly interface for interacting with your AI automation platform.

*   **Agent Builder**: Customize and configure your own AI agents via the intuitive, low-code interface.
*   **Workflow Management**: Build, modify, and optimize automation workflows with ease, connecting modular blocks for action-based agents.
*   **Deployment Controls**: Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents**: Select from our library of pre-configured agents and put them to work immediately.
*   **Agent Interaction**: Run and interact with your own or pre-configured agents through a user-friendly interface.
*   **Monitoring and Analytics**: Track agent performance and gain insights to improve automation processes.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The engine that powers your AI agents, running and deploying agents.

*   **Source Code:** The core logic that drives our agents and automation processes.
*   **Infrastructure:** Robust systems that ensure reliable and scalable performance.
*   **Marketplace:** A comprehensive marketplace where you can find and deploy a wide range of pre-built agents.

## Example Use Cases

1.  **Generate Viral Videos:** Automate the creation of short-form videos based on trending topics.
2.  **Identify Top Quotes:** Automatically extract impactful quotes from videos and generate social media posts.

## License

*   **Polyform Shield License:** Code and content within the `autogpt_platform` folder. [Read more](https://agpt.co/blog/introducing-the-autogpt-platform)
*   **MIT License:**  All other parts of the AutoGPT repository (excluding the `autogpt_platform` folder), including the original AutoGPT Agent and related projects.

## AutoGPT Classic

Information about the classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

<!-- TODO: insert visual demonstrating the benchmark -->

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

<!-- TODO: insert screenshot of front end -->

The frontend works out-of-the-box with all agents in the repo. Just use the [CLI] to run your agent of choice!

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend) about the Frontend

### ⌨️ CLI

[CLI]: #-cli

To make it as easy as possible to use all of the tools offered by the repository, a CLI is included at the root of the repo:

```shell
$ ./run
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent      Commands to create, start and stop agents
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

Just clone the repo, install dependencies with `./run setup`, and you should be good to go!

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure someone else hasn't created an issue for the same topic.

## 🤝 Sister projects

### 🔄 Agent Protocol

To maintain a uniform standard and ensure seamless compatibility with many current and future applications, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

---

## Stars stats

<p align="center">
<a href="https://star-history.com/#Significant-Gravitas/AutoGPT">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
  </picture>
</a>
</p>


## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>