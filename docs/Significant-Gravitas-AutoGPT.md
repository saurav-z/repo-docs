# AutoGPT: Unleash the Power of AI with Autonomous Agents

**Build, deploy, and run AI agents to automate complex tasks with AutoGPT – your gateway to the future of AI-powered automation.**  ([View on GitHub](https://github.com/Significant-Gravitas/AutoGPT))

## Key Features

*   **AI Agent Creation:** Design and configure your own AI agents with an intuitive, low-code agent builder.
*   **Workflow Management:** Easily build, modify, and optimize your automation workflows.
*   **Deployment & Management:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Access a library of ready-to-use agents for immediate deployment.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Performance Monitoring:** Track agent performance and gain insights for continuous improvement.

## Hosting Options

*   **Self-Host (Free!):** Download and set up AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon!):** Join the [waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted platform.

## Self-Hosting Guide

> Setting up and hosting AutoGPT requires technical expertise. For a simpler experience, consider joining the cloud-hosted beta.

### System Requirements

**Hardware:**
*   CPU: 4+ cores recommended
*   RAM: 8GB minimum, 16GB recommended
*   Storage: 10GB+ free space

**Software:**
*   Operating Systems: Linux (Ubuntu 20.04+ recommended), macOS (10.15+), Windows 10/11 with WSL2
*   Required Software: Docker Engine (20.10.0+), Docker Compose (2.0.0+), Git (2.30+), Node.js (16.x+), npm (8.x+), VSCode (1.60+) or any modern code editor

**Network:**
*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Outbound HTTPS connections

### Installation

**Simplified Setup (Recommended):** Use the one-line script for a quick installation:

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```
*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

This script automates dependency installation, Docker configuration, and local instance launch.

### 🔑 AutoGPT Frontend

Interact with your AI automation platform. Key features include:
*   Agent Builder
*   Workflow Management
*   Deployment Controls
*   Ready-to-Use Agents
*   Agent Interaction
*   Monitoring and Analytics

[Learn more about building custom blocks.](https://docs.agpt.co/platform/new_blocks/)

### ⚙️ AutoGPT Server

The core of the platform, where agents run. Includes:

*   Source Code
*   Infrastructure
*   Marketplace

### 🤖 Example Agents

Here's what you can do with AutoGPT:
1.  **Generate Viral Videos:** Create short-form videos from trending topics on Reddit.
2.  **Identify Top Quotes:** Extract impactful quotes from your YouTube videos for social media.

Customize workflows to build agents for any use case.

---

### Licensing

*   **Polyform Shield License:** Code within the `autogpt_platform` folder. [Learn more](https://agpt.co/blog/introducing-the-autogpt-platform)
*   **MIT License:** All other parts of the repository, including the original AutoGPT agent, Forge, agbenchmark and the AutoGPT Classic GUI. Also used for projects like GravitasML and Code Ability.

---

### Mission

Our mission empowers you to:

*   🏗️ **Build** amazing AI applications.
*   🧪 **Test** and refine your agents.
*   🤝 **Delegate** tasks to AI.

Be part of the AI revolution!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic

> Below is information about the classic version of AutoGPT.

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

Report issues and request features on [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT utilizes the [agent protocol](https://agentprotocol.ai/) standard for seamless compatibility.

---

## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>