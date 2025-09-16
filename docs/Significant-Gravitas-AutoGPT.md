# AutoGPT: Unleash the Power of AI with Autonomous Agents

**Automate complex workflows and revolutionize your productivity with AutoGPT, the leading platform for building, deploying, and managing AI agents.**  [Visit the original repository](https://github.com/Significant-Gravitas/AutoGPT)

[![Discord](https://img.shields.io/discord/1098725023475025449?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

AutoGPT empowers you to create, deploy, and manage AI agents that automate intricate tasks and workflows, transforming the way you work.

## Key Features

*   **Agent Builder:**  Design and configure custom AI agents with an intuitive, low-code interface.
*   **Workflow Management:**  Build, modify, and optimize automation workflows with ease through a drag-and-drop interface.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents for immediate deployment and use.
*   **Agent Interaction:**  Easily run and interact with your agents via a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights to continuously improve your automation processes.

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon!):** [Join the Waitlist](https://bit.ly/3ZDijAI) for access to the cloud-hosted beta.

## Getting Started: Self-Hosting

**Important Note:** Self-hosting requires technical expertise. For an easier experience, consider joining the cloud-hosted beta.

### System Requirements

Ensure your system meets these requirements before installation:

*   **Hardware:** 4+ CPU cores (recommended), Minimum 8GB RAM (16GB recommended), At least 10GB storage
*   **Software:**
    *   Operating Systems: Linux (Ubuntu 20.04+ recommended), macOS (10.15+), Windows 10/11 with WSL2
    *   Docker Engine (20.10.0+), Docker Compose (2.0.0+), Git (2.30+), Node.js (16.x+), npm (8.x+), VSCode (1.60+) or any modern code editor
*   **Network:** Stable internet connection, Access to required ports (configured in Docker), Outbound HTTPS connections.

### Setup Instructions

The AutoGPT Platform now has its own dedicated documentation.

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git, and npm installed.

### Quick Setup

Simplify your setup with our one-line script (recommended for local hosting):

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```

*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

This script installs dependencies, configures Docker, and launches your local instance.

## AutoGPT Frontend: Your AI Automation Hub

The AutoGPT frontend is the user interface for interacting with the AI automation platform.

*   **Agent Builder:** Customize agents with a low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Select and deploy pre-configured agents immediately.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Keep track of your agents' performance.

[Learn how to build your custom blocks](https://docs.agpt.co/platform/new_blocks/).

## AutoGPT Server: The AI Engine

The AutoGPT Server houses the core components of the platform.

*   **Source Code:** The engine that drives the agents.
*   **Infrastructure:** Ensures reliable and scalable performance.
*   **Marketplace:** Discover and deploy pre-built agents.

## Example Agents: Real-World Applications

*   **Generate Viral Videos:** Reads trending topics, creates short-form videos.
*   **Identify Top Quotes:** Transcribes YouTube videos, extracts impactful quotes, and generates social media posts.

## Licenses

*   **Polyform Shield License:** All code within the `autogpt_platform` folder. [Learn More](https://agpt.co/blog/introducing-the-autogpt-platform)
*   **MIT License:** All other portions of the repository, including [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge), [agbenchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark), and the [AutoGPT Classic GUI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend).

## Mission

Our mission is to empower you to:

*   🏗️ **Build** - Lay the foundation for your automation.
*   🧪 **Test** - Fine-tune your agents to perfection.
*   🤝 **Delegate** - Let AI work for you.

Join the AI revolution.  **AutoGPT** is at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

## AutoGPT Classic

This section provides information on the classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

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

## Get Help

*   **Discord 💬:** [Join our Discord community](https://discord.gg/autogpt) for support and discussions.

*   **GitHub Issues:** Report bugs or request features:  [Create a GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## Sister Projects

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation to ensure compatibility with numerous applications.

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

## Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>