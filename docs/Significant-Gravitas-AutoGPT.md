# AutoGPT: Unleash the Power of AI Agents 🤖

**Automate complex workflows and revolutionize your productivity with AutoGPT, a platform to build, deploy, and manage autonomous AI agents. [Explore the original repository](https://github.com/Significant-Gravitas/AutoGPT).**

[![Discord](https://img.shields.io/discord/1068441469437163550?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features:

*   **AI Agent Creation:** Design and configure your own AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows with ease using a drag-and-drop interface.
*   **Deployment & Management:** Deploy, manage, and monitor the lifecycle of your agents, from testing to production.
*   **Pre-Built Agents:** Jumpstart your projects with a library of ready-to-use, pre-configured agents.
*   **Agent Interaction:** Seamlessly run and interact with your custom or pre-built agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track your agents' performance and gain insights to continually improve your automation processes.
*   **Flexible Hosting:** Self-host for free or join the [cloud-hosted beta](https://bit.ly/3ZDijAI) (coming soon).

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own hardware.  See the setup instructions below.
*   **Cloud-Hosted Beta:**  [Join the waitlist](https://bit.ly/3ZDijAI) for the upcoming cloud-hosted beta for a managed experience.

## Self-Hosting AutoGPT

**Note:** Self-hosting requires a degree of technical expertise. If you prefer a simplified setup, consider joining the cloud-hosted beta.

### System Requirements

Ensure your system meets these requirements before proceeding:

#### Hardware:
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

#### Software:
*   Operating Systems:
    *   Linux (Ubuntu 20.04 or newer recommended)
    *   macOS (10.15 or newer)
    *   Windows 10/11 with WSL2
*   Required Software (with minimum versions):
    *   Docker Engine (20.10.0 or newer)
    *   Docker Compose (2.0.0 or newer)
    *   Git (2.30 or newer)
    *   Node.js (16.x or newer)
    *   npm (8.x or newer)
    *   VSCode (1.60 or newer) or any modern code editor

#### Network:
*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Outbound HTTPS connections

### Setup Guide

For the most up-to-date setup instructions, please refer to the official documentation:
👉 [Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

This guide assumes you have Docker, VSCode, git, and npm installed.

### Quick Setup with One-Line Script (Local Hosting)

Simplify the setup process with our automated script:

**For macOS/Linux:**
```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**
```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```
This script installs dependencies, configures Docker, and launches your local instance.

## AutoGPT Platform Components

### 🧱 Frontend

Interact with and leverage our AI automation platform through the user-friendly frontend, which offers:

*   **Agent Builder:**  Customize agents with a visual, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:**  Deploy pre-configured agents instantly.
*   **Agent Interaction:** Run and engage with your agents through the interface.
*   **Monitoring and Analytics:** Track and improve agent performance.

[Learn more about building custom blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 Server

The server is where your agents execute:

*   **Source Code:** The core logic that drives agents and automation.
*   **Infrastructure:** Ensures reliable and scalable performance.
*   **Marketplace:** A platform to discover and deploy agents.

### 🐙 Example Agents

See what's possible with AutoGPT!

1.  **Generate Viral Videos:** Create short-form videos from trending topics.
2.  **Extract Quotes for Social Media:** Automatically transcribe, summarize, and post impactful quotes from your videos.

You can create custom workflows for any use case.

---

## Licensing

🛡️ **Polyform Shield License:** Code in the `autogpt_platform` folder.  [Learn More](https://agpt.co/blog/introducing-the-autogpt-platform)

🦉 **MIT License:** All other parts of the repository, including the original AutoGPT Agent, Forge, the benchmark, and the GUI.
MIT-licensed projects include [GravitasML](https://github.com/Significant-Gravitas/gravitasml) and [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability).

---
### Mission

Our mission is to empower you to:

*   🏗️ **Build** - Create something amazing.
*   🧪 **Test** - Fine-tune your agents.
*   🤝 **Delegate** - Let AI automate your tasks.

**Join the AI revolution!**

**📖 [Documentation](https://docs.agpt.co)** &ensp;|&ensp; **🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic (Legacy Version)

Information about the original AutoGPT project:

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

[Create a GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose) to report a bug or request a feature.  Check for existing issues first.

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) for standardized communication between agents, frontends, and benchmarks.

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