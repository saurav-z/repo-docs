# AutoGPT: Unleash the Power of Autonomous AI Agents

**Build, deploy, and manage powerful AI agents that automate complex tasks with AutoGPT, the leading open-source platform for AI automation. ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))**

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://zdoc.app/de/Significant-Gravitas/AutoGPT) | 
[Español](https://zdoc.app/es/Significant-Gravitas/AutoGPT) | 
[français](https://zdoc.app/fr/Significant-Gravitas/AutoGPT) | 
[日本語](https://zdoc.app/ja/Significant-Gravitas/AutoGPT) | 
[한국어](https://zdoc.app/ko/Significant-Gravitas/AutoGPT) | 
[Português](https://zdoc.app/pt/Significant-Gravitas/AutoGPT) | 
[Русский](https://zdoc.app/ru/Significant-Gravitas/AutoGPT) | 
[中文](https://zdoc.app/zh/Significant-Gravitas/AutoGPT)

## Key Features

*   **Autonomous AI Agents:** Create and deploy AI agents capable of self-directed task execution.
*   **Agent Builder:** A low-code interface to design and configure custom AI agents.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   **Deployment Controls:** Manage your agents' lifecycle, from testing to production.
*   **Pre-built Agents:** Access a library of ready-to-use AI agents.
*   **Real-time Monitoring & Analytics:** Track agent performance and refine automation processes.
*   **Open Source:** Leverage the power of community-driven development and customization.

## Hosting Options

*   **Self-Hosting:** Download and run AutoGPT locally for free.
*   **Cloud-Hosted Beta:** [Join the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Coming Soon!).

## Getting Started with Self-Hosting

> [!NOTE]
> Self-hosting AutoGPT is a technical process. Consider joining the [cloud-hosted beta](https://bit.ly/3ZDijAI) for an easier experience.

### System Requirements

Ensure your system meets these requirements:

#### Hardware
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB free space

#### Software
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

#### Network
*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Outbound HTTPS connections

### Updated Setup Instructions:

For detailed setup instructions, visit the official documentation site:  👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

### Quick Setup with One-Line Script

Simplify setup with our automatic script:

For macOS/Linux:
```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## AutoGPT Platform: The Core Components

### 🧱 AutoGPT Frontend

The user-friendly interface for interacting with and managing your AI agents.  Key features include:

*   **Agent Builder:** Design and customize agents.
*   **Workflow Management:** Create and optimize automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Deploy pre-built agents instantly.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance.

[Learn how to build your own custom blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The backend that powers your AI agents. It includes:

*   **Source Code:** The core logic of agents and automations.
*   **Infrastructure:** Systems for reliable performance.
*   **Marketplace:** A hub for pre-built agents.

## 🐙 Example Agents: Automate with AI

1.  **Viral Video Generation:** Create short-form videos from trending topics on Reddit.
2.  **Social Media Quote Extraction:** Automatically identify and post impactful quotes from your YouTube videos.

## License and Contributions

*   **Polyform Shield License:**  Applies to the `autogpt_platform` folder. [Learn more](https://agpt.co/blog/introducing-the-autogpt-platform)
*   **MIT License:**  Applies to the rest of the AutoGPT repository, including the original AutoGPT Agent, Forge, agbenchmark, and the AutoGPT Classic GUI.

### Mission

*   🏗️ **Build** amazing AI solutions.
*   🧪 **Test** and refine your agents.
*   🤝 **Delegate** tasks and let AI work for you.

Join the AI revolution!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic

Explore the classic AutoGPT version and its powerful tools:

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

## 🤔 Questions, Issues, or Suggestions?

*   **Get Help:** [Discord 💬](https://discord.gg/autogpt)
*   **Report Issues/Features:** [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) by the AI Engineer Foundation. This standardizes communication between your agent, frontend, and benchmark tools, allowing for seamless integration.

---

## Stars Stats

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