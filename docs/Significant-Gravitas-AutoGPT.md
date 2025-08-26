# AutoGPT: The Future of AI Automation

**Unleash the power of autonomous AI agents to automate complex tasks and revolutionize your workflows.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1092035855217800212?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

---
**Key Features**

*   **Autonomous AI Agents:** Build, deploy, and manage AI agents capable of performing complex tasks independently.
*   **Flexible Hosting Options:** Choose to self-host or join the cloud-hosted beta for easy access.
*   **Agent Builder:** Design and customize your own AI agents with a low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows with ease.
*   **Ready-to-Use Agents:** Leverage pre-configured agents for immediate automation solutions.
*   **Monitoring and Analytics:** Track agent performance and gain insights to improve your automation processes.
---

## Getting Started with AutoGPT

AutoGPT offers a platform to create, deploy, and manage continuous AI agents.

### Hosting Options

*   **Self-Host:** Download and run AutoGPT for free on your own infrastructure.
*   **Cloud-Hosted Beta:** Join the waitlist for the upcoming cloud-hosted beta version. ([Join the Waitlist](https://bit.ly/3ZDijAI))

### Self-Hosting Guide

> [!NOTE]
> Self-hosting requires some technical expertise. Consider joining the cloud-hosted beta for a simpler experience.

#### System Requirements

Ensure your system meets the following requirements before installation:

**Hardware:**

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

**Software:**

*   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), Windows 10/11 with WSL2
*   Required Software (with minimum versions):
    *   Docker Engine (20.10.0 or newer)
    *   Docker Compose (2.0.0 or newer)
    *   Git (2.30 or newer)
    *   Node.js (16.x or newer)
    *   npm (8.x or newer)
    *   VSCode (1.60 or newer) or any modern code editor

**Network:**

*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Ability to make outbound HTTPS connections

#### Setup Instructions

For detailed self-hosting instructions, follow the official documentation:
👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

**Quick Setup (Recommended)**

Use the following one-line script for a rapid setup:

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```
*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

## Core Components

### 🧱 AutoGPT Frontend

Interact with your AI automation platform. This is the interface where you'll bring your AI automation ideas to life.

*   **Agent Builder:** Customize and configure AI agents.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the agent lifecycle.
*   **Ready-to-Use Agents:** Access pre-configured agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance.

[Read this guide](https://docs.agpt.co/platform/new_blocks/) to learn how to build your own custom blocks.

### 💽 AutoGPT Server

The engine that powers your agents.

*   **Source Code:** The core logic for agents and automation.
*   **Infrastructure:** Reliable and scalable systems.
*   **Marketplace:** A marketplace for pre-built agents.

## 🐙 Example Agents

See what's possible with AutoGPT:

1.  **Generate Viral Videos:** Create videos from trending topics on Reddit.
2.  **Extract Quotes:** Identify and summarize impactful quotes from YouTube videos for social media.

## 📜 License

*   All code and content within the `autogpt_platform` folder is licensed under the [Polyform Shield License](https://agpt.co/blog/introducing-the-autogpt-platform).
*   All other parts of the repository are licensed under the [MIT License](https://github.com/Significant-Gravitas/AutoGPT).

## 🤝 Mission

-   🏗️ **Building** - Lay the foundation for something amazing.
-   🧪 **Testing** - Fine-tune your agent to perfection.
-   🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Be part of the revolution! **AutoGPT** is here to stay, at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic (Legacy Version)

> Information about the classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

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

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend) about the Frontend

### ⌨️ CLI

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

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) to ensure compatibility.

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