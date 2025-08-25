# AutoGPT: Build, Deploy, and Run AI Agents to Automate Anything

**Supercharge your workflow with AutoGPT, the open-source platform for building, deploying, and managing autonomous AI agents.** ([Back to Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1096523705506507826?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://zdoc.app/de/Significant-Gravitas/AutoGPT) | 
[Español](https://zdoc.app/es/Significant-Gravitas/AutoGPT) | 
[français](https://zdoc.app/fr/Significant-Gravitas/AutoGPT) | 
[日本語](https://zdoc.app/ja/Significant-Gravitas/AutoGPT) | 
[한국어](https://zdoc.app/ko/Significant-Gravitas/AutoGPT) | 
[Português](https://zdoc.app/pt/Significant-Gravitas/AutoGPT) | 
[Русский](https://zdoc.app/ru/Significant-Gravitas/AutoGPT) | 
[中文](https://zdoc.app/zh/Significant-Gravitas/AutoGPT)

## Key Features:

*   **AI Agent Creation:** Design and customize AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows with ease.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-Built Agents:** Access a library of ready-to-use AI agents for immediate deployment.
*   **Agent Interaction:** Easily run and interact with your custom or pre-configured agents.
*   **Monitoring and Analytics:** Track agent performance and gain insights to improve automation.

## Hosting Options

*   **Self-Host:** Download and run AutoGPT on your own hardware (Free!).  See below for self-hosting instructions.
*   **Cloud-Hosted Beta:**  [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Coming Soon!)

## Self-Hosting AutoGPT

> [!NOTE]
> Self-hosting AutoGPT requires technical expertise. Consider the cloud-hosted beta for an easier experience.

### System Requirements

**Hardware:**
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

**Software:**
*   Operating Systems: Linux (Ubuntu 20.04+ recommended), macOS (10.15+), Windows 10/11 with WSL2
*   Required Software: Docker Engine (20.10.0+), Docker Compose (2.0.0+), Git (2.30+), Node.js (16.x+), npm (8.x+), VSCode (1.60+) or any modern code editor

**Network:**
*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Ability to make outbound HTTPS connections

### Updated Setup Instructions

For detailed setup and configuration instructions, please refer to the official documentation:  👉 [Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

### Quick Setup with One-Line Script (Recommended for Local Hosting)

Simplify the setup process with our automated script:

**macOS/Linux:**
```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**Windows (PowerShell):**
```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automates dependency installation, Docker configuration, and local instance launching.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The frontend provides a user-friendly interface for interacting with your AI agents:

*   **Agent Builder:** Customize agents with a low-code interface.
*   **Workflow Management:** Design and modify automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Deploy pre-configured agents immediately.
*   **Agent Interaction:** Run and interact with your agents through the interface.
*   **Monitoring and Analytics:** Track agent performance and improve automation.

[Learn to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The AutoGPT Server powers your AI agents, running them continuously:

*   **Source Code:** The core logic that drives agents.
*   **Infrastructure:** Robust systems for reliable performance.
*   **Marketplace:** A marketplace to discover and deploy agents.

### 🐙 Example Agents

Discover the possibilities with these example agent applications:

1.  **Generate Viral Videos:** Automatically create short-form videos based on trending topics from Reddit.
2.  **Identify Top Quotes:** Subscribe to a YouTube channel, transcribe videos, and identify impactful quotes for social media posts.

## License

*   **Polyform Shield License:** Code and content within the `autogpt_platform` folder.
*   **MIT License:** All other parts of the repository (original AutoGPT, Forge, agbenchmark, AutoGPT Classic GUI, etc.).

## Mission

Our mission is to empower you with the tools to:

*   🏗️ **Build:** Create amazing AI-powered solutions.
*   🧪 **Test:** Fine-tune your agents to perfection.
*   🤝 **Delegate:** Let AI handle the work and bring your ideas to life.

## AutoGPT Classic

> Below is information about the classic version of AutoGPT.

### 🛠️ Build Your Own Agent - Quickstart

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

## Get Involved

### 🤔 Questions, Problems, or Suggestions?

*   **Get help:** [Discord 💬](https://discord.gg/autogpt)
*   **Report issues/feature requests:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation for seamless compatibility with various applications.

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