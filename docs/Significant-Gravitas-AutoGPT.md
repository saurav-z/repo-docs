# AutoGPT: Automate Your World with AI Agents

**Unleash the power of AI with AutoGPT, a platform that empowers you to build, deploy, and manage intelligent AI agents for unparalleled automation.** ([Back to the original repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

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

*   **AI Agent Creation:** Design and configure custom AI agents tailored to your specific needs.
*   **Workflow Automation:** Build, modify, and optimize automation workflows with an intuitive interface.
*   **Agent Management:** Deploy, test, and manage the lifecycle of your AI agents with ease.
*   **Pre-built Agent Library:** Access a library of ready-to-use agents for immediate deployment.
*   **Agent Interaction:** Seamlessly run and interact with your agents through a user-friendly interface.
*   **Performance Monitoring:** Track agent performance and gain insights for continuous improvement.
*   **Self-Hosting Option:** Download and self-host for free, or join the waitlist for the cloud-hosted beta.

## Hosting Options

*   Download to self-host (Free!)
*   [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Closed Beta - Public release Coming Soon!)

## Self-Hosting: Get Started Quickly

> [!NOTE]
> Self-hosting AutoGPT requires technical expertise. We recommend joining the [cloud-hosted beta waitlist](https://bit.ly/3ZDijAI) for a simpler experience.

### System Requirements

**Hardware:**

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

**Software:**

*   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), Windows 10/11 with WSL2
*   Required Software (with minimum versions): Docker Engine (20.10.0 or newer), Docker Compose (2.0.0 or newer), Git (2.30 or newer), Node.js (16.x or newer), npm (8.x or newer), VSCode (1.60 or newer) or any modern code editor

**Network:**

*   Stable internet connection
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

### Setup Instructions

For detailed, up-to-date instructions, please refer to the official documentation: [https://docs.agpt.co/platform/getting-started/](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

---

### ⚡ One-Line Quick Setup (Recommended)

Simplify the setup process with our automatic script.

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user interface for interacting with and leveraging AI agents.

*   **Agent Builder:** Customize agents with a low-code interface.
*   **Workflow Management:** Easily build and optimize automation workflows.
*   **Deployment Controls:** Manage the entire agent lifecycle.
*   **Ready-to-Use Agents:** Deploy pre-configured agents instantly.
*   **Agent Interaction:** Run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance.

[Learn more about building custom blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The engine that powers your AI agents.

*   **Source Code:** The core logic for agents and automation.
*   **Infrastructure:** Robust systems for reliable performance.
*   **Marketplace:** Access a wide range of pre-built agents.

## 🐙 Example Agents

*   **Generate Viral Videos:** Create short-form videos from trending topics.
*   **Identify Top Quotes from Videos:** Transcribe videos, identify impactful quotes, and automatically publish social media posts.

## 🛡️ Licensing

*   **Polyform Shield License:** (`autogpt_platform` folder)
*   **MIT License:** (All other parts of the repository)

---

### Mission

*   **Building:** Lay the foundation for something amazing.
*   **Testing:** Fine-tune your agent to perfection.
*   **Delegating:** Let AI work for you, and have your ideas come to life.

Be part of the AI revolution!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic

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

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) by the AI Engineer Foundation for seamless compatibility.

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