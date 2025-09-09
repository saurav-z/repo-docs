# AutoGPT: Unleash AI Automation and Build Intelligent Agents

**AutoGPT empowers you to build, deploy, and run AI agents, automating complex workflows and revolutionizing productivity.** ([Back to Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

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

## Key Features

*   **AI Agent Creation and Management:** Design, deploy, and manage continuous AI agents.
*   **Intuitive Agent Builder:**  Low-code interface for customizing AI agents using blocks.
*   **Workflow Management:** Easily build, modify, and optimize automation workflows.
*   **Pre-Built Agents:** Leverage ready-to-use agents for immediate productivity.
*   **Agent Interaction:** Interact with both custom and pre-configured agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for improvement.

## Hosting Options

*   **Self-Host (Free!):**  Download and host the platform yourself.
*   **Cloud-Hosted Beta (Coming Soon!):** [Join the Waitlist](https://bit.ly/3ZDijAI)

## Self-Hosting Guide

> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. 
> If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

### System Requirements

#### Hardware
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

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
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

### Setup Instructions

For detailed, up-to-date setup instructions, see the official documentation: 👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

### Quick Setup (One-Line Script)

Simplify setup with an automated script.

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user interface for interacting with and managing AI agents:

*   **Agent Builder:** Build and customize AI agents.
*   **Workflow Management:** Create and optimize automation workflows.
*   **Deployment Controls:** Manage agent lifecycles (testing to production).
*   **Ready-to-Use Agents:** Access pre-configured agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance.

[Learn more about building custom blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The core of the platform where agents run, featuring:

*   **Source Code:** The foundational logic for agents and automation.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** Access to a wide range of pre-built agents.

### 🐙 Example Agents

Automate tasks with pre-built or custom agents:

1.  **Generate Viral Videos:** Reads trending topics on Reddit, creates short-form videos.
2.  **Identify Top Quotes from Videos:** Transcribes YouTube videos, identifies key quotes, and generates social media posts.

## ⚙️ AutoGPT Classic

### 🛠️ Forge

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

## 🤔 Questions, Problems, or Suggestions?

*   **Get Help:** [Discord 💬](https://discord.gg/autogpt)
*   **Report Issues:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard for seamless compatibility.

---

## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>

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
---

### **License Overview:**

🛡️ **Polyform Shield License:**
All code and content within the `autogpt_platform` folder is licensed under the Polyform Shield License. This new project is our in-developlemt platform for building, deploying and managing agents.</br>_[Read more about this effort](https://agpt.co/blog/introducing-the-autogpt-platform)_

🦉 **MIT License:**
All other portions of the AutoGPT repository (i.e., everything outside the `autogpt_platform` folder) are licensed under the MIT License. This includes the original stand-alone AutoGPT Agent, along with projects such as [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge), [agbenchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) and the [AutoGPT Classic GUI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend).</br>We also publish additional work under the MIT Licence in other repositories, such as [GravitasML](https://github.com/Significant-Gravitas/gravitasml) which is developed for and used in the AutoGPT Platform. See also our MIT Licenced [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability) project.

---
### Mission
Our mission is to provide the tools, so that you can focus on what matters:

- 🏗️ **Building** - Lay the foundation for something amazing.
- 🧪 **Testing** - Fine-tune your agent to perfection.
- 🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Be part of the revolution! **AutoGPT** is here to stay, at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**