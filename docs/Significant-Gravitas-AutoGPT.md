# AutoGPT: Automate Your World with AI Agents

**Unleash the power of AI by building, deploying, and running intelligent agents with AutoGPT, a cutting-edge platform for automating complex workflows.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1094853723394828338?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
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

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents using an intuitive, low-code interface.
*   **Workflow Automation:** Easily build, modify, and optimize your automation workflows through a visual block-based system.
*   **Agent Deployment & Management:**  Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Utilize a library of pre-configured agents for immediate use.
*   **Agent Interaction:** Run and interact with your created or pre-configured agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights for continuous improvement.

## Hosting Options

*   **Self-host (Free):** Download and set up AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon!):** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

## Self-Hosting Guide

> [!NOTE]
> Self-hosting AutoGPT requires technical expertise.  Consider the cloud-hosted beta if you prefer an easier setup.

### System Requirements

Ensure your system meets the following criteria before installation:

#### Hardware Requirements

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

#### Software Requirements

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

#### Network Requirements

*   Stable internet connection
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

### Setup Instructions

For the most up-to-date installation instructions, please refer to our comprehensive documentation:

👉 [Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

---

#### ⚡ Quick Setup with One-Line Script (Recommended for Local Hosting)

Simplify the setup process with our automated script:

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script handles dependency installation, Docker configuration, and local instance launch.

### 🧱 AutoGPT Frontend

The user interface for interacting with your AI agents.

*   **Agent Builder:** Create custom AI agents using a low-code interface.
*   **Workflow Management:** Design, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Select and deploy pre-configured agents.
*   **Agent Interaction:** Run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for improvements.

[Learn more about building custom blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The core engine where your AI agents run.

*   **Source Code:** Core logic for agents and automation.
*   **Infrastructure:** Ensures reliable, scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

### 🐙 Example Agents

Discover the potential of AutoGPT:

1.  **Generate Viral Videos:** Automates video creation from trending topics on Reddit.
2.  **Extract Quotes for Social Media:** Transcribes YouTube videos and identifies impactful quotes for social media posts.

Create custom workflows and build agents for any use case.

---

### **License Overview:**

🛡️ **Polyform Shield License:**
All code and content within the `autogpt_platform` folder is licensed under the Polyform Shield License. This new project is our in-developlemt platform for building, deploying and managing agents.</br>_[Read more about this effort](https://agpt.co/blog/introducing-the-autogpt-platform)_

🦉 **MIT License:**
All other portions of the AutoGPT repository (i.e., everything outside the `autogpt_platform` folder) are licensed under the MIT License. This includes the original stand-alone AutoGPT Agent, along with projects such as [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge), [agbenchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) and the [AutoGPT Classic GUI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend).</br>We also publish additional work under the MIT Licence in other repositories, such as [GravitasML](https://github.com/Significant-Gravitas/gravitasml) which is developed for and used in the AutoGPT Platform. See also our MIT Licenced [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability) project.

---

### Mission

Our mission is to empower you to focus on what matters:

*   🏗️ **Building** - Lay the foundation for something amazing.
*   🧪 **Testing** - Fine-tune your agent to perfection.
*   🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Join the AI revolution! **AutoGPT** is at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic

> Information about the classic version of AutoGPT.

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

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Ensure no duplicate issues exist.

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) by the AI Engineer Foundation, ensuring uniform standards and seamless compatibility with applications.

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