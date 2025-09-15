# AutoGPT: Automate and Innovate with AI Agents

**Unleash the power of autonomous AI agents to build, deploy, and manage complex workflows with AutoGPT, and revolutionize your approach to automation.** ([Back to Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1095667715202077204?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
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

## Key Features of AutoGPT

*   **AI Agent Creation:** Design and configure custom AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease using a block-based approach.
*   **Deployment & Management:** Control the lifecycle of your agents from testing to production.
*   **Ready-to-Use Agents:** Quickly deploy pre-configured agents from a comprehensive library.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Performance Monitoring:** Track your agents' performance and gain insights to continually improve automation processes.

## Hosting Options

*   **Self-Host (Free):** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon):**  Join the [waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta and experience AutoGPT with minimal setup.

## Getting Started with Self-Hosting

> [!NOTE]
> Self-hosting requires technical knowledge. Consider the cloud-hosted beta for a simpler experience.

### System Requirements

Ensure your system meets these prerequisites for a smooth setup:

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

### Installation Guide

For detailed self-hosting instructions, consult the official documentation:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

---

#### ⚡ Quick Setup (Recommended)

Simplify installation with our one-line script, ideal for local hosting:

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automates dependency installation, Docker configuration, and local instance launching.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user interface to interact with AI agents. Key features include:

*   **Agent Builder:** Customization through a low-code interface.
*   **Workflow Management:** Easily manage and optimize your automation processes.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Select from pre-configured agents and deploy immediately.
*   **Agent Interaction:** Run and interact with your agents through our user-friendly interface.
*   **Monitoring and Analytics:** Keep track of your agents' performance and gain insights to continually improve your automation processes.

Learn how to build custom blocks: [How to build custom blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The core of the platform, housing the following:

*   **Source Code:** The core logic powering agents and automation.
*   **Infrastructure:** Reliable systems for scalable performance.
*   **Marketplace:**  A comprehensive marketplace to find and deploy agents.

### 🐙 Example Agents

Here are a couple examples of what you can achieve with AutoGPT:

1.  **Generate Viral Videos from Trending Topics:**
    *   Reads topics on Reddit.
    *   Identifies trending topics.
    *   Creates a short-form video based on the content.

2.  **Identify Top Quotes from Videos for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Uses AI to identify impactful quotes and generate a summary.
    *   Writes a post for automated social media publishing.

## Licensing

*   🛡️ **Polyform Shield License:** Code and content within the `autogpt_platform` folder. ([Read more about this effort](https://agpt.co/blog/introducing-the-autogpt-platform))
*   🦉 **MIT License:** All other parts of the repository, including the original AutoGPT Agent, Forge, agbenchmark, and the AutoGPT Classic GUI.

## Mission

Our mission is to empower you with the tools to:

*   🏗️ **Build** - Lay the foundation for something amazing.
*   🧪 **Test** - Fine-tune your agent to perfection.
*   🤝 **Delegate** - Let AI work for you, and have your ideas come to life.

Join the AI revolution! **AutoGPT** is at the forefront of innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic (Legacy Version)

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

For bug reports or feature requests, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure no duplicates exist.

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) from the AI Engineer Foundation to ensure compatibility and standardization.

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