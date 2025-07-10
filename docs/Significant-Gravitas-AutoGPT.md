<!-- AutoGPT: The Ultimate Guide to AI Agent Automation -->
# AutoGPT: Create, Deploy, and Automate with AI Agents

**Unlock the power of AI automation with AutoGPT, the platform that empowers you to build, deploy, and run intelligent AI agents.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features of AutoGPT

*   **AI Agent Creation:** Design and configure custom AI agents using a low-code, intuitive interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease using a visual, drag-and-drop approach.
*   **Deployment Controls:** Manage the complete lifecycle of your AI agents, from testing to production.
*   **Pre-Built Agents:** Access a library of ready-to-use agents for instant automation.
*   **Agent Interaction:** Easily run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights to continually improve your automation processes.

## Hosting Options

*   **Self-Hosting:** Download and host the platform yourself. See the "Setup Instructions" section below.
*   **Cloud-Hosted Beta:**  [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta version.

## Setup Instructions (Self-Hosting)

> [!NOTE]
> Self-hosting requires technical expertise. For a simpler experience, consider joining the cloud-hosted beta.

### System Requirements

**Hardware Requirements**

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

**Software Requirements**

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

**Network Requirements**

*   Stable internet connection
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

### Detailed Setup Guide

For comprehensive, regularly updated setup instructions, please follow the official documentation:

👉 [Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

This guide assumes you have Docker, VSCode, git, and npm installed.

## AutoGPT Components

### 🧱 AutoGPT Frontend

The user interface for interacting with your AI automation platform:

*   **Agent Builder:** Customize AI agents with a low-code interface.
*   **Workflow Management:** Create and optimize automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Access pre-configured agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The engine that powers your agents, designed for continuous operation:

*   **Source Code:** The core logic of your agents.
*   **Infrastructure:** Robust systems for reliable performance.
*   **Marketplace:** Discover and deploy pre-built agents.

### 🐙 Example Agents

Here are two example use cases:

1.  **Generate Viral Videos:**
    *   Reads trending topics from Reddit.
    *   Creates short-form videos.

2.  **Identify Top Quotes for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Identifies impactful quotes.
    *   Generates social media posts.

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

*   **Get help:** [Discord 💬](https://discord.gg/autogpt)
    [![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)
*   **Report issues:** [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) by the AI Engineer Foundation for standardized communication.

---

## Star History

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