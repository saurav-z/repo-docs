# AutoGPT: Unleash the Power of AI Agents to Automate Workflows

**AutoGPT** empowers you to build, deploy, and manage AI agents that revolutionize how you automate tasks.  [Explore the original repo](https://github.com/Significant-Gravitas/AutoGPT)!

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features:

*   **AI Agent Creation:** Design and configure custom AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   **Agent Deployment & Management:** Deploy, test, and manage the lifecycle of your AI agents.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents for immediate use.
*   **Agent Interaction:** Easily run and interact with your custom or pre-built agents.
*   **Monitoring & Analytics:** Track agent performance and gain insights for improvement.

## Hosting Options:

*   **Self-Hosting:** Download and host AutoGPT yourself (see setup instructions below).
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

## Getting Started with Self-Hosting:

> [!NOTE]
> Self-hosting AutoGPT requires technical expertise. Consider the cloud-hosted beta for an easier experience.

### System Requirements:

Ensure your system meets these requirements before installation:

#### Hardware Requirements:

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

#### Software Requirements:

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

#### Network Requirements:

*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Ability to make outbound HTTPS connections

### Setup Instructions:

For the most up-to-date self-hosting instructions, please refer to the official documentation:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

## AutoGPT Platform Components:

### 🧱 Frontend:

The frontend provides the user interface for interacting with your AI agents. Key features include:

*   **Agent Builder:** Design and customize AI agents.
*   **Workflow Management:**  Build and modify automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Access pre-configured agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance.

[Learn how to build your own blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 Server:

The server is the core engine where your AI agents execute. It includes:

*   **Source Code:** Core logic driving agents and automation.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

### 🐙 Example Agents:

Discover the possibilities with these examples:

1.  **Generate Viral Videos:** Create short-form videos from trending topics.
2.  **Identify Top Quotes:** Extract and summarize impactful quotes from videos for social media.

## 🤖 AutoGPT Classic:

### 🏗️ Forge

**Forge your own agent!**  Forge is a ready-to-go toolkit to build your own agent application.
*   🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) 
*   📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### 🎯 Benchmark

**Measure your agent's performance!**
*   📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
*   📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

**Makes agents easy to use!**
*   📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

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

Report bugs or request features via [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard for seamless compatibility.

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