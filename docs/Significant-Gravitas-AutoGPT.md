# AutoGPT: Unleash the Power of Autonomous AI Agents

**Automate complex tasks and workflows with AutoGPT, a powerful platform for building, deploying, and running AI agents.** ([See the original repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

*   **AI Agent Creation & Deployment:** Build and deploy your own custom AI agents.
*   **Workflow Automation:** Design, modify, and optimize complex automated workflows.
*   **Pre-built Agents:** Leverage a library of ready-to-use AI agents for immediate productivity.
*   **Agent Interaction & Monitoring:** Easily run, interact with, and monitor agent performance.
*   **Open Source & Community Driven:** Join a vibrant community of AI enthusiasts.

## Hosting Options

*   **Self-Hosting:** Download and host AutoGPT yourself (detailed instructions below).
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

## Self-Hosting Setup Guide

**Note:** Setting up and hosting AutoGPT requires technical expertise.  Consider the cloud-hosted beta if you prefer a simpler setup.

### System Requirements

**Hardware Requirements:**

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

**Software Requirements:**

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

**Network Requirements:**

*   Stable internet connection
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

### Installation

For detailed setup instructions, please refer to the official documentation:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git, and npm installed.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The AutoGPT frontend provides the user interface for interacting with your AI agents.

*   **Agent Builder:** Customize AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents from testing to production.
*   **Ready-to-Use Agents:** Select and deploy pre-configured agents instantly.
*   **Agent Interaction:** Run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights.

[Read this guide](https://docs.agpt.co/platform/new_blocks/) to learn how to build your own custom blocks.

### 💽 AutoGPT Server

The AutoGPT Server is the core of the platform where your agents run.

*   **Source Code:** The core logic for agents and automation.
*   **Infrastructure:** Robust systems for reliable and scalable performance.
*   **Marketplace:** Find and deploy a wide variety of pre-built agents.

## Example Agents

See what AutoGPT can do:

1.  **Generate Viral Videos from Trending Topics:**
    *   Reads Reddit for trending topics.
    *   Creates short-form videos automatically.

2.  **Identify Top Quotes from Videos for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Uses AI to find impactful quotes and generate summaries.
    *   Automates social media posting.

## AutoGPT Classic

### 🛠️ Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)

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

## Mission and Licensing

Our mission is to provide the tools, so that you can focus on what matters:

- 🏗️ **Building** - Lay the foundation for something amazing.
- 🧪 **Testing** - Fine-tune your agent to perfection.
- 🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Be part of the revolution! **AutoGPT** is here to stay, at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

**Licensing:**

MIT License: The majority of the AutoGPT repository is under the MIT License.

Polyform Shield License: This license applies to the autogpt_platform folder.

For more information, see https://agpt.co/blog/introducing-the-autogpt-platform

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure someone else hasn't created an issue for the same topic.

## 🤝 Sister projects

### 🔄 Agent Protocol

To maintain a uniform standard and ensure seamless compatibility with many current and future applications, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

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