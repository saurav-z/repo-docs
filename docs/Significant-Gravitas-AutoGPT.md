# AutoGPT: Build, Deploy, and Automate with AI Agents

**Supercharge your workflows and automate complex tasks with AutoGPT, the cutting-edge platform for creating, deploying, and managing AI agents.** ([Original Repository](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoGPT empowers you to build, deploy, and manage AI agents that automate tasks and transform your workflow.

## Key Features:

*   **AI Agent Creation & Customization**: Design and configure your own AI agents with an intuitive, low-code Agent Builder.
*   **Workflow Management**: Build, modify, and optimize your automation workflows with ease.
*   **Deployment Controls**: Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents**: Choose from a library of pre-configured agents.
*   **Agent Interaction**: Run and interact with your custom or pre-configured agents through a user-friendly interface.
*   **Monitoring and Analytics**: Track agent performance and gain insights.

## Hosting Options

*   **Self-Hosting**: Download and host the platform yourself. (See below for setup instructions.)
*   **Cloud-Hosted Beta (Coming Soon)**: [Join the Waitlist](https://bit.ly/3ZDijAI)

## Self-Hosting AutoGPT

> [!NOTE]
> For a hassle-free experience, consider joining the cloud-hosted beta.

Setting up AutoGPT yourself is a technical process.

### Prerequisites:

#### Hardware Requirements
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

#### Software Requirements
*   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), or Windows 10/11 with WSL2
*   Required Software (with minimum versions): Docker Engine (20.10.0 or newer), Docker Compose (2.0.0 or newer), Git (2.30 or newer), Node.js (16.x or newer), npm (8.x or newer), VSCode (1.60 or newer) or any modern code editor

#### Network Requirements
*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Outbound HTTPS connections

### Setup Instructions:

[Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git, and npm installed.

## AutoGPT Platform Components

### 🧱 Frontend

The AutoGPT frontend provides the interface for interacting with and managing AI agents:

*   **Agent Builder:** Customize and configure your own AI agents through a low-code interface.
*   **Workflow Management:** Easily create, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage your agent's lifecycle.
*   **Ready-to-Use Agents:** Utilize pre-configured agents for immediate use.
*   **Agent Interaction:** Interact with your agents through the user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance for continuous improvement.

### 💽 Server

The AutoGPT Server houses the core components for agent operation:

*   **Source Code**: Contains the core logic that drives agents and automation processes.
*   **Infrastructure**: Provides reliable and scalable performance.
*   **Marketplace**: Discover and deploy a range of pre-built agents.

## 🐙 Example Agents

Here are two examples to illustrate AutoGPT's capabilities:

1.  **Generate Viral Videos from Trending Topics**:
    *   Reads topics on Reddit.
    *   Identifies trending topics.
    *   Automatically creates short-form videos.

2.  **Identify Top Quotes from Videos for Social Media**:
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Identifies impactful quotes for summaries.
    *   Writes and publishes social media posts.

## 🤖 AutoGPT Classic

Below is information about the classic version of AutoGPT.

### 🛠️ Build your own Agent - Quickstart

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

Report bugs or request features via [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure there isn't an existing issue for your topic.

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) to standardize communication pathways for seamless compatibility.

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