# AutoGPT: Build, Deploy, and Run AI Agents to Automate Complex Workflows

**Unleash the power of AI by creating, deploying, and managing autonomous agents with AutoGPT, a cutting-edge platform for AI automation.** ([Original Repository](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoGPT empowers you to build, deploy, and manage continuous AI agents, automating intricate tasks and streamlining workflows. This platform provides the tools you need to create intelligent agents capable of independent operation.

## Key Features:

*   **AI Agent Creation & Customization:** Design and configure your own AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Easily build, modify, and optimize automation workflows by connecting blocks.
*   **Deployment & Management:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Quickly leverage ready-to-use agents from our library.
*   **Agent Interaction:** Interact with both custom-built and pre-configured agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights for continuous improvement.

## Hosting Options

*   **Self-Hosting:** Download and set up AutoGPT on your own infrastructure. See below for instructions.
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta to experience AutoGPT without the hassle of self-hosting.

## Self-Hosting Setup

> [!NOTE]
> Self-hosting AutoGPT requires technical expertise. Consider joining the [cloud-hosted beta](https://bit.ly/3ZDijAI) for a simpler experience.

### Hardware Requirements

*   **CPU:** 4+ cores recommended
*   **RAM:** Minimum 8GB, 16GB recommended
*   **Storage:** At least 10GB of free space

### Software Requirements

*   **Operating Systems:**
    *   Linux (Ubuntu 20.04 or newer recommended)
    *   macOS (10.15 or newer)
    *   Windows 10/11 with WSL2
*   **Required Software:**
    *   Docker Engine (20.10.0 or newer)
    *   Docker Compose (2.0.0 or newer)
    *   Git (2.30 or newer)
    *   Node.js (16.x or newer)
    *   npm (8.x or newer)
    *   VSCode (1.60 or newer) or any modern code editor

### Network Requirements

*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Ability to make outbound HTTPS connections

### Updated Setup Instructions:

For the most up-to-date self-hosting instructions, please refer to the official documentation site:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, Git, and npm installed.

## AutoGPT Components

### 🧱 AutoGPT Frontend

The AutoGPT frontend provides the user interface for interacting with and managing AI agents. Key features include:

*   **Agent Builder:** Customize agents using a low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Deploy and use pre-configured agents instantly.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track and improve your automation processes.

Learn how to build your own custom blocks: [Read this guide](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The AutoGPT Server is the core of the platform, powering your agents. Key components include:

*   **Source Code:** The underlying logic driving agents and automation.
*   **Infrastructure:** Reliable systems ensuring scalable performance.
*   **Marketplace:** Discover and deploy a wide range of pre-built agents.

## 🐙 Example Agents

Here are a couple of examples of what you can do with AutoGPT:

1.  **Generate Viral Videos from Trending Topics:**
    *   Reads topics from Reddit.
    *   Identifies trending topics.
    *   Automatically creates a short-form video.

2.  **Identify Top Quotes from Videos for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Uses AI to identify impactful quotes.
    *   Generates a summary and automatically posts it to social media.

## 🚀 AutoGPT Classic

The classic version of AutoGPT, offering additional tools for building and testing your agents.

### 🛠️ Forge

A toolkit for building custom agents, handling boilerplate code so you can focus on innovation. Learn more: [https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)

### 🎯 Benchmark

Measure your agent's performance. The `agbenchmark` is used with any agent that supports the agent protocol. Learn more: [https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

A user-friendly interface for controlling and monitoring your agents, compatible with agents using the agent protocol. Learn more: [https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### ⌨️ CLI

A command-line interface to easily use all the tools offered by the repository.

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

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

Report bugs and request features on [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) for standardized communication.

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