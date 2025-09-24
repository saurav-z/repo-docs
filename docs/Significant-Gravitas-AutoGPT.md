# AutoGPT: Unleash the Power of AI Agents 🚀

**Build, deploy, and manage AI agents that automate complex tasks with AutoGPT, the innovative platform for continuous AI automation.**  [Visit the Original Repo](https://github.com/Significant-Gravitas/AutoGPT)

[![Discord](https://img.shields.io/discord/1092888266930801172?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features

*   **Autonomous AI Agents:** Create and deploy AI agents capable of independent operation.
*   **Agent Builder:**  Easily design and customize your own AI agents using a low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Pre-built Agents:** Leverage a library of ready-to-use, pre-configured agents.
*   **Agent Interaction:**  A user-friendly interface allows you to run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.
*   **Open Source and Flexible:** Designed for self-hosting and cloud deployment, with options for customizability.
*   **Extensive Documentation and Community:**  Benefit from comprehensive documentation and a supportive community on Discord.

## Table of Contents
*   [Hosting Options](#hosting-options)
*   [How to Self-Host](#how-to-self-host-the-autogpt-platform)
    *   [System Requirements](#system-requirements)
    *   [Quick Setup](#quick-setup-with-one-line-script)
*   [AutoGPT Platform Components](#autogpt-platform-components)
    *   [AutoGPT Frontend](#autogpt-frontend)
    *   [AutoGPT Server](#autogpt-server)
*   [Example Agents](#example-agents)
*   [AutoGPT Classic](#autogpt-classic)
    *   [Forge](#forge)
    *   [Benchmark](#benchmark)
    *   [UI](#ui)
    *   [CLI](#cli)
*   [Get Help](#questions-problems-suggestions)
*   [Sister Projects](#sister-projects)
*   [Contributors](#contributors)
*   [Stars Stats](#stars-stats)

## Hosting Options

*   **Self-Host (Free!):** Download and set up AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon!):** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

## How to Self-Host the AutoGPT Platform

> [!NOTE]
> Self-hosting AutoGPT requires technical expertise.  For ease of use, consider the cloud-hosted beta.

### System Requirements

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

### Updated Setup Instructions:

Refer to the fully maintained and regularly updated documentation site for setup guidance.

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

#### Quick Setup with One-Line Script

(Recommended for Local Hosting)

Simplify setup with an automated script.

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script handles dependency installation, Docker configuration, and local instance launch.

## AutoGPT Platform Components

### AutoGPT Frontend

The user interface for interacting with and leveraging the AI automation platform.

*   **Agent Builder:**  Design and configure your own AI agents.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:**  Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with agents through an easy-to-use interface.
*   **Monitoring and Analytics:**  Track agent performance.

[Learn how to build your own custom blocks.](https://docs.agpt.co/platform/new_blocks/)

### AutoGPT Server

The core of the platform where your agents run and operate continuously.

*   **Source Code:** The core logic that drives our agents.
*   **Infrastructure:** Systems that ensure reliable and scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

## Example Agents

Demonstrations of AutoGPT's capabilities:

1.  **Generate Viral Videos from Trending Topics:**
    *   Reads topics on Reddit.
    *   Identifies trending topics.
    *   Creates short-form videos based on the content.

2.  **Identify Top Quotes from Videos for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Identifies impactful quotes to generate summaries.
    *   Writes posts for social media.

## AutoGPT Classic

Information about the classic version of AutoGPT.

### Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

<!-- TODO: insert visual demonstrating the benchmark -->

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

<!-- TODO: insert screenshot of front end -->

The frontend works out-of-the-box with all agents in the repo. Just use the [CLI] to run your agent of choice!

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend) about the Frontend

### CLI

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

## Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

For bugs or feature requests, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).  Check if a similar issue exists.

## Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation for seamless compatibility.

---
## Contributors

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