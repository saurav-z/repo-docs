# AutoGPT: Build, Deploy, and Run AI Agents to Automate Complex Workflows

**Supercharge your productivity with AutoGPT, the open-source platform for creating and deploying autonomous AI agents.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoGPT empowers you to design, build, and manage AI agents that automate tasks, streamline workflows, and unlock new levels of efficiency. Whether you're a seasoned developer or just starting, AutoGPT offers the tools and resources you need to bring your AI automation ideas to life.

## Key Features of AutoGPT:

*   **AI Agent Creation:** Design and configure custom AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease through a visual block-based system.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Explore a library of pre-configured agents for immediate productivity gains.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.

## Getting Started with AutoGPT

### Hosting Options
*   **Self-Hosting:** Download and install AutoGPT for complete control. See setup instructions below.
*   **Cloud-Hosted Beta:**  [Join the Waitlist](https://bit.ly/3ZDijAI) for access to the cloud-hosted beta version for a streamlined experience.

### Self-Hosting Setup
> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. 
> If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

**Prerequisites:** Ensure your system meets the requirements below. For detailed instructions, consult the [official self-hosting guide](https://docs.agpt.co/platform/getting-started/).

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

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The frontend provides the user interface for interacting with and managing your AI agents.  Key features include:

*   **Agent Builder:** A low-code interface for designing and configuring AI agents.
*   **Workflow Management:** A visual, block-based system for building and optimizing workflows.
*   **Deployment Controls:** Tools for managing the agent lifecycle, from testing to production.
*   **Ready-to-Use Agents:** A library of pre-configured agents for immediate use.
*   **Agent Interaction:** A user-friendly interface for running and interacting with agents.
*   **Monitoring and Analytics:** Tools to track agent performance and improve automation processes.

Learn more about building custom blocks: [Build Custom Blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The server is the core engine where your agents run, powered by:

*   **Source Code:** The foundational logic that drives agents and automation.
*   **Infrastructure:** Robust systems for reliable and scalable performance.
*   **Marketplace:** A marketplace to discover and deploy pre-built agents.

### 🐙 Example Agents

Here are examples of AutoGPT's capabilities:

1.  **Generate Viral Videos from Trending Topics:** This agent identifies trending topics on Reddit and automatically creates short-form videos.
2.  **Identify Top Quotes from Videos for Social Media:** This agent transcribes your YouTube videos, uses AI to find the best quotes, and generates social media posts.

## 🤖 AutoGPT Classic

AutoGPT Classic offers tools for building and measuring agent performance.

### 🏗️ Forge

Forge provides a toolkit for building your agent applications.

*   🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)
*   📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### 🎯 Benchmark

The `agbenchmark` measures agent performance.

*   📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
*   📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

The `frontend` provides a user-friendly interface for controlling and monitoring agents.

*   📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### ⌨️ CLI

The CLI simplifies using the tools offered by the repository.

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

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT utilizes the [agent protocol](https://agentprotocol.ai/) standard to ensure seamless compatibility.

---

## Get Involved

*   **Documentation:** [Documentation](https://docs.agpt.co)
*   **Contributing:** [Contributing](CONTRIBUTING.md)
*   **Get Help:** [Discord 💬](https://discord.gg/autogpt)
*   **Report Issues:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

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