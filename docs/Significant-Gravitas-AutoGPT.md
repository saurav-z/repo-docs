# AutoGPT: Build, Deploy, and Run Powerful AI Agents

**Automate complex tasks and revolutionize your workflows with AutoGPT, the leading platform for creating and managing autonomous AI agents. ([See the original repo](https://github.com/Significant-Gravitas/AutoGPT))**

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features

*   **AI Agent Creation:** Design and configure AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents for immediate use.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.

## Hosting Options

*   **Self-Hosting:** Download and install AutoGPT on your own infrastructure. Detailed instructions are provided below.
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) to access the cloud-hosted beta version for a simplified experience.

## Getting Started with Self-Hosting

**Important Note:** Self-hosting AutoGPT requires technical skills. The cloud-hosted beta is recommended for ease of use.

### System Requirements

Ensure your system meets these requirements before installation:

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

### Installation

For detailed self-hosting instructions, follow the official guide: [AutoGPT Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

**Quick Setup (Recommended for Local Hosting)**

Use this one-line script to quickly set up AutoGPT:

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automates dependency installation, Docker configuration, and local instance launch.

## AutoGPT Components

### 🧱 AutoGPT Frontend

The frontend provides a user-friendly interface for interacting with AI agents. Key features include:

*   Agent Builder
*   Workflow Management
*   Deployment Controls
*   Ready-to-Use Agents
*   Agent Interaction
*   Monitoring and Analytics

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The server is the engine that powers your AI agents. It includes:

*   Source Code
*   Infrastructure
*   Marketplace

### 🐙 Example Agents

Explore the possibilities with these example agents:

1.  **Generate Viral Videos from Trending Topics:**  Identifies trending topics, and creates short-form videos.
2.  **Identify Top Quotes from Videos for Social Media:**  Transcribes videos, identifies impactful quotes, and publishes social media posts.

Create custom workflows to build agents for any use case.

## 🤖 AutoGPT Classic (Legacy Version)

**Note:** The following information pertains to the classic version of AutoGPT.

### 🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)

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

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) to standardize communication, ensuring compatibility across various applications.

## 🤔 Questions? Problems? Suggestions?

*   Get help on [Discord 💬](https://discord.gg/autogpt)
    [![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)
*   Report bugs and request features via [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## Licensing and Resources

*   **Documentation:** [Documentation](https://docs.agpt.co)
*   **Contributing:** [Contributing](CONTRIBUTING.md)
*   **Licensing:** MIT License (main repository), Polyform Shield License (autogpt\_platform folder). See [AutoGPT Platform Blog](https://agpt.co/blog/introducing-the-autogpt-platform) for details.

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