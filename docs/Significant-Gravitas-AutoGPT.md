# AutoGPT: Build, Deploy, and Run AI Agents to Automate Anything

**AutoGPT empowers you to create, deploy, and manage powerful AI agents that automate complex tasks, saving you time and increasing productivity. Check out the original repo [here](https://github.com/Significant-Gravitas/AutoGPT)!**

[![Discord](https://img.shields.io/discord/1098785029270672936?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows with ease.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Utilize pre-configured agents from our library for immediate use.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.
*   **Automated Setup:** Get started quickly with a one-line setup script.

## Hosting Options

*   **Self-Hosting:** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) to try our cloud-hosted beta version.

## Getting Started with Self-Hosting

> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

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

### Detailed Setup Instructions

For comprehensive setup instructions, please refer to our updated and regularly maintained documentation site:

👉 [Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

### Quick Setup (Recommended)

Skip the manual steps and get started in minutes using our automatic setup script.

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automates the installation of dependencies, configures Docker, and launches your local AutoGPT instance.

## Core Components

### 🧱 AutoGPT Frontend

The user interface for interacting with and leveraging your AI agents. Features include:

*   **Agent Builder:** Build custom agents with a low-code interface.
*   **Workflow Management:** Design, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage agent lifecycles, from testing to production.
*   **Ready-to-Use Agents:** Utilize pre-configured agents.
*   **Agent Interaction:** Run and interact with agents via a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance.

[Learn how to build your own custom blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The core of the platform, where your agents run.

*   **Source Code:** The core logic that drives agents and automation.
*   **Infrastructure:** Robust systems for reliable and scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

## Example Agents

1.  **Generate Viral Videos:**
    *   Reads Reddit topics.
    *   Identifies trending topics.
    *   Creates short-form videos.
2.  **Identify Top Quotes from Videos for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Identifies impactful quotes for summaries.
    *   Writes social media posts.

## AutoGPT Classic (Legacy Version)

### 🛠️ Forge

*   **Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### 🎯 Benchmark

*   **Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### 💻 UI

*   **Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend) about the Frontend

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

## Mission and Licensing

Our mission is to provide the tools for building, testing, and delegating your ideas to AI.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

**Licensing:**

*   MIT License (majority of the repository)
*   Polyform Shield License (autogpt\_platform folder)
*   See [https://agpt.co/blog/introducing-the-autogpt-platform](https://agpt.co/blog/introducing-the-autogpt-platform) for more information.

## Agent Protocol

AutoGPT adheres to the [Agent Protocol](https://agentprotocol.ai/) standard.

## Get Help

*   **Discord:** [Join our Discord](https://discord.gg/autogpt) for support and discussions.
*   **GitHub Issues:** Report bugs and request features by creating a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

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

## Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>