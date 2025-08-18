# AutoGPT: Unleash the Power of AI Automation

**Automate complex workflows and build intelligent AI agents with AutoGPT, a leading platform for AI innovation.** ([View on GitHub](https://github.com/Significant-Gravitas/AutoGPT))

## Key Features

*   **AI Agent Creation:** Design, build, and customize your own AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Easily build, modify, and optimize your automation workflows.
*   **Deployment & Management:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Utilize a library of ready-to-use, pre-configured agents for immediate deployment.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights to improve automation processes.
*   **Self-Hosting Options:** Host AutoGPT on your own hardware for free! Or, join the waitlist for the cloud-hosted beta.

## Getting Started

### Hosting Options

*   **Self-Host:** Download and host AutoGPT for free! (See the instructions below).
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for early access to the cloud-hosted platform.

### Self-Hosting Guide

> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

#### 1. System Requirements

Ensure your system meets these minimum requirements before proceeding:

*   **Hardware:**
    *   CPU: 4+ cores recommended
    *   RAM: Minimum 8GB, 16GB recommended
    *   Storage: At least 10GB of free space
*   **Software:**
    *   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), Windows 10/11 with WSL2
    *   Required Software (with minimum versions): Docker Engine (20.10.0 or newer), Docker Compose (2.0.0 or newer), Git (2.30 or newer), Node.js (16.x or newer), npm (8.x or newer), VSCode (1.60 or newer) or any modern code editor
*   **Network:**
    *   Stable internet connection
    *   Access to required ports (configured in Docker)
    *   Ability to make outbound HTTPS connections

#### 2. Setup Instructions

We've moved to a fully maintained and regularly updated documentation site.

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

#### 3. Quick Setup

Simplify the setup process with our one-line script, which automatically installs dependencies and configures Docker.

*   **macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

*   **Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user interface for building and interacting with AI agents.

*   **Agent Builder:** Low-code interface to design and configure agents.
*   **Workflow Management:** Create, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with built or pre-configured agents.
*   **Monitoring and Analytics:** Monitor agent performance and gain insights.

Learn how to build custom blocks: [Build Custom Blocks Guide](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The core engine that runs your AI agents.

*   **Source Code:** Drives agents and automations.
*   **Infrastructure:** Provides reliable, scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

### 🐙 Example Agents

Discover the power of AutoGPT with these examples:

1.  **Generate Viral Videos from Trending Topics:** Reads topics on Reddit, identifies trends, and generates short-form videos.
2.  **Identify Top Quotes from Videos for Social Media:** Transcribes YouTube videos, identifies impactful quotes, and generates social media posts.

## AutoGPT Classic

Explore the classic version of AutoGPT, and the tools that make it great.

### 🛠️ Forge

**Build your own agent!** - Forge is a ready-to-go toolkit to build your own agent application.

*   **Getting Started with Forge** [Forge Quickstart](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)
*   **Learn More:** [Forge Docs](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### 🎯 Benchmark

**Measure your agent's performance!** - The `agbenchmark` can be used with any agent that supports the agent protocol

*   **Pypi:** [`agbenchmark`](https://pypi.org/project/agbenchmark/)
*   **Learn More:** [Benchmark Docs](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

**Makes agents easy to use!** - The `frontend` gives you a user-friendly interface to control and monitor your agents.

*   **Learn More:** [UI Docs](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

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

## License Information

*   **Polyform Shield License:** Code and content within the `autogpt_platform` folder.
*   **MIT License:** All other parts of the repository (including AutoGPT Classic and related projects).

## Get Involved

*   **Documentation:** [Documentation](https://docs.agpt.co)
*   **Contribute:** [Contributing](CONTRIBUTING.md)
*   **Discord:** [Discord 💬](https://discord.gg/autogpt)
*   **Report Issues:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

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