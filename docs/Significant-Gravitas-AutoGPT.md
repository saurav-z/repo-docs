# AutoGPT: Unleash the Power of AI Agents 🤖

**AutoGPT empowers you to build, deploy, and manage autonomous AI agents that automate complex tasks, streamlining your workflows and boosting productivity.** [(View Original Repo)](https://github.com/Significant-Gravitas/AutoGPT)

[![Discord](https://img.shields.io/discord/1074442748774436864?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features

*   **AI Agent Creation:** Design and configure your own AI agents with an intuitive, low-code Agent Builder.
*   **Workflow Management:** Build, modify, and optimize your automation workflows with ease using a visual interface.
*   **Deployment & Control:** Manage the lifecycle of your agents, from testing to production with Agent Deployment Controls.
*   **Pre-Built Agents:**  Access a library of ready-to-use agents for instant automation.
*   **Agent Interaction:** Easily run and interact with your agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights to improve your automation.
*   **Open-Source & Self-Hosting:** Take control of your AI with free, self-hosted options.
*   **Continuous Improvement:** Regularly updated with new features, improvements, and integrations.

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own hardware.
*   **Cloud-Hosted Beta (Coming Soon!):**  [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

## Getting Started: Self-Hosting AutoGPT

> [!NOTE]
> Self-hosting requires some technical expertise. For the easiest experience, consider joining the cloud-hosted beta.

### System Requirements

Ensure your system meets these requirements:

#### Hardware
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

#### Software
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

#### Network
*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Outbound HTTPS connections

### Setup Instructions

For detailed setup instructions, please refer to the official documentation:  [Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

#### ⚡ Quick Setup (Recommended)

Automate the setup process with a single command:

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```
*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

This script installs dependencies, configures Docker, and launches your local instance.

## AutoGPT Platform Components

### 🧱 Frontend

The user interface for interacting with and controlling your AI agents.

*   **Agent Builder:** Create custom agents with a low-code interface.
*   **Workflow Management:** Build and optimize agent workflows visually.
*   **Deployment:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Quickly deploy pre-built agents.
*   **Interaction:** Run and test agents.
*   **Monitoring:** Track performance and gain insights.

Learn how to build your own custom blocks: [Build Custom Blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 Server

The core of the platform where your agents run.

*   **Source Code:** The core logic of the agents.
*   **Infrastructure:** Reliable systems for scalability.
*   **Marketplace:** Discover and deploy pre-built agents.

### 🐙 Example Agents

Automate tasks with these examples:

1.  **Generate Viral Videos:** Create short-form videos from trending topics on Reddit.
2.  **Extract Top Quotes:** Identify impactful quotes from videos for social media.

Customize workflows to build agents for any use case.

---

## License Overview

*   🛡️ **Polyform Shield License:** Code within the `autogpt_platform` folder. [Read more about this effort](https://agpt.co/blog/introducing-the-autogpt-platform)
*   🦉 **MIT License:** Other parts of the repository, including the original AutoGPT Agent and related projects like Forge, agbenchmark, and the AutoGPT Classic GUI. Also used for projects like [GravitasML](https://github.com/Significant-Gravitas/gravitasml) and [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability).

---

### Mission

Our mission is to empower you to:

*   🏗️ **Build** your AI agents.
*   🧪 **Test** your agents for peak performance.
*   🤝 **Delegate** tasks to AI.

Join the AI revolution!  **AutoGPT** is leading the way in AI innovation.

**📖 [Documentation](https://docs.agpt.co)** &ensp;|&ensp; **🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic (Legacy)

> Below is information about the classic version of AutoGPT.

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

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

Report issues or suggest features via [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) for standardized communication, ensuring compatibility with various applications.

---

## Stars Stats

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