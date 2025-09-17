# AutoGPT: Automate Your World with Autonomous AI Agents

**Unleash the power of AI by building, deploying, and running autonomous agents with AutoGPT, streamlining complex tasks and boosting productivity.** ([Back to the original repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1092625192975070238?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://zdoc.app/de/Significant-Gravitas/AutoGPT) | 
[Español](https://zdoc.app/es/Significant-Gravitas/AutoGPT) | 
[français](https://zdoc.app/fr/Significant-Gravitas/AutoGPT) | 
[日本語](https://zdoc.app/ja/Significant-Gravitas/AutoGPT) | 
[한국어](https://zdoc.app/ko/Significant-Gravitas/AutoGPT) | 
[Português](https://zdoc.app/pt/Significant-Gravitas/AutoGPT) | 
[Русский](https://zdoc.app/ru/Significant-Gravitas/AutoGPT) | 
[中文](https://zdoc.app/zh/Significant-Gravitas/AutoGPT)

## Key Features

*   **Create & Deploy AI Agents:** Design and deploy AI agents tailored to your specific needs.
*   **Automate Workflows:** Automate complex tasks and processes with ease.
*   **User-Friendly Interface:** Interact with your agents through a simple and intuitive interface.
*   **Pre-Built Agent Library:** Get started quickly with a selection of ready-to-use agents.
*   **Monitoring & Analytics:** Track agent performance and optimize your automation processes.
*   **Flexible Hosting Options:** Choose to self-host for free or join the cloud-hosted beta.

## Hosting Options

*   **Self-Host:** Download and set up AutoGPT on your own hardware (Free!).
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Coming Soon!).

## Getting Started: Self-Hosting Guide

> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. 
> If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

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
*   Access to required ports (configured in Docker)
*   Ability to make outbound HTTPS connections

### Updated Setup Instructions

For the most up-to-date and comprehensive instructions, please refer to the official documentation:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

### Quick Setup (Recommended)

Get up and running in minutes with our automated setup script:

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```
*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

This script will install dependencies, configure Docker, and launch your local instance.

## AutoGPT Frontend: Your AI Automation Control Center

The AutoGPT frontend is your gateway to interacting with and utilizing AI automation.

*   **Agent Builder:** Design and configure your own AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Select from our library of pre-configured agents to start automating immediately.
*   **Agent Interaction:** Run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance to improve your processes.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/) to customize your agents.

## AutoGPT Server: The Engine of Automation

The AutoGPT Server is the core of the platform, powering your AI agents.

*   **Source Code:** The underlying logic that drives your agents.
*   **Infrastructure:** Robust systems ensuring reliable and scalable performance.
*   **Marketplace:** Find and deploy a variety of pre-built agents.

## Example Agents: Unleash the Possibilities

Here are examples of what AutoGPT can do:

1.  **Viral Video Generator:** Reads trending topics, creates short-form videos.
2.  **Social Media Quote Curator:** Transcribes your videos, identifies impactful quotes, and generates social media posts.

You can create custom workflows to build agents for any use case.

---

## License

*   **Polyform Shield License:** All code and content within the `autogpt_platform` folder. Read more: [AGPT Platform Blog](https://agpt.co/blog/introducing-the-autogpt-platform)
*   **MIT License:** All other parts of the AutoGPT repository. Includes projects such as Forge, agbenchmark, and the AutoGPT Classic GUI. Also includes the [GravitasML](https://github.com/Significant-Gravitas/gravitasml) and [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability) projects.

---

### Mission

Our mission is to empower you to:

*   🏗️ **Build** - Create amazing solutions.
*   🧪 **Test** - Refine your agents.
*   🤝 **Delegate** - Let AI work for you.

Join the AI revolution!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic

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

## 🤔 Questions, Problems, or Suggestions?

### Get Help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

Report bugs or request features via a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) for compatibility.

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