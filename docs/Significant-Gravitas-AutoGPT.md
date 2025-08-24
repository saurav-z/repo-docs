# AutoGPT: Unleash AI Automation for Your World 🚀

**AutoGPT empowers you to build, deploy, and manage AI agents that automate complex tasks, transforming how you work and create.** ([Back to Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1073248236014757918?logo=discord&label=Discord&color=7289da)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)
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

*   **AI Agent Creation:** Design and configure your own AI agents using an intuitive, low-code interface, or use pre-built agents.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Utilize pre-configured agents for immediate deployment.
*   **Agent Interaction:** Easily run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights to improve your automation processes.

## Hosting Options

*   **Self-Host:** Download the platform and host it yourself (Free!).  See instructions below.
*   **Cloud-Hosted Beta:** Join the [waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Coming Soon!).

## Getting Started: Self-Hosting

**Self-hosting AutoGPT requires a bit of technical know-how.** If you prefer a simpler experience, consider joining the cloud-hosted beta (linked above).

### System Requirements

Ensure your system meets these requirements before installation:

**Hardware:**

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB free space

**Software:**

*   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), Windows 10/11 with WSL2
*   Required Software: Docker Engine (20.10.0+), Docker Compose (2.0.0+), Git (2.30+), Node.js (16.x+), npm (8.x+), VSCode (1.60+) or any modern code editor

**Network:**

*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Ability to make outbound HTTPS connections

### Setup Instructions

For detailed and up-to-date instructions, follow the official guide:

👉 [Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

This guide assumes you have Docker, VSCode, git and npm installed.

#### Quick Setup with One-Line Script (Recommended for Local Hosting)

Get up and running in minutes with our automated setup script.

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script installs dependencies, configures Docker, and launches your local instance in one step.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

Interact with and control your AI agents. Key features include:

*   **Agent Builder:** Customize AI agents using a low-code interface.
*   **Workflow Management:** Create, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Choose from a library of pre-configured agents.
*   **Agent Interaction:** Easily run and interact with your agents.
*   **Monitoring and Analytics:** Track agent performance.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The engine that powers your AI agents, enabling continuous operation and external triggers.

*   **Source Code:** The core logic behind the agents.
*   **Infrastructure:** Robust systems ensuring performance and scalability.
*   **Marketplace:** A marketplace to discover and deploy pre-built agents.

## Example Agents: What Can You Build?

1.  **Generate Viral Videos:** Automatically create short-form videos based on trending topics from Reddit.
2.  **Identify Top Quotes from Videos:** Transcribe your YouTube videos, identify impactful quotes, and generate social media posts.

## License & Contributions

*   **Polyform Shield License:**  Code and content within the `autogpt_platform` folder. ([Read more](https://agpt.co/blog/introducing-the-autogpt-platform))
*   **MIT License:**  Other parts of the repository.

### Mission

Our mission is to provide the tools to build, test, and delegate tasks to AI. Join the AI revolution with AutoGPT!

**📖 [Documentation](https://docs.agpt.co)** &ensp;|&ensp; **🚀 [Contributing](CONTRIBUTING.md)**

## AutoGPT Classic 🛠️ (Legacy)

> Information on the classic version of AutoGPT

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

## Get Help and Contribute

### Questions? Problems? Suggestions?

*   Get help on [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

*   Report a bug or request a feature: [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) by the AI Engineer Foundation for seamless compatibility with various applications.

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