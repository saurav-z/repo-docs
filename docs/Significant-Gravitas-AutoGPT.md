# AutoGPT: Automate Your World with AI Agents

**Unlock the power of autonomous AI agents to automate complex tasks and revolutionize your workflows with AutoGPT – the leading platform for building, deploying, and running AI agents.**  [Explore the original repository](https://github.com/Significant-Gravitas/AutoGPT)

[![Discord](https://img.shields.io/discord/1098178728057858671?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt) &ensp;
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

## Key Features:

*   **Effortless AI Agent Creation:**  Build custom AI agents tailored to your needs with an intuitive, low-code interface, or use pre-built agents.
*   **Flexible Workflow Management:**  Design, modify, and optimize automation workflows with ease using a visual block-based system.
*   **Simplified Deployment:** Easily manage the lifecycle of your agents, from testing to production with robust deployment controls.
*   **Pre-built Agent Library:** Access a marketplace of ready-to-use agents to get started instantly.
*   **Real-time Agent Interaction:** Run and interact with your agents through a user-friendly interface, whether you've built them or are using pre-configured options.
*   **Performance Monitoring & Analytics:** Track agent performance and gain insights to continually improve your automation processes.

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon):** [Join the Waitlist](https://bit.ly/3ZDijAI) for access to our cloud-hosted beta and experience AutoGPT with minimal setup.

## Self-Hosting: Getting Started

> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. 
> If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

### System Requirements

Before you begin, ensure your system meets these requirements:

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

#### Network Requirements

*   Stable internet connection
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

### Step-by-Step Guide

For detailed instructions, please follow the official self-hosting guide:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

### Quick Setup

For rapid local deployment, utilize the following one-line script:

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script streamlines the process by installing dependencies, configuring Docker, and launching your local instance in one go.

## AutoGPT Platform Components

### 🧱 Frontend

The user interface for interacting with and managing your AI agents, providing tools for:

*   **Agent Builder:**  Create and customize agents using a low-code interface.
*   **Workflow Management:**  Build, modify, and optimize automation workflows.
*   **Deployment Controls:**  Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:**  Quickly deploy pre-configured agents.
*   **Agent Interaction:**  Run and interact with both custom and pre-built agents.
*   **Monitoring and Analytics:**  Track agent performance and gain insights.

[Learn more about building custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 Server

The backend powerhouse where your AI agents run, comprising:

*   **Source Code:**  The core logic driving your agents.
*   **Infrastructure:**  Systems ensuring reliable and scalable performance.
*   **Marketplace:**  A growing marketplace for pre-built agents.

## Example Agents

Harness the potential of AutoGPT with these example use cases:

1.  **Generate Viral Videos from Trending Topics:** Automate video creation based on trending content from Reddit.
2.  **Identify Top Quotes from Videos for Social Media:** Automatically transcribe YouTube videos, identify key quotes, and generate social media posts.

## Licensing

*   **Polyform Shield License:**  All code within the `autogpt_platform` folder. [Read more](https://agpt.co/blog/introducing-the-autogpt-platform)
*   **MIT License:**  The remainder of the AutoGPT repository, including the original AutoGPT Agent and related projects like Forge and the CLI.

## Mission

**Our mission is to empower you to:**

*   🏗️ **Build:** Construct the foundation for innovative AI solutions.
*   🧪 **Test:** Refine your agents to achieve peak performance.
*   🤝 **Delegate:**  Let AI handle the heavy lifting, bringing your ideas to life.

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

---

## Get Help

*   **Join our Community:** [Discord 💬](https://discord.gg/autogpt)

*   **Report Issues:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🤝 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) by the AI Engineer Foundation for standardized communication, ensuring compatibility with a broad ecosystem of applications.

---

## Additional Resources

*   **📖 [Documentation](https://docs.agpt.co)**
*   **🚀 [Contributing](CONTRIBUTING.md)**

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