# AutoGPT: Unleash the Power of AI Automation

**Automate complex workflows and build intelligent agents with AutoGPT, a cutting-edge platform for AI-driven automation. Discover more and contribute at the original repository: [https://github.com/Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)**

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features of AutoGPT

*   **AI Agent Creation:** Design and configure custom AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease using a visual block-based system.
*   **Agent Deployment & Management:** Manage the full lifecycle of your AI agents, from testing to production.
*   **Pre-Built Agents:** Leverage a library of ready-to-use, pre-configured agents for immediate automation.
*   **Agent Interaction:**  Easily run and interact with your custom or pre-configured agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement of your automation processes.
*   **Open Source:** Build with a project that is at the forefront of AI innovation.
*   **Continuous Development:** Benefit from a regularly updated platform.

## Getting Started with AutoGPT

AutoGPT offers flexible hosting options to meet your needs.  You can choose to self-host the platform, or join the waitlist for the cloud-hosted beta for an easier, managed experience.

### Self-Hosting Requirements

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

For detailed self-hosting instructions, please follow the [official self-hosting guide](https://docs.agpt.co/platform/getting-started/).

#### Quick Setup (Recommended)

Get started in minutes with a one-line script:

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```
*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

## AutoGPT Frontend

The frontend provides the user interface for interacting with your AI agents.  Key features include:

*   **Agent Builder:** Create custom agents using a low-code interface.
*   **Workflow Management:** Build and modify automation workflows visually.
*   **Deployment Controls:** Manage the deployment and lifecycle of your agents.
*   **Ready-to-Use Agents:** Access and deploy pre-configured agents instantly.
*   **Agent Interaction:** Run and interact with agents through an intuitive interface.
*   **Monitoring & Analytics:** Track agent performance and gain actionable insights.

Learn more about building custom blocks:  [https://docs.agpt.co/platform/new_blocks/](https://docs.agpt.co/platform/new_blocks/)

## AutoGPT Server

The AutoGPT Server is the core engine where your AI agents execute.  It's designed for robust, scalable performance.

*   **Source Code:** The core logic that drives our agents and automation processes.
*   **Infrastructure:** Robust systems that ensure reliable and scalable performance.
*   **Marketplace:** A comprehensive marketplace where you can find and deploy a wide range of pre-built agents.

## Example AI Agents

**1.  Generate Viral Videos:**
    *   Reads trending topics from Reddit.
    *   Identifies trending topics.
    *   Automatically creates short-form videos based on the content.

**2.  Identify Top Quotes for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes new video uploads.
    *   Uses AI to identify impactful quotes.
    *   Generates social media posts with summaries.

## Mission & Licensing

Our mission is to empower you to:

*   🏗️ **Build** - Lay the foundation for amazing AI solutions.
*   🧪 **Test** - Fine-tune your agents to perfection.
*   🤝 **Delegate** - Let AI work for you, and have your ideas come to life.

**AutoGPT** is at the forefront of AI innovation, empowering you to create cutting-edge automations.

*   📖 [Documentation](https://docs.agpt.co)
*   🚀 [Contributing](CONTRIBUTING.md)

**Licensing:**

*   MIT License:  Applies to the majority of the AutoGPT repository.
*   Polyform Shield License:  Applies to the `autogpt_platform` folder.

For more details: [https://agpt.co/blog/introducing-the-autogpt-platform](https://agpt.co/blog/introducing-the-autogpt-platform)

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

## 🤔 Questions? Problems? Suggestions?

*   **Get Help:** [Discord 💬](https://discord.gg/autogpt)

    [![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

*   **Report Issues:** Create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT utilizes the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation for seamless integration and compatibility.

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