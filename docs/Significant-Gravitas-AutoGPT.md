# AutoGPT: The Future of AI Automation

**Automate complex workflows and build intelligent agents with AutoGPT, the leading open-source platform for AI-powered automation.** ([Original Repository](https://github.com/Significant-Gravitas/AutoGPT))

AutoGPT empowers you to build, deploy, and manage AI agents that streamline your tasks and enhance productivity. Whether you're a developer or a business user, AutoGPT offers the tools you need to harness the power of AI.

## Key Features:

*   **Agent Builder:** Design and configure custom AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Easily build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Leverage a library of pre-configured agents for immediate automation.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights to improve your automation processes.
*   **Open-Source Flexibility:** Benefit from a thriving community and customizable options.
*   **Self-Hosting:** Host AutoGPT locally for free or utilize the cloud-hosted beta (coming soon).

## Hosting Options:

*   **Self-Host:** Download and run AutoGPT on your own hardware (Free!). See instructions below.
*   **Cloud-Hosted Beta:** Join the waitlist for the cloud-hosted beta for a managed experience (Coming Soon!). [Join the Waitlist](https://bit.ly/3ZDijAI)

## Getting Started with Self-Hosting

Setting up and hosting the AutoGPT Platform yourself is a technical process.

If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

### System Requirements

Ensure your system meets the following requirements before installation:

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

### Updated Setup Instructions

We've moved to a fully maintained and regularly updated documentation site.

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

---

#### ⚡ Quick Setup with One-Line Script (Recommended for Local Hosting)

Skip the manual steps and get started in minutes using our automatic setup script.

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This will install dependencies, configure Docker, and launch your local instance — all in one go.

## 🧱 AutoGPT Platform Overview

### 🧱 AutoGPT Frontend

The AutoGPT frontend provides a user-friendly interface for interacting with the AI automation platform.

*   **Agent Builder:** Customize AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with your agents.
*   **Monitoring and Analytics:** Track agent performance.

[Read this guide](https://docs.agpt.co/platform/new_blocks/) to learn how to build your own custom blocks.

### 💽 AutoGPT Server

The AutoGPT Server powers the platform and runs your agents.

*   **Source Code:** Drives the agents and automation processes.
*   **Infrastructure:** Ensures reliable and scalable performance.
*   **Marketplace:** Find and deploy a wide range of pre-built agents.

## 🐙 Example Agents

Here are two examples of what you can do with AutoGPT:

1.  **Generate Viral Videos from Trending Topics**
    *   Reads topics on Reddit.
    *   Identifies trending topics.
    *   Creates short-form videos.
2.  **Identify Top Quotes from Videos for Social Media**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Identifies impactful quotes.
    *   Generates a summary for social media.

## 📜 License Overview

*   🛡️ **Polyform Shield License:**  All code and content within the `autogpt_platform` folder is licensed under the Polyform Shield License.
    _[Read more about this effort](https://agpt.co/blog/introducing-the-autogpt-platform)_
*   🦉 **MIT License:** All other portions of the AutoGPT repository (i.e., everything outside the `autogpt_platform` folder) are licensed under the MIT License. This includes projects like Forge, agbenchmark, and the AutoGPT Classic GUI.

## 🤖 AutoGPT Classic

This section contains information about the classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

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

*   **Get help:** [Discord 💬](https://discord.gg/autogpt)
    [![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)
*   **Report issues or suggest features:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🤝 Sister Projects

*   **🔄 Agent Protocol:** AutoGPT uses the [agent protocol](https://agentprotocol.ai/) by the AI Engineer Foundation to standardize communication.

---

## Contributors & Stats

*(Include these sections if you want - they can be a nice visual.)*

## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>