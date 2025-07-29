# AutoGPT: Build, Deploy, and Unleash AI Agents to Automate Workflows

**AutoGPT empowers you to create and manage AI agents that automate complex tasks, revolutionizing how you approach workflows.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1097510977527760986?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   **Deployment and Management:** Control the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Access a library of ready-to-use agents for immediate deployment.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights for continuous improvement.
*   **Self-Hosting:** Host AutoGPT on your own hardware for free!

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own machine.
*   **Cloud-Hosted Beta (Coming Soon!):**  [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted version. (Closed Beta)

## Getting Started: Self-Hosting

**Note:** Self-hosting requires a technical setup. The cloud-hosted option is recommended for ease of use.

### System Requirements

Ensure your system meets these requirements before installing:

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

### Quick Setup (Recommended)

Use the one-line script for a fast setup on your local machine:

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script installs dependencies, configures Docker, and launches your local AutoGPT instance.

### Detailed Setup

For comprehensive setup instructions, refer to the official documentation:

👉 [Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

---
## AutoGPT Platform Overview

The AutoGPT platform consists of two main components:

### 🧱 AutoGPT Frontend

The frontend provides a user-friendly interface to interact with your AI agents. Key features include:

*   **Agent Builder:** Easily design and configure custom AI agents.
*   **Workflow Management:** Build, modify, and optimize your workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Deploy pre-configured agents directly.
*   **Agent Interaction:** Run and interact with your agents.
*   **Monitoring & Analytics:** Track agent performance and gain insights.

Build your own custom blocks: [Custom Blocks Guide](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The AutoGPT Server is the engine that powers your AI agents, including:

*   **Source Code:** Core logic that drives agents and automation processes.
*   **Infrastructure:** Reliable systems ensuring scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

---

## Example Agents: Unleash the Power

Discover the potential of AutoGPT with these examples:

1.  **Generate Viral Videos:** Create short-form videos from trending topics automatically.
2.  **Identify Top Quotes:** Extract and summarize impactful quotes from your videos for social media.

Create customized workflows to build agents for any use case.

---

### License Overview

*   🛡️ **Polyform Shield License:** `autogpt_platform` folder.
    *   [Read more](https://agpt.co/blog/introducing-the-autogpt-platform)
*   🦉 **MIT License:** All other parts of the repository.

---

### Mission: Your Automation Partner

*   🏗️ **Build:** Lay the foundation for something amazing.
*   🧪 **Test:** Fine-tune your agent to perfection.
*   🤝 **Delegate:** Let AI work for you.

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

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

For bug reports or feature requests, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) to ensure compatibility and standards within the AI Engineer Foundation.

---

## Star Statistics
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
```
Key improvements and SEO optimizations:

*   **Headline Optimization:**  Used a compelling, keyword-rich headline.
*   **Concise Introduction:**  Provided a one-sentence hook for immediate understanding.
*   **Clear Headings:**  Organized the content with clear, descriptive headings.
*   **Bulleted Lists:**  Made key features and requirements easy to scan.
*   **Keyword Integration:**  Incorporated relevant keywords like "AI agents," "automation," "workflows," and "self-hosting."
*   **Simplified Language:**  Used clear and concise language for better readability.
*   **Stronger Calls to Action:**  Encouraged users to join the Discord, explore the documentation, and contribute.
*   **Emphasis on Benefits:**  Focused on the value AutoGPT provides to users.
*   **Removed Redundancy:** Streamlined the text to avoid repetition.
*   **Added ALT text to images:** For better SEO
*   **Included the classic autoGPT information**