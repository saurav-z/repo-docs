# AutoGPT: Automate Anything with AI Agents

**Unleash the power of AI with AutoGPT, a revolutionary platform for building, deploying, and managing autonomous AI agents that transform complex workflows.**  [Explore the original repo](https://github.com/Significant-Gravitas/AutoGPT).

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features of AutoGPT:

*   **Autonomous AI Agents:** Create and deploy agents capable of performing complex tasks with minimal human intervention.
*   **Agent Builder:** Design and customize AI agents through an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Quickly get started with a library of pre-configured agents.
*   **Agent Interaction:** Easily run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights to improve your automation processes.

## Hosting Options:

*   **Self-Hosting (Free!):** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta:** Join the waitlist for our cloud-hosted beta (coming soon!). [Join the Waitlist](https://bit.ly/3ZDijAI)

## Setting up AutoGPT:

### System Requirements

Ensure your system meets these requirements for optimal performance:

#### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: Minimum 8GB, 16GB recommended
- Storage: At least 10GB of free space

#### Software Requirements
- Operating Systems:
  - Linux (Ubuntu 20.04 or newer recommended)
  - macOS (10.15 or newer)
  - Windows 10/11 with WSL2
- Required Software (with minimum versions):
  - Docker Engine (20.10.0 or newer)
  - Docker Compose (2.0.0 or newer)
  - Git (2.30 or newer)
  - Node.js (16.x or newer)
  - npm (8.x or newer)
  - VSCode (1.60 or newer) or any modern code editor

#### Network Requirements
- Stable internet connection
- Access to required ports (will be configured in Docker)
- Ability to make outbound HTTPS connections

### Installation

Follow these updated instructions for setting up AutoGPT:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

### Quick Setup with One-Line Script (Recommended for Local Hosting)

Simplify your setup with our automatic script.

For macOS/Linux:
```
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script handles dependencies, configures Docker, and launches your instance in one go.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

Interact with your AI agents using the intuitive AutoGPT frontend:

*   **Agent Builder:** Design and customize your agents.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Access and deploy pre-configured agents.
*   **Agent Interaction:** Run and interact with your agents.
*   **Monitoring and Analytics:** Track agent performance and gain insights.

### 💽 AutoGPT Server

The powerhouse of the AutoGPT platform, housing:

*   **Source Code:** The core logic that drives our agents.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** Discover and deploy a wide array of pre-built agents.

## Example Agents

Explore the possibilities with these example agents:

1.  **Generate Viral Videos:** Create short-form videos from trending topics.
2.  **Identify Top Quotes:** Extract and summarize key quotes from videos for social media.

## AutoGPT Classic

> Below is information about the classic version of AutoGPT.

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

---

## License Overview:

*   🛡️ **Polyform Shield License:** Applied to code and content within the `autogpt_platform` folder.
*   🦉 **MIT License:** Applied to the rest of the AutoGPT repository, including the original AutoGPT Agent, Forge, the benchmark, and the AutoGPT Classic GUI.

---
### Mission
Our mission is to provide the tools, so that you can focus on what matters:

- 🏗️ **Building** - Lay the foundation for something amazing.
- 🧪 **Testing** - Fine-tune your agent to perfection.
- 🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Be part of the revolution! **AutoGPT** is here to stay, at the forefront of AI innovation.

---

## Get Help and Contribute

*   **Documentation:**  [Documentation](https://docs.agpt.co)
*   **Contributing:** [Contributing](CONTRIBUTING.md)
*   **Questions/Support:** [Discord 💬](https://discord.gg/autogpt)
*   **Report Issues:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Agent Protocol
To maintain a uniform standard and ensure seamless compatibility with many current and future applications, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

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

## Contributors
<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>
```
Key improvements and SEO considerations:

*   **Clear and Concise Hook:** The one-sentence hook immediately introduces the core value proposition.
*   **Targeted Keywords:** Uses relevant keywords like "AI agents," "automation," "build," "deploy," and "manage" to improve search visibility.
*   **Structured Headings and Subheadings:** Organizes the information logically for readability and SEO.
*   **Bulleted Lists:**  Highlights key features and benefits, making the content easy to scan.
*   **Internal Linking:** Links to other relevant sections within the document to improve user experience.
*   **External Linking:**  Includes links to the original repository, documentation, and community resources.
*   **Concise Language:** Avoids unnecessary jargon and focuses on clarity.
*   **Mobile-Friendly:** The formatting adapts well to various screen sizes.
*   **Updated for Relevance:** Includes the latest information about the project, including the cloud-hosted beta.
*   **Emphasis on Benefits:**  Focuses on what users can *do* with AutoGPT (automate tasks, build agents, etc.).
*   **Call to Actions:**  Encourages users to join the waitlist, explore documentation, and contribute.