# AutoGPT: Automate Complex Workflows with AI Agents

**Unleash the power of AI with AutoGPT, a platform designed to build, deploy, and manage AI agents that revolutionize automation.** 
[Explore the AutoGPT Repository](https://github.com/Significant-Gravitas/AutoGPT)

[![Discord](https://img.shields.io/discord/1092979056589419018?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

---

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents to automate specific tasks.
*   **Workflow Automation:** Build, modify, and optimize automation workflows with an intuitive interface.
*   **Deployment & Management:** Manage the lifecycle of your AI agents, from testing to production.
*   **Pre-built Agents:** Access a library of ready-to-use agents for immediate deployment.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights for continuous improvement.

---

## Getting Started with AutoGPT

AutoGPT offers flexible hosting options to fit your needs:

*   **Self-Hosting (Free):** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon):** Join the waitlist for our cloud-hosted platform. [Join the Waitlist](https://bit.ly/3ZDijAI)

---

### 🚀 Self-Hosting Guide

#### System Requirements

Before you begin, ensure your system meets these requirements:

##### Hardware:

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

##### Software:

*   **Operating Systems:** Linux (Ubuntu 20.04+ recommended), macOS (10.15+), Windows 10/11 with WSL2
*   **Required Software (with minimum versions):**
    *   Docker Engine (20.10.0+)
    *   Docker Compose (2.0.0+)
    *   Git (2.30+)
    *   Node.js (16.x+)
    *   npm (8.x+)
    *   VSCode (1.60+) or any modern code editor

##### Network:

*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Ability to make outbound HTTPS connections

#### Installation

**We recommend using our streamlined setup script for ease of use.**

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```

*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

This script automates the installation of dependencies and the launch of your local instance.

---

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The frontend provides a user-friendly interface for interacting with and managing your AI agents.

*   **Agent Builder:** Low-code interface for designing and configuring agents.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage your agents throughout their lifecycle.
*   **Ready-to-Use Agents:** Select from a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance and optimize.

[Learn how to build custom blocks here](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The server is the core of the platform, where your agents run and perform their tasks.

*   **Source Code:** The core logic that drives the agents and automations.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

---

## 🤖 Example Agents: Real-World Applications

1.  **Generate Viral Videos:** Automate video creation from trending topics on Reddit.
2.  **Identify Top Quotes:** Automatically extract key quotes from your YouTube videos for social media.

---

### 🛡️ License Overview

*   **Polyform Shield License:** The `autogpt_platform` folder is licensed under the Polyform Shield License.
*   **MIT License:** All other parts of the repository are licensed under the MIT License.

---

### 🤔 Questions? Problems? Suggestions?

*   **Get Help:** [Join us on Discord 💬](https://discord.gg/autogpt)
*   **Report Issues:** [Create a GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

---

## 🤖 AutoGPT Classic (Legacy)

Below is information about the classic version of AutoGPT.

### 🏗️ Forge

**Build your own agent!** - Forge is a toolkit to build your own agent application.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### ⌨️ CLI

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

---

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation.

---

## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>