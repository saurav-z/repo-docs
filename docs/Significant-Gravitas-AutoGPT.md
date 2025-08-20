# AutoGPT: Build, Deploy, and Run AI Agents to Automate Your World

**Unleash the power of AI with AutoGPT, a cutting-edge platform empowering you to create, deploy, and manage intelligent agents that automate complex tasks. [Explore the original repository](https://github.com/Significant-Gravitas/AutoGPT).**

[![Discord](https://img.shields.io/discord/1097691256422786108?logo=discord&label=Discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Get started instantly with a library of ready-to-use AI agents.
*   **Agent Interaction:** Seamlessly run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.

## Hosting Options

*   **Self-Host (Free):** Download and run AutoGPT on your own infrastructure.  See below for setup instructions.
*   **Cloud-Hosted Beta (Coming Soon!):** Join the [waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta for a hassle-free experience.

## Getting Started: Self-Hosting AutoGPT

**Please Note:** Self-hosting AutoGPT requires some technical expertise.  Consider joining the cloud-hosted beta if you prefer a simpler setup.

### System Requirements

Ensure your system meets these requirements before installation:

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
*   Ability to make outbound HTTPS connections

### Setup Instructions

Detailed, up-to-date instructions are available on the official documentation site:

👉 [Follow the self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

#### Quick Setup (Recommended for Local Hosting)

Automate the setup process with a single-line script:

*   **macOS/Linux:**
    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```

*   **Windows (PowerShell):**
    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

This script installs dependencies, configures Docker, and launches your local instance.

## AutoGPT Components

### 🧱 AutoGPT Frontend

Interact with your AI agents through the AutoGPT frontend:

*   **Agent Builder:** Design and customize your own AI agents.
*   **Workflow Management:** Easily build, modify, and optimize automation workflows by connecting blocks.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Select from our library of pre-configured agents.
*   **Agent Interaction:** Run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agents' performance and gain insights.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The powerhouse that runs your agents:

*   **Source Code:** Core logic that drives agents and automations.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** Deploy a wide range of pre-built agents.

### 🐙 Example Agents

Get inspired by these examples:

1.  **Generate Viral Videos:**
    *   Reads trending topics on Reddit.
    *   Creates short-form videos automatically.
2.  **Identify Top Quotes from Videos:**
    *   Transcribes your YouTube videos.
    *   Identifies impactful quotes.
    *   Generates social media posts.

## Licensing

*   🛡️ **Polyform Shield License:** The `autogpt_platform` folder is licensed under the Polyform Shield License. [Learn more](https://agpt.co/blog/introducing-the-autogpt-platform)
*   🦉 **MIT License:** Other parts of the repository are licensed under the MIT License.  This includes projects such as Forge, agbenchmark, and the AutoGPT Classic GUI.

## Mission

Our mission is to empower you to:

*   🏗️ **Build**: Lay the foundation for something amazing.
*   🧪 **Test**: Fine-tune your agent to perfection.
*   🤝 **Delegate**: Let AI work for you.

Be part of the AI revolution!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---
## 🤖 AutoGPT Classic

> Below is information about the classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forge your own agent!** Forge handles boilerplate code, letting you focus on your agent's unique features.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` offers a stringent testing environment.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### ⌨️ CLI

Use the CLI at the repo root:

```shell
$ ./run
```

## 🤔 Questions? Problems? Suggestions?

### Get Help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

Report bugs and request features on [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) to standardize communication.

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