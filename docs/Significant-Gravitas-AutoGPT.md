# AutoGPT: Automate Your World with AI Agents

**Unleash the power of AI with AutoGPT, a platform that allows you to build, deploy, and run intelligent agents to automate complex tasks.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1095976555302157332?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt)  [![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

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

*   **AI Agent Creation:** Design and configure your own AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease using a visual block-based system.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agent Library:** Quickly get started with ready-to-use agents for various tasks.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights to improve your automation processes.

## Hosting Options

*   **Self-Host:** Download and set up the platform on your own infrastructure (Free!).
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the upcoming cloud-hosted beta release.

## Getting Started: Self-Hosting

Self-hosting AutoGPT allows you to have full control over your AI agents.

### System Requirements

Ensure your system meets the following before installation:

#### Hardware Requirements

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

#### Software Requirements

*   Operating Systems:
    *   Linux (Ubuntu 20.04 or newer recommended)
    *   macOS (10.15 or newer)
    *   Windows 10/11 with WSL2
*   Required Software:
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

### Detailed Setup Instructions

For comprehensive setup instructions, please refer to the official documentation: [https://docs.agpt.co/platform/getting-started/](https://docs.agpt.co/platform/getting-started/)

### Quick Setup (Recommended)

Simplify the setup process with our automated script:

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automatically installs dependencies, configures Docker, and launches your local instance.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user interface for interacting with your AI agents.

*   **Agent Builder:** Design and customize agents.
*   **Workflow Management:** Build and modify automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Utilize pre-configured agents.
*   **Agent Interaction:** Run and interact with your agents.
*   **Monitoring and Analytics:** Track and improve agent performance.

**Build Custom Blocks:** Learn how to build your own custom blocks [here](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The core of the platform where your AI agents execute.

*   **Source Code:** The logic behind your agents and automations.
*   **Infrastructure:** Ensures reliable and scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

## Example Agents

Here are examples of agents you can build with AutoGPT:

1.  **Generate Viral Videos:** Create short-form videos from trending topics on Reddit.
2.  **Identify Top Quotes:** Transcribe YouTube videos, identify key quotes, and generate social media posts.

## Licensing

*   **Polyform Shield License:**  Applies to all code and content within the `autogpt_platform` folder. ([Learn More](https://agpt.co/blog/introducing-the-autogpt-platform))
*   **MIT License:** Applies to the rest of the repository, including projects like Forge, agbenchmark, and the AutoGPT Classic GUI.

## Mission

Empowering you to:

*   🏗️ **Build** innovative AI solutions.
*   🧪 **Test** and refine your agents.
*   🤝 **Delegate** tasks to AI for increased productivity.

Join the AI revolution!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic (Legacy)

> Information about the original, classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forge your own agent!** Forge is a toolkit to build your own agent application.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### 🎯 Benchmark

**Measure your agent's performance!**

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

**Makes agents easy to use!**

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### ⌨️ CLI

[CLI]: #-cli

```shell
$ ./run
Usage: cli.py [OPTIONS] COMMAND [ARGS]...
```

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

For issues or feature requests, submit a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation.

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