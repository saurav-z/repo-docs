# AutoGPT: Unleash the Power of AI Agents to Automate Workflows

**AutoGPT empowers you to build, deploy, and manage AI agents that automate complex tasks, revolutionizing how you approach productivity.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1098685200192401458?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social&logo=twitter)](https://twitter.com/Auto_GPT)

## Key Features:

*   🤖 **Automated AI Agents:** Create and deploy AI agents capable of autonomous operation.
*   🏗️ **Agent Builder:** Design and configure custom AI agents through an intuitive, low-code interface.
*   🚀 **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   📦 **Ready-to-Use Agents:** Leverage pre-configured agents for immediate automation solutions.
*   📊 **Monitoring and Analytics:** Track agent performance and gain insights to improve automation processes.
*   🌐 **Self-Hosting & Cloud Options:** Host AutoGPT locally or explore upcoming cloud-based solutions.
*   🔬 **Benchmark:** Measure your agent's performance for real-world applications.
*   💻 **UI:** User-friendly interface to control and monitor agents.

## Get Started

### Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own hardware.
*   **Cloud-Hosted Beta (Coming Soon!):** Join the waitlist for a cloud-hosted experience. [Join the Waitlist](https://bit.ly/3ZDijAI)

### Self-Hosting

**Prerequisites:**

Ensure your system meets the following requirements before setting up AutoGPT:

**Hardware Requirements**
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

**Software Requirements**
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

**Network Requirements**
*   Stable internet connection
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

**Detailed Setup:**

Follow the official self-hosting guide for detailed instructions: [Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

**Quick Setup (Recommended):**

Use the one-line script for a fast and easy setup:

*   **macOS/Linux:**

    ```bash
    curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
    ```

*   **Windows (PowerShell):**

    ```powershell
    powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
    ```

## AutoGPT Components:

### 🧱 AutoGPT Frontend

The user interface for interacting with and managing your AI agents.

*   **Agent Builder:** Design and customize AI agents.
*   **Workflow Management:** Build and optimize automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/)

### 💽 AutoGPT Server

The engine that powers your agents.

*   **Source Code:** The core logic behind the agents and automation.
*   **Infrastructure:** Robust systems for reliable performance.
*   **Marketplace:** Discover and deploy pre-built agents.

## 🐙 Example Agents

Examples showcasing the potential of AutoGPT:

1.  **Generate Viral Videos from Trending Topics:** Automatically creates short-form videos based on trending topics.
2.  **Identify Top Quotes from Videos for Social Media:** Transcribes videos, identifies impactful quotes, and generates social media posts.

## AutoGPT Classic

Information about the classic version of AutoGPT.

### 🏗️ Forge
Toolkit to build your own agent application.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

### 🎯 Benchmark
Measure your agent's performance for real-world applications.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi

### 💻 UI
User-friendly interface to control and monitor agents.

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

## Licensing

*   🛡️ **Polyform Shield License:** Code within the `autogpt_platform` folder.
*   🦉 **MIT License:** All other portions of the repository.

---

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) to standardize communication with the frontend and benchmark.

---

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure someone else hasn't created an issue for the same topic.

## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>