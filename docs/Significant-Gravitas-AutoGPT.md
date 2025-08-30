# AutoGPT: Automate Your World with AI Agents 🤖

**Unleash the power of AI to automate complex tasks and workflows with AutoGPT, a cutting-edge platform for building, deploying, and managing intelligent agents.** [Explore the original repository](https://github.com/Significant-Gravitas/AutoGPT).

[![Discord](https://img.shields.io/discord/1098456757647109170?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
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

*   **AI Agent Creation:** Design and configure your own AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents to get started quickly.
*   **Agent Interaction:** Easily run and interact with your custom or pre-configured agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track your agents' performance and gain insights for continuous improvement.
*   **Self-Hosting Options:** Host the platform on your own hardware (Free!)
*   **Cloud Hosted Beta:** Join the waitlist for the hosted Beta (Coming Soon!)

## Getting Started: Self-Hosting

AutoGPT offers self-hosting, allowing you to have full control over your AI agents.

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

For detailed setup instructions, including the latest updates, please refer to the [official self-hosting guide](https://docs.agpt.co/platform/getting-started/).

This tutorial assumes you have Docker, VSCode, git and npm installed.

#### ⚡ Quick Setup with One-Line Script (Recommended for Local Hosting)

Simplify your setup and launch your local instance in minutes using our automatic setup script.

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script handles dependencies, configures Docker, and launches your local instance in one step.

## Core Components

### 🧱 AutoGPT Frontend

The user interface where you'll design, manage, and interact with your AI automation projects.

### 💽 AutoGPT Server

The engine room where your AI agents run, providing the source code, infrastructure, and marketplace.

### 🐙 Example Agents

Here are two examples of what you can do with AutoGPT:

1.  **Generate Viral Videos from Trending Topics**
    *   This agent reads topics on Reddit.
    *   It identifies trending topics.
    *   It then automatically creates a short-form video based on the content.

2.  **Identify Top Quotes from Videos for Social Media**
    *   This agent subscribes to your YouTube channel.
    *   When you post a new video, it transcribes it.
    *   It uses AI to identify the most impactful quotes to generate a summary.
    *   Then, it writes a post to automatically publish to your social media.

These examples show just a glimpse of what you can achieve with AutoGPT! You can create customized workflows to build agents for any use case.

---

## 🤝 Supporting Projects & Technologies

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation to ensure seamless compatibility.

---

## License Overview

🛡️ **Polyform Shield License:**
All code and content within the `autogpt_platform` folder is licensed under the Polyform Shield License. This new project is our in-developlemt platform for building, deploying and managing agents.</br>_[Read more about this effort](https://agpt.co/blog/introducing-the-autogpt-platform)_

🦉 **MIT License:**
All other portions of the AutoGPT repository (i.e., everything outside the `autogpt_platform` folder) are licensed under the MIT License. This includes the original stand-alone AutoGPT Agent, along with projects such as [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge), [agbenchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) and the [AutoGPT Classic GUI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend).</br>We also publish additional work under the MIT Licence in other repositories, such as [GravitasML](https://github.com/Significant-Gravitas/gravitasml) which is developed for and used in the AutoGPT Platform. See also our MIT Licenced [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability) project.

---

### Mission

Our mission is to provide the tools, so that you can focus on what matters:

*   🏗️ **Building** - Lay the foundation for something amazing.
*   🧪 **Testing** - Fine-tune your agent to perfection.
*   🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Be part of the revolution! **AutoGPT** is here to stay, at the forefront of AI innovation.

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

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure someone else hasn't created an issue for the same topic.

## 🤝 Sister projects

### 🔄 Agent Protocol

To maintain a uniform standard and ensure seamless compatibility with many current and future applications, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

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