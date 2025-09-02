# AutoGPT: Unleash the Power of AI Agents (Your Guide to Autonomous Automation)

**Build, deploy, and run AI agents to automate complex workflows with AutoGPT, an open-source platform.**

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://zdoc.app/de/Significant-Gravitas/AutoGPT) | 
[Español](https://zdoc.app/es/Significant-Gravitas/AutoGPT) | 
[français](https://zdoc.app/fr/Significant-Gravitas/AutoGPT) | 
[日本語](https://zdoc.app/ja/Significant-Gravitas/AutoGPT) | 
[한국어](https://zdoc.app/ko/Significant-Gravitas/AutoGPT) | 
[Português](https://zdoc.app/pt/Significant-Gravitas/AutoGPT) | 
[Русский](https://zdoc.app/ru/Significant-Gravitas/AutoGPT) | 
[中文](https://zdoc.app/zh/Significant-Gravitas/AutoGPT)

[**View the original repository on GitHub**](https://github.com/Significant-Gravitas/AutoGPT)

## Key Features of AutoGPT:

*   **AI Agent Creation:** Design and configure your own AI agents with an intuitive, low-code Agent Builder.
*   **Workflow Automation:** Easily build, modify, and optimize automation workflows.
*   **Deployment Control:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Access a library of ready-to-use agents for immediate tasks.
*   **Agent Interaction:** Run and interact with agents through a user-friendly interface.
*   **Performance Monitoring:** Track agent performance with built-in monitoring and analytics.
*   **Forge Toolkit:** Build your own agent application and handles most of the boilerplate code.
*   **Benchmark Testing:** Test your agent's performance with automated and objective performance evaluations.
*   **User-Friendly Interface (UI):** Control and monitor your agents.

## Hosting Options:

*   **Self-Host (Free):** Download and set up AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon!):** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Closed Beta - Public release Coming Soon!).

## Getting Started: Self-Hosting AutoGPT

> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. 
> If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

### System Requirements

*   **Hardware:**
    *   CPU: 4+ cores recommended
    *   RAM: Minimum 8GB, 16GB recommended
    *   Storage: At least 10GB of free space
*   **Software:**
    *   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), Windows 10/11 with WSL2
    *   Docker Engine (20.10.0 or newer)
    *   Docker Compose (2.0.0 or newer)
    *   Git (2.30 or newer)
    *   Node.js (16.x or newer)
    *   npm (8.x or newer)
    *   VSCode (1.60 or newer) or any modern code editor
*   **Network:**
    *   Stable internet connection
    *   Access to required ports (configured in Docker)
    *   Outbound HTTPS connections

### Installation Guide

We've moved to a fully maintained and regularly updated documentation site.

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

---

#### Quick Setup: One-Line Script

For macOS/Linux:
```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script simplifies the setup process, installing dependencies, configuring Docker, and launching your local instance.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The frontend provides the user interface for interacting with your AI automation platform.

   **Agent Builder:** Customize AI agents.
   **Workflow Management:** Build and modify automation workflows.
   **Deployment Controls:** Manage agent lifecycle.
   **Ready-to-Use Agents:** Utilize pre-configured agents.
   **Agent Interaction:** Run and interact with your agents.
   **Monitoring and Analytics:** Track agent performance.

[Learn how to build your own custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The AutoGPT Server is the core of the platform.

   **Source Code:** Drives agent and automation processes.
   **Infrastructure:** Ensures reliable and scalable performance.
   **Marketplace:** Find and deploy pre-built agents.

## Example AI Agents

1.  **Viral Video Generator:** Identifies trending topics on Reddit and creates short-form videos.
2.  **Social Media Quote Extractor:** Extracts impactful quotes from YouTube videos for social media posts.

## Licensing

*   🛡️ **Polyform Shield License:** `autogpt_platform` folder.
*   🦉 **MIT License:** All other parts of the repository, including classic AutoGPT, Forge, agbenchmark, and the frontend.

---

### Mission

Our mission is to empower you to:

-   🏗️ **Build** amazing solutions.
-   🧪 **Test** and refine your agents.
-   🤝 **Delegate** tasks to AI.

Join the AI revolution with AutoGPT!

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
```
Key improvements and SEO considerations:

*   **Strong Headline:**  Uses a keyword-rich headline and a compelling one-sentence hook to grab attention and improve search ranking.
*   **Clear Structure:**  Organized with clear headings and subheadings for readability.
*   **Keyword Optimization:**  Includes relevant keywords like "AI agents," "automation," "deploy," "open source," and related terms throughout the content.
*   **Bulleted Lists:** Uses bullet points for key features, benefits, and requirements, making the information easily scannable.
*   **Concise Language:**  Rephrases some sections to be more concise and easier to understand.
*   **Emphasis on Value:**  Highlights the benefits of using AutoGPT.
*   **Call to Action:** Includes links to documentation and contributing guidelines.
*   **Internal Linking:** Uses internal links to other relevant sections.
*   **Concise Summaries:**  Provides brief descriptions of each component (Frontend, Server, etc.).
*   **Removed Redundancy:** Streamlined the content and removed redundant phrases.
*   **Alt Text on Images:** Added alt text where placeholder image tags were added, improving accessibility and potentially SEO.
*   **Removed outdated information:** Streamlined sections to be up-to-date and relevant
*   **Added an image of the star history chart** Improving the overall look of the README