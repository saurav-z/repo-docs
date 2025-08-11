# AutoGPT: Build, Deploy, and Run AI Agents

**Create and manage powerful AI agents to automate complex tasks with AutoGPT, the leading platform for AI automation.**

[![Discord](https://img.shields.io/discord/1067889187065961502?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

AutoGPT empowers you to build, deploy, and manage continuous AI agents that automate complex workflows. This includes a variety of tools such as an Agent Builder for customizing agents, ready-to-use pre-configured agents, and workflow management.

**[Visit the AutoGPT GitHub Repository](https://github.com/Significant-Gravitas/AutoGPT)**

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents using an intuitive, low-code interface.
*   **Workflow Automation:** Build, modify, and optimize automation workflows with ease using a visual, block-based system.
*   **Agent Deployment & Management:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-Built Agents:** Access a library of ready-to-use agents for immediate deployment.
*   **Agent Interaction:** Interact with both custom and pre-configured agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights to improve automation processes.
*   **Open Source Classic Version:** Build your own agent using Forge, measure its performance using the Benchmark, and use the UI to interact.

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own infrastructure.
    *   [Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)
    *   Quick Setup:
        *   **macOS/Linux:** `curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh`
        *   **Windows (PowerShell):** `powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"`
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Coming Soon!).

## System Requirements (Self-Hosting)

### Hardware Requirements
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

### Software Requirements
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

### Network Requirements
*   Stable internet connection
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user interface for interacting with and managing your AI agents:

*   **Agent Builder:** Design and configure custom AI agents.
*   **Workflow Management:** Build and optimize automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Access pre-configured agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance and gain insights.

### 💽 AutoGPT Server

The core of the platform where your agents run:

*   **Source Code:** The logic that drives agents.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

## 🐙 Example Agents

*   **Generate Viral Videos:** Create short-form videos from trending topics on Reddit.
*   **Identify Top Quotes:** Transcribe videos and identify impactful quotes for social media posts.

## License

*   **Polyform Shield License:** `autogpt_platform` folder
*   **MIT License:** All other parts of the repository (including AutoGPT Classic)

## Mission

Build, test, and delegate tasks to AI agents, to help transform the future of work.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic

Learn about the classic version of AutoGPT, and use Forge to build your own agent!

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

*   **Concise Hook:** The one-sentence hook is now at the very beginning, highlighting the core value proposition.
*   **Clear Headings:**  Used descriptive and SEO-friendly headings to structure the content logically.
*   **Keyword Optimization:**  Included relevant keywords like "AI agents," "automation," "build," "deploy," "manage."
*   **Bulleted Lists:** Used bullet points to make key features and requirements easy to scan.
*   **Actionable Language:** Used active verbs and calls to action throughout (e.g., "Create," "Build," "Join the Waitlist").
*   **Emphasis on Value:**  Focused on what users *can* do with AutoGPT.
*   **Clear Structure:** Organized information into logical sections (Features, Hosting, System Requirements, Components, Examples).
*   **Internal Links:**  Linked to relevant sections within the document (e.g., "Learn More about Forge").
*   **External Links:** All links are intact.
*   **Conciseness:**  Removed redundant phrases.
*   **Alt Text for Images:**  Added `alt` text to the contributor image.
*   **Removed unnecessary information** Some of the original information could be removed to make it easier to read.