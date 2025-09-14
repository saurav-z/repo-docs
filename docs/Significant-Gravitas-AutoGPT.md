# AutoGPT: Unleash the Power of AI Agents to Automate Workflows

**AutoGPT is the leading platform for building, deploying, and running autonomous AI agents, transforming how you approach automation.**  [Learn More](https://github.com/Significant-Gravitas/AutoGPT)

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

## Key Features

*   **AI Agent Creation:** Design and customize AI agents using an intuitive, low-code Agent Builder.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease, connecting blocks to perform actions.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-Built Agents:** Access a library of ready-to-use agents for immediate deployment.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.
*   **Self-Hosting:** Download the platform for free, allowing you to customize and control.
*   **Cloud-Hosted Beta:** Join the waitlist for a cloud-hosted beta for a seamless experience. (Coming Soon!)

## Hosting Options

*   **Self-Host:** Download and run AutoGPT on your own hardware (Free!).
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the upcoming cloud-hosted version.

## Getting Started with Self-Hosting

> [!NOTE]
> Self-hosting requires technical expertise.  The cloud-hosted beta is recommended for ease of use.

### System Requirements

*   **Hardware:** CPU (4+ cores recommended), RAM (8GB minimum, 16GB recommended), Storage (10GB+ free).
*   **Software:** Docker, Docker Compose, Git, Node.js, npm, and a modern code editor. (Specific versions are detailed in the full documentation.)
*   **Network:** Stable internet, access to required ports, and outbound HTTPS connections.

### Installation

1.  **Follow the Official Guide:** Detailed instructions are available on the official documentation site: [https://docs.agpt.co/platform/getting-started/](https://docs.agpt.co/platform/getting-started/)
2.  **Quick Setup Script (Recommended for Local Hosting):**

    *   **macOS/Linux:**
        ```bash
        curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
        ```
    *   **Windows (PowerShell):**
        ```powershell
        powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
        ```
    This script automates installation and configuration.

## Components of AutoGPT

### AutoGPT Frontend

The user interface for interacting with AI agents, offering features like:

*   **Agent Builder:** Design and configure your agents with a low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage agent lifecycles from testing to production.
*   **Ready-to-Use Agents:** Access pre-configured agents for instant use.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance.

### AutoGPT Server

The backend that powers your AI agents, including:

*   **Source Code:** Core logic for agent operations.
*   **Infrastructure:** Reliable and scalable systems.
*   **Marketplace:** Access to pre-built agents.

## Example Agents

*   **Generate Viral Videos:** Creates short-form videos from trending topics on Reddit.
*   **Identify Top Quotes from Videos:** Transcribes YouTube videos, identifies key quotes, and generates social media posts.

## License

*   **AutoGPT Platform:** Polyform Shield License (within the `autogpt_platform` folder). [Read More](https://agpt.co/blog/introducing-the-autogpt-platform)
*   **AutoGPT Repository (Classic & other projects):** MIT License. Includes the original AutoGPT Agent, Forge, Benchmark, and UI, as well as projects like GravitasML and Code Ability.

## Mission

AutoGPT is designed to help you:

*   🏗️ **Build:** Lay the foundation for incredible automation.
*   🧪 **Test:** Refine your agents for peak performance.
*   🤝 **Delegate:** Empower AI to work for you and bring your ideas to life.

Join the AI revolution with AutoGPT!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic (Legacy)

This section contains information about the classic, original version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forge your own agent!** &ndash; Forge is a toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

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

## Support and Community

### Need Help?

*   [Discord 💬](https://discord.gg/autogpt) - Get support and connect with the community.
*   [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose) - Report bugs or request features.

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard for seamless compatibility with various applications, standardizing the communication pathways.

---

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

## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>
```
Key improvements and SEO considerations:

*   **Strong Hook:** The opening sentence immediately grabs attention and highlights the core benefit.
*   **Clear Headings:** Uses clear, concise headings and subheadings to organize information.
*   **Bulleted Lists:** Emphasizes key features and benefits using bullet points for easy scanning.
*   **Keyword Optimization:** Includes relevant keywords like "AI agents," "automation," "workflow," and "self-hosting."
*   **Concise Language:** Uses clear and direct language, avoiding jargon where possible.
*   **Call to Action:** Encourages users to join the waitlist and explore the documentation.
*   **Focus on Benefits:** Highlights what users *get* from using AutoGPT.
*   **Clear Structure:**  Logically organizes the information for better readability and user experience.
*   **Links:** Uses more links to direct people to relevant content.
*   **Removed redundancy:**  Streamlined the content where similar information was presented multiple times.
*   **Expanded Classic Information**: Incorporated more information from the classic version.