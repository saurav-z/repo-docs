# AutoGPT: Unleash the Power of AI Agents for Automated Workflows

**Automate complex tasks and workflows with AutoGPT, a powerful platform for creating, deploying, and managing AI agents.** Explore the possibilities, then dive in and build your own! ([See the original repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features of AutoGPT

*   **Build, Deploy, and Run AI Agents:** Create and manage continuous AI agents to automate intricate workflows.
*   **Intuitive Frontend:** Interact with your AI agents through a user-friendly interface.
*   **Customizable Agent Builder:** Design and configure AI agents with a low-code interface.
*   **Workflow Management:** Easily build, modify, and optimize automation workflows.
*   **Ready-to-Use Agents:** Utilize pre-configured agents for immediate task automation.
*   **Comprehensive Monitoring & Analytics:** Track agent performance and gain actionable insights.
*   **Extensible Server Architecture:** Utilize core logic, robust infrastructure, and a marketplace for pre-built agents.
*   **Classic AutoGPT:** The classic version includes Forge (agent building toolkit), Benchmark (performance measurement), UI (user interface for agent interaction), and CLI (command-line interface).
*   **Agent Protocol:** Compatible with the Agent Protocol for seamless integration with the broader AI ecosystem.

## Get Started with AutoGPT

### Hosting Options

*   **Self-Hosting:** Download and set up AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) to access the cloud-hosted beta version.

### Self-Hosting Setup

> [!NOTE]
> Self-hosting requires technical expertise. The cloud-hosted beta is recommended for ease of use.

**Prerequisites:**

#### Hardware Requirements

*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

#### Software Requirements

*   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), Windows 10/11 with WSL2
*   Required Software (with minimum versions): Docker Engine (20.10.0 or newer), Docker Compose (2.0.0 or newer), Git (2.30 or newer), Node.js (16.x or newer), npm (8.x or newer), VSCode (1.60 or newer) or any modern code editor

#### Network Requirements

*   Stable internet connection
*   Access to required ports (will be configured in Docker)
*   Ability to make outbound HTTPS connections

**Installation:**
👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

### Classic AutoGPT Quickstart

**Build your own agent with Forge!** Components from Forge can also be used individually to speed up development and reduce boilerplate in your agent project.
🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)

### 🤖 AutoGPT Classic

#### 🏗️ Forge

**Forge your own agent!** Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec).

#### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

#### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

#### ⌨️ CLI

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

## 💡 Example Agents

*   **Generate Viral Videos:** Automatically create short-form videos from trending topics.
*   **Identify Top Quotes:** Extract and summarize impactful quotes from videos for social media.

These are just a few examples of the limitless possibilities with AutoGPT!

---
## Mission and Licensing

Our mission is to provide the tools to focus on what matters:
- 🏗️ **Building** - Lay the foundation for something amazing.
- 🧪 **Testing** - Fine-tune your agent to perfection.
- 🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

**Licensing:**
MIT License: The majority of the AutoGPT repository is under the MIT License.
Polyform Shield License: This license applies to the autogpt_platform folder. 

For more information, see https://agpt.co/blog/introducing-the-autogpt-platform

---

## 🤔 Questions? Problems? Suggestions?

*   Get help - [Discord 💬](https://discord.gg/autogpt)
*   Report bugs or request features: [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT utilizes the [agent protocol](https://agentprotocol.ai/) standard for seamless compatibility.

---

## Star Stats

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

*   **Concise Hook:** Starts with a compelling sentence to grab attention.
*   **Clear Headings:** Uses descriptive headings to organize the content.
*   **Bulleted Key Features:** Highlights the core functionalities of AutoGPT.
*   **SEO Keywords:** Uses relevant keywords like "AI agents," "automation," "workflows," "deploy," and "manage."
*   **Detailed Sections:** Provides thorough information about setup, classic version, and examples.
*   **Call to Actions:** Encourages users to join the Discord, access documentation, and contribute.
*   **Clean Formatting:** Uses markdown for readability.
*   **Links:** Includes links to the original repo, documentation, and other resources.
*   **Concise Summary:** Removes unnecessary wording and focuses on key information.
*   **Emphasis on Benefits:** Highlights the advantages of using AutoGPT.
*   **Classic AutoGPT Section:** Highlights the older code that some people may still be using.
*   **Clearer Pre-Requisites:** Clearly breaks down the hardware, software, and network prerequisites for self-hosting.