```markdown
# AutoGPT: Build Powerful AI Agents to Automate Workflows 🚀

**AutoGPT** is the groundbreaking platform that empowers you to design, deploy, and manage intelligent AI agents, streamlining complex tasks and automating your workflow.  [View the original repo](https://github.com/Significant-Gravitas/AutoGPT).

[![Discord](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features of AutoGPT

*   **AI Agent Creation:** Design and configure custom AI agents to automate tasks.
*   **Workflow Automation:** Build, modify, and optimize automated workflows.
*   **Agent Management:** Deploy and manage your AI agents, from testing to production.
*   **Ready-to-Use Agents:** Utilize pre-built agents for immediate productivity.
*   **Agent Interaction:** Easily run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.

## Getting Started

### Self-Hosting (Technical Users)

> **Note:** Setting up AutoGPT requires technical expertise.  For a simpler experience, consider the [cloud-hosted beta](https://bit.ly/3ZDijAI) (waitlist).

Follow the official self-hosting guide:  [https://docs.agpt.co/platform/getting-started/](https://docs.agpt.co/platform/getting-started/)

**Requirements:** Docker, VSCode, Git, npm

### System Requirements

#### Hardware Requirements
* CPU: 4+ cores recommended
* RAM: Minimum 8GB, 16GB recommended
* Storage: At least 10GB of free space

#### Software Requirements
* Operating Systems:
  * Linux (Ubuntu 20.04 or newer recommended)
  * macOS (10.15 or newer)
  * Windows 10/11 with WSL2
* Required Software (with minimum versions):
  * Docker Engine (20.10.0 or newer)
  * Docker Compose (2.0.0 or newer)
  * Git (2.30 or newer)
  * Node.js (16.x or newer)
  * npm (8.x or newer)
  * VSCode (1.60 or newer) or any modern code editor

#### Network Requirements
* Stable internet connection
* Access to required ports (will be configured in Docker)
* Ability to make outbound HTTPS connections


## AutoGPT Frontend: Your AI Agent Interface

The AutoGPT frontend provides a user-friendly interface to interact with and manage your AI agents.

*   **Agent Builder:** Customize agents using a low-code interface.
*   **Workflow Management:** Easily build and adjust automation workflows.
*   **Deployment Controls:** Manage the agent lifecycle.
*   **Ready-to-Use Agents:** Access pre-configured agents.
*   **Agent Interaction:** Run and engage with agents.
*   **Monitoring and Analytics:** Track performance and gain insights.

Learn more about custom blocks: [https://docs.agpt.co/platform/new_blocks/](https://docs.agpt.co/platform/new_blocks/)

## AutoGPT Server: The Agent's Engine

The AutoGPT Server powers your AI agents, providing the infrastructure for continuous operation.

*   **Source Code:** The core logic driving agent functionality.
*   **Infrastructure:** Ensures reliable, scalable performance.
*   **Marketplace:** A marketplace for pre-built agents.

## Example Agents: See the Power of AutoGPT

1.  **Generate Viral Videos from Trending Topics**
    *   Reads trending topics from Reddit.
    *   Creates short-form videos.

2.  **Identify Top Quotes from Videos for Social Media**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Identifies impactful quotes for social media posts.

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

##  Mission and Licensing

Our mission is to empower you to build, test, and delegate tasks to AI.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

**Licensing:**

*   MIT License: Main repository.
*   Polyform Shield License: Applies to the `autogpt_platform` folder.
For more information, see [https://agpt.co/blog/introducing-the-autogpt-platform](https://agpt.co/blog/introducing-the-autogpt-platform)

## Need Help?

### [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

Report bugs and suggest features via [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation for seamless integration with other applications.

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
Key improvements and SEO optimizations:

*   **Concise hook:** A strong opening sentence to grab attention.
*   **Clear Headings:**  Uses descriptive headings and subheadings for readability and SEO.
*   **Bulleted Key Features:** Highlights key advantages.
*   **Keyword Optimization:** Incorporates relevant keywords like "AI Agents," "Workflow Automation," "Automation," and "AI".
*   **Call to Action:** Encourages users to join the waitlist.
*   **Structure:**  Organized information for easy scanning.
*   **Links:**  Includes links to the documentation, Discord, and original repository.
*   **HTML Comments:**  Added HTML comments to note missing images.
*   **Clearer language and organization**:  Simplified some sentences and reorganized content for clarity.
*   **Added System Requirements**: Included system requirements for self-hosting.
