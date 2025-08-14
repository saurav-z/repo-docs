# AutoGPT: Build, Deploy, and Run Powerful AI Agents

**Unleash the power of AI with AutoGPT, a platform that empowers you to create, deploy, and manage autonomous AI agents for a wide range of applications.** ([View on GitHub](https://github.com/Significant-Gravitas/AutoGPT))

## Key Features

*   🤖 **Autonomous AI Agents:** Build, deploy, and manage AI agents capable of automating complex workflows.
*   🛠️ **Agent Builder:** Design and configure your own AI agents with a user-friendly, low-code interface.
*   🔄 **Workflow Management:** Easily build, modify, and optimize your automation workflows with a drag-and-drop interface.
*   🚀 **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   📦 **Ready-to-Use Agents:** Access a library of pre-configured agents for immediate use.
*   📊 **Monitoring and Analytics:** Track agent performance and gain insights to continually improve your automation processes.
*   🌐 **Self-Hosting Options:** Download and self-host the platform for free. 
*   ☁️ **Cloud-Hosted Beta (Coming Soon!):** Join the waitlist for the cloud-hosted beta version for an even easier experience.

## Getting Started

### Prerequisites

#### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: Minimum 8GB, 16GB recommended
- Storage: At least 10GB of free space

#### Software Requirements
- Operating Systems:
  - Linux (Ubuntu 20.04 or newer recommended)
  - macOS (10.15 or newer)
  - Windows 10/11 with WSL2
- Required Software (with minimum versions):
  - Docker Engine (20.10.0 or newer)
  - Docker Compose (2.0.0 or newer)
  - Git (2.30 or newer)
  - Node.js (16.x or newer)
  - npm (8.x or newer)
  - VSCode (1.60 or newer) or any modern code editor

#### Network Requirements
- Stable internet connection
- Access to required ports (will be configured in Docker)
- Ability to make outbound HTTPS connections

### Installation

For a quick setup, use the one-line script:

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

### Updated Setup Instructions:
We've moved to a fully maintained and regularly updated documentation site.

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

## Core Components

### AutoGPT Frontend

The intuitive interface where users interact with AI automation platform agents.

*   **Agent Builder:** Customize agents through an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the agent lifecycle.
*   **Ready-to-Use Agents:** Deploy pre-configured agents instantly.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track agent performance and optimize automation processes.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

### AutoGPT Server

The core of the platform where the agents run.

*   **Source Code:** The logic that drives agents and automation processes.
*   **Infrastructure:** Ensure reliable and scalable performance.
*   **Marketplace:** Deploy a variety of pre-built agents.

### Example Agents

#### 1. Generate Viral Videos

*   Reads trending topics from Reddit.
*   Identifies trending topics.
*   Creates a short-form video based on the content.

#### 2. Identify Top Quotes from Videos

*   Subscribes to your YouTube channel.
*   Transcribes new videos.
*   Uses AI to identify impactful quotes.
*   Generates a summary.
*   Writes a post for social media.

## AutoGPT Classic (Legacy Version)

**The original standalone AutoGPT agent, along with related tools:**

### Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

<!-- TODO: insert visual demonstrating the benchmark -->

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

<!-- TODO: insert screenshot of front end -->

The frontend works out-of-the-box with all agents in the repo. Just use the [CLI] to run your agent of choice!

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend) about the Frontend

### CLI

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

## 🌐 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation.

## License Overview

*   🛡️ **Polyform Shield License:**  All code and content within the `autogpt_platform` folder.
*   🦉 **MIT License:** All other portions of the AutoGPT repository.

## 🤝 Contribute

Help us build the future of AI! See our [Contributing Guide](CONTRIBUTING.md).

## 💬 Get Help

*   [Discord](https://discord.gg/autogpt)
*   [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Stars History

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