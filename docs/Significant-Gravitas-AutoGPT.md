# AutoGPT: Unleash the Power of AI Agents to Automate Workflows

**AutoGPT empowers you to build, deploy, and manage intelligent AI agents that automate complex tasks, revolutionizing how you approach automation.**  ([Original Repository](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features

*   **AI Agent Creation & Management:** Design, deploy, and control continuous AI agents.
*   **Intuitive Agent Builder:**  A low-code interface to customize and configure your AI agents.
*   **Workflow Automation:** Build, modify, and optimize automation workflows with ease.
*   **Pre-built Agent Library:** Access a collection of ready-to-use agents for immediate deployment.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Performance Monitoring:** Track agent performance and gain insights for continuous improvement.
*   **Flexible Deployment:** Choose to self-host or join the cloud-hosted beta (coming soon!).

## Getting Started: Self-Hosting

### System Requirements
Ensure your system meets these requirements before proceeding:
*   **Hardware:** 4+ CPU cores, 8GB+ RAM, and 10GB+ storage recommended.
*   **Software:** Docker, Docker Compose, Git, Node.js, npm, and a code editor.
*   **Network:** Stable internet, access to required ports, and outbound HTTPS connections.

### Setup Guide
[Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

### Quick Setup with One-Line Script (Recommended)

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## AutoGPT Platform Overview

AutoGPT is comprised of:

### AutoGPT Frontend

The user interface to interact with the AI automation platform. Key features include:
*   **Agent Builder:** Design and configure your own AI agents. 
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Utilize pre-configured agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Keep track of your agents' performance.

[Read this guide](https://docs.agpt.co/platform/new_blocks/) to learn how to build your own custom blocks.

### AutoGPT Server

The backend that powers the agents. It includes:

*   **Source Code:** The core logic that drives agents.
*   **Infrastructure:** Robust systems for scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

### Example Agents

*   **Generate Viral Videos:** Reads trending topics and creates short-form videos.
*   **Identify Top Quotes:** Transcribes videos and identifies key quotes for social media.

## AutoGPT Classic

Information on the classic version of AutoGPT.

### Forge

*   **Build your own agent!** A toolkit to build agent applications.
*   **Getting Started:** [Getting Started with Forge](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)
*   **Learn More:** [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### Benchmark

*   **Measure your agent's performance!**
*   **PyPi:** [`agbenchmark`](https://pypi.org/project/agbenchmark/) 
*   **Learn More:** [Benchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### UI

*   **Makes agents easy to use!**
*   **Learn More:** [UI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### CLI

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

## Licensing

*   **Polyform Shield License:** AutoGPT Platform code.
*   **MIT License:** Other parts of the repository (including the original AutoGPT Agent, Forge, agbenchmark, and the AutoGPT Classic GUI).

## Mission

**AutoGPT empowers you to build, test, and delegate tasks to AI agents, so you can focus on what matters.**

*   🏗️ **Building** - Lay the foundation for something amazing.
*   🧪 **Testing** - Fine-tune your agent to perfection.
*   🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

## Additional Resources

### Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) standard for seamless compatibility.

## Get Help

*   [Discord 💬](https://discord.gg/autogpt)
*   [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>

## Star History

<p align="center">
<a href="https://star-history.com/#Significant-Gravitas/AutoGPT">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
  </picture>
</a>
</p>