# AutoGPT: Unleash the Power of AI Agents to Automate Workflows

**AutoGPT empowers you to build, deploy, and manage AI agents that revolutionize automation.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1095784939348170260?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt)  [![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features:

*   **Automated AI Agents:** Create, deploy, and manage AI agents capable of complex tasks.
*   **Agent Builder:** Design and configure custom AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows easily.
*   **Pre-built Agents:** Utilize ready-to-use agents from our library for immediate productivity.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.
*   **Open Source:** AutoGPT is open-source, fostering community-driven innovation.

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon!):** Join the waitlist for our cloud-hosted beta for a seamless experience. ([Join the Waitlist](https://bit.ly/3ZDijAI))

## Self-Hosting Guide

> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. 
> If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

Follow the official self-hosting guide here: 👉 [Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

### Quick Setup with One-Line Script (Recommended)

For macOS/Linux:
```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## AutoGPT Frontend: Your AI Agent Interface

The AutoGPT frontend provides a user-friendly interface to interact with and manage your AI agents.

*   **Agent Builder:** Easily customize agents with a low-code interface.
*   **Workflow Management:** Connect blocks to design and optimize workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents from testing to production.
*   **Ready-to-Use Agents:** Deploy pre-configured agents quickly.
*   **Agent Interaction:** Run and engage with your agents effortlessly.
*   **Monitoring and Analytics:** Track performance and refine automation processes.

[Read this guide](https://docs.agpt.co/platform/new_blocks/) to learn how to build your own custom blocks.

## AutoGPT Server: The AI Agent Engine

The AutoGPT Server is the backbone of the platform, where your agents execute.

*   **Source Code:** The core logic driving agents and automation.
*   **Infrastructure:** Reliable systems ensure scalability and performance.
*   **Marketplace:** Discover and deploy pre-built agents.

## Example Agents

See the potential of AutoGPT with these examples:

1.  **Generate Viral Videos from Trending Topics**
2.  **Identify Top Quotes from Videos for Social Media**

## AutoGPT Classic

Information about the classic version of AutoGPT.

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

## Agent Protocol

AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation.

## License Overview

*   **Polyform Shield License:** All code and content within the `autogpt_platform` folder.
*   **MIT License:** All other portions of the repository.

## Get Help and Contribute

*   **[Discord 💬](https://discord.gg/autogpt)**
*   Create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Mission

Build amazing things with AutoGPT.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

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