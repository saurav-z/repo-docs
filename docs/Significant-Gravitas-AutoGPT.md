# AutoGPT: Build, Deploy, and Automate with AI Agents

**Unleash the power of AI automation with AutoGPT, a cutting-edge platform for creating, deploying, and managing autonomous AI agents.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

*   **AI Agent Creation & Deployment:** Design, build, deploy, and manage continuous AI agents for complex workflows.
*   **Customizable Agents:** Utilize a low-code interface to design and configure your own AI agents.
*   **Workflow Management:** Easily build, modify, and optimize your automation workflows.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents to get started quickly.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track your agents' performance and gain insights to improve your automation processes.
*   **Classic Version:** Build your own AI agent using Forge, or use the Benchmark and UI tools to improve and test your agent.
*   **Agent Protocol:** AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard.

## Hosting Options

*   **Self-Hosting:** Download and host AutoGPT on your own infrastructure.  See below for requirements and setup.
*   **Cloud-Hosted Beta:**  [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

## Setup for Self-Hosting

> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process.
> If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

### System Requirements

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

### Updated Setup Instructions:

*   Follow the official self-hosting guide here:  [https://docs.agpt.co/platform/getting-started/](https://docs.agpt.co/platform/getting-started/)

*   This tutorial assumes you have Docker, VSCode, git and npm installed.

## AutoGPT Frontend

The AutoGPT frontend is where users interact with our powerful AI automation platform. It offers multiple ways to engage with and leverage our AI agents. This is the interface where you'll bring your AI automation ideas to life.

*   **Agent Builder:** Customize agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Select from a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with agents via a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/)

## AutoGPT Server

The AutoGPT Server is the powerhouse of our platform where your agents run.  Once deployed, agents can be triggered by external sources and can operate continuously.

*   **Source Code:** Core logic that drives agents and automation processes.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** Find and deploy a wide range of pre-built agents.

## Example Agents

*   **Generate Viral Videos from Trending Topics:**
    *   Reads topics from Reddit.
    *   Identifies trending topics.
    *   Creates short-form videos based on the content.
*   **Identify Top Quotes from Videos for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes new videos.
    *   Identifies impactful quotes and generates summaries.
    *   Posts to social media.

## AutoGPT Classic

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

## 🤝 Sister projects

### 🔄 Agent Protocol

To maintain a uniform standard and ensure seamless compatibility with many current and future applications, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

---

## ℹ️ Resources

*   **[Documentation](https://docs.agpt.co)**
*   **[Contributing](CONTRIBUTING.md)**

## Get Help

*   **[Discord 💬](https://discord.gg/autogpt)**
    [![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)
*   Report a bug or request a feature: [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Licensing

*   MIT License: The majority of the AutoGPT repository is under the MIT License.
*   Polyform Shield License: This license applies to the autogpt_platform folder.
*   For more information, see https://agpt.co/blog/introducing-the-autogpt-platform

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