# AutoGPT: Build, Deploy, and Unleash the Power of AI Agents

**Automate complex workflows and revolutionize your productivity with AutoGPT, the leading platform for creating, deploying, and managing AI agents.** [Visit the original repository](https://github.com/Significant-Gravitas/AutoGPT).

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features

*   **AI Agent Creation:** Design and configure AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Easily build, modify, and optimize your automation workflows.
*   **Deployment Control:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents for immediate use.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for process improvement.

## Hosting Options

*   **Self-Hosting:** Download and host AutoGPT for free. (See the [official self-hosting guide](https://docs.agpt.co/platform/getting-started/)).
*   **Cloud-Hosted Beta:** Join the waitlist for the cloud-hosted beta (Closed Beta - Public release Coming Soon!) [Join the Waitlist](https://bit.ly/3ZDijAI).

## Quick Setup (Self-Hosting)

Get up and running quickly with our one-line setup scripts:

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## AutoGPT Platform: Deep Dive

The AutoGPT platform consists of a Frontend and a Server, working together to build and execute AI agents.

### 🧱 AutoGPT Frontend

The user-friendly interface where you create, manage, and interact with your AI agents. Key components include:

*   **Agent Builder:** Build custom agents with a low-code interface.
*   **Workflow Management:** Connect blocks to create automated workflows.
*   **Deployment Controls:** Manage agent lifecycles from testing to production.
*   **Ready-to-Use Agents:** Choose from pre-configured agents to begin automating immediately.
*   **Agent Interaction:** Run and interact with built and pre-configured agents.
*   **Monitoring and Analytics:** Monitor performance, analyze data, and improve automation processes.

### 💽 AutoGPT Server

The engine that powers your AI agents. This includes:

*   **Source Code:** The core logic driving agents and automation.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

### 🐙 Example Agents

*   **Generate Viral Videos:** Automatically create videos based on trending topics.
*   **Extract Top Quotes:** Transcribe videos and extract impactful quotes for social media.

## 🤖 AutoGPT Classic

The classic version of AutoGPT provides a toolkit for building and benchmarking AI agents.

### 🏗️ Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

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

## 🤝 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) for standardized communication between agents, the frontend, and the benchmark.

---

## 🤔 Questions? Problems? Suggestions?

*   **Get Help:** [Discord 💬](https://discord.gg/autogpt)
*   **Report Issues:** [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

---
### **License Overview:**

🛡️ **Polyform Shield License:**
All code and content within the `autogpt_platform` folder is licensed under the Polyform Shield License. This new project is our in-developlemt platform for building, deploying and managing agents.</br>_[Read more about this effort](https://agpt.co/blog/introducing-the-autogpt-platform)_

🦉 **MIT License:**
All other portions of the AutoGPT repository (i.e., everything outside the `autogpt_platform` folder) are licensed under the MIT License. This includes the original stand-alone AutoGPT Agent, along with projects such as [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge), [agbenchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) and the [AutoGPT Classic GUI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend).</br>We also publish additional work under the MIT Licence in other repositories, such as [GravitasML](https://github.com/Significant-Gravitas/gravitasml) which is developed for and used in the AutoGPT Platform. See also our MIT Licenced [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability) project.

---

### Mission

Our mission is to provide the tools, so that you can focus on what matters:

- 🏗️ **Building** - Lay the foundation for something amazing.
- 🧪 **Testing** - Fine-tune your agent to perfection.
- 🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Be part of the revolution! **AutoGPT** is here to stay, at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

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