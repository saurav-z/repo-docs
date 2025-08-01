# AutoGPT: Unleash the Power of AI Agents (Self-Host or Join the Beta)

**Create, deploy, and manage continuous AI agents that automate complex workflows with AutoGPT!** ([See the original repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1098986462834813952?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

AutoGPT is a powerful platform empowering you to build, deploy, and manage AI agents for automated tasks. Choose to self-host for free or join the cloud-hosted beta for a streamlined experience.

## Key Features

*   **🤖 AI Agent Creation:** Design and configure custom AI agents with an intuitive, low-code Agent Builder.
*   **⚙️ Workflow Automation:** Easily build, modify, and optimize automation workflows using a visual, block-based interface.
*   **🚀 Deployment & Management:** Control the lifecycle of your agents, from testing to production.
*   **📦 Ready-to-Use Agents:** Get started instantly with a library of pre-configured, ready-to-deploy agents.
*   **📊 Monitoring & Analytics:** Track agent performance and gain insights for continuous improvement.
*   **🌐 Cloud Beta:** Join the waitlist for the cloud-hosted beta for a hassle-free experience.
*   **🛠️ Forge:** Easily build your own agent applications with this ready-to-go toolkit.
*   **🎯 Benchmark:** Measure your agent's performance.
*   **💻 UI:** User-friendly interface to control and monitor your agents.
*   **⌨️ CLI:** Easy to use command line interface.

## Hosting Options

*   **Self-Host (Free!):**  Download and set up AutoGPT on your own hardware.
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Closed Beta - Public release Coming Soon!).

## Self-Hosting AutoGPT: Getting Started

**For detailed setup instructions and documentation, please visit the official guide:**  [https://docs.agpt.co/platform/getting-started/](https://docs.agpt.co/platform/getting-started/)

### ⚡ Quick Setup

Get up and running in minutes with our one-line setup script! (Requires Docker, VSCode, git and npm)

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user-friendly interface for interacting with and managing your AI agents. Key features include:

*   **Agent Builder:** Customize your agents.
*   **Workflow Management:** Build workflows.
*   **Deployment Controls:** Manage your agents.
*   **Ready-to-Use Agents:** Start using agents immediately.
*   **Agent Interaction:** Run and interact with your agents.
*   **Monitoring and Analytics:** Track your agents' performance.

### 💽 AutoGPT Server

The engine that powers your AI agents.

*   **Source Code:** The core logic that drives agents.
*   **Infrastructure:** Reliable and scalable systems.
*   **Marketplace:** Find and deploy pre-built agents.

## Example Agents

See the potential with these example agents:

1.  **Generate Viral Videos from Trending Topics:** Creates short-form videos based on trending topics.
2.  **Identify Top Quotes from Videos for Social Media:** Transcribes videos, identifies key quotes, and automatically creates social media posts.

## Licensing

*   **Polyform Shield License:** The `autogpt_platform` folder.
*   **MIT License:** All other parts of the repository, including [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge), [agbenchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) and the [AutoGPT Classic GUI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend).

## Mission

Focus on what matters:

*   **Building:** Lay the foundation for something amazing.
*   **Testing:** Fine-tune your agent to perfection.
*   **Delegating:** Let AI work for you.

## 🤖 AutoGPT Classic

For information about the classic version of AutoGPT, see the following sections.

### 🛠️ Build your own Agent

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application.

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

## 🤔 Questions? Problems? Suggestions?

*   **Get Help:** [Discord 💬](https://discord.gg/autogpt)
*   **Report Issues/Feature Requests:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard for seamless compatibility.

---

## Star History

<!-- Removed the star history, but it's ready to add back -->

## ⚡ Contributors

<!-- Removed the contributors section. It's ready to add back -->