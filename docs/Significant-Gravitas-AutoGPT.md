# AutoGPT: Build, Deploy, and Automate with AI Agents

**AutoGPT empowers you to create, deploy, and manage powerful AI agents that automate complex tasks, revolutionizing your workflow.** [Explore the Original Repo](https://github.com/Significant-Gravitas/AutoGPT)

[![Discord](https://img.shields.io/discord/1090468611003823636?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features

*   **AI Agent Creation & Customization:** Design and configure your own AI agents using an intuitive, low-code interface with the Agent Builder.
*   **Workflow Management:** Build, modify, and optimize your automation workflows with ease using the Workflow Management system.
*   **Deployment & Management:** Control the lifecycle of your agents, from testing to production with Deployment Controls.
*   **Pre-Built Agents:** Access a library of ready-to-use agents for immediate automation.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Performance Monitoring:** Track agent performance and gain insights to improve automation processes with Monitoring and Analytics.
*   **Open-Source Classic Version:** Leverage the classic version of AutoGPT to build your own agent application.
*   **Benchmark for Performance Testing:** Measure and evaluate your agent's performance with the `agbenchmark`.
*   **User-Friendly Interface:** Easily control and monitor your agents with the `frontend`.
*   **Command-Line Interface (CLI):**  Simplify agent management with the CLI.

## Hosting Options

*   **Self-Hosting:** Download and host AutoGPT yourself, following the detailed setup instructions.
*   **Cloud-Hosted Beta (Join Waitlist):**  Sign up for the cloud-hosted beta for a hassle-free experience. ([Join the Waitlist](https://bit.ly/3ZDijAI))

## Getting Started with Self-Hosting

### System Requirements

Ensure your system meets these requirements before installing:

#### Hardware
*   CPU: 4+ cores recommended
*   RAM: 8GB minimum, 16GB recommended
*   Storage: 10GB+ free space

#### Software
*   Operating Systems: Linux (Ubuntu 20.04+ recommended), macOS (10.15+), Windows 10/11 with WSL2
*   Required Software: Docker Engine (20.10.0+), Docker Compose (2.0.0+), Git (2.30+), Node.js (16.x+), npm (8.x+), VSCode (1.60+) or any modern code editor

#### Network
*   Stable internet connection
*   Access to required ports
*   Ability to make outbound HTTPS connections

### Setup Instructions

For detailed and updated setup guides, please refer to the official documentation site: [https://docs.agpt.co/platform/getting-started/](https://docs.agpt.co/platform/getting-started/)

#### One-Line Quick Setup (Recommended)

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automates the installation of dependencies and launches your local instance.

## AutoGPT Frontend

The frontend provides a user-friendly interface for interacting with and leveraging the power of AI automation.

*   **Agent Builder:** Customize agents through a low-code, intuitive interface.
*   **Workflow Management:** Easily build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the agent lifecycle, from testing to production.
*   **Ready-to-Use Agents:** Utilize pre-configured agents directly.
*   **Agent Interaction:** Run and interact with your agents.
*   **Monitoring and Analytics:** Track and improve your automation processes.

[Learn how to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

## AutoGPT Server

The AutoGPT Server is the core of the platform where your AI agents run, enabling continuous operation and external trigger capabilities.

*   **Source Code:** The essential logic driving agents and automation.
*   **Infrastructure:** Reliable systems ensuring scalable performance.
*   **Marketplace:** Discover and deploy a wide range of pre-built agents.

## Example Agents

See what's possible with AutoGPT:

1.  **Generate Viral Videos from Trending Topics:**
    *   Reads trending topics on Reddit.
    *   Creates short-form videos based on content.

2.  **Identify Top Quotes from Videos for Social Media:**
    *   Subscribes to your YouTube channel.
    *   Transcribes and summarizes videos.
    *   Generates social media posts with impactful quotes.

## AutoGPT Classic

### Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart.

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

## Mission and Licensing

*   **Building:** Lay the foundation for something amazing.
*   **Testing:** Fine-tune your agent to perfection.
*   **Delegating:** Let AI work for you, and have your ideas come to life.

**License:**

*   MIT License (majority of the repository)
*   Polyform Shield License (autogpt\_platform folder)

## Community and Support

*   **Get help on Discord:** [Discord 💬](https://discord.gg/autogpt)
*   **Report issues on GitHub:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT adheres to the [agent protocol](https://agentprotocol.ai/) standard for seamless compatibility.

---

## Stars Stats

[Star History Chart](https://star-history.com/#Significant-Gravitas/AutoGPT)

## ⚡ Contributors

[![Contributors](https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10)](https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors)