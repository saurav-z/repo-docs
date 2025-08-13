# AutoGPT: Unleash AI Automation to Automate Complex Workflows

**AutoGPT empowers you to build, deploy, and manage powerful AI agents, revolutionizing how you automate tasks.** [(View on GitHub)](https://github.com/Significant-Gravitas/AutoGPT)

[![Discord](https://img.shields.io/discord/1098832540836902922?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features

*   **Automated Workflow Creation:** Design and manage AI agents to automate complex processes.
*   **Intuitive Interface:** Build, modify, and optimize your automation workflows with ease.
*   **Flexible Deployment Options:** Self-host AutoGPT for free or join the cloud-hosted beta.
*   **Pre-built Agents:** Utilize a library of ready-to-use agents for instant productivity.
*   **Performance Monitoring:** Track agent performance and gain insights for continuous improvement.

## Get Started

### Hosting Options
   - Download to self-host (Free!)
   - [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Closed Beta - Public release Coming Soon!)

### 1. Self-Hosting (Free & Customizable)

Follow the setup instructions to get your instance up and running, and configure Docker.

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

#### Quick Setup with One-Line Script (Recommended for Local Hosting)

For macOS/Linux:
```
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script will install dependencies, configure Docker, and launch your local instance.

### 2. Explore the AutoGPT Platform

#### AutoGPT Frontend: Your AI Automation Hub

*   **Agent Builder:** Build your own AI agents using a low-code interface.
*   **Workflow Management:** Connect blocks to build and modify automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Select pre-configured agents.
*   **Agent Interaction:** Easily run and interact with agents through the user-friendly interface.
*   **Monitoring and Analytics:** Keep track of your agents' performance.

#### AutoGPT Server: The Engine

*   **Source Code:** Core logic that drives agents and automation.
*   **Infrastructure:** Robust systems that ensure reliable and scalable performance.
*   **Marketplace:** Find and deploy a wide range of pre-built agents.

### 3. Example Agents

*   **Generate Viral Videos:** Automate video creation from trending topics.
*   **Extract Top Quotes:** Identify impactful quotes from videos for social media.

## AutoGPT Classic

Learn more about the classic version of AutoGPT.

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

## 🤝 Agent Protocol

To maintain a uniform standard and ensure seamless compatibility with many current and future applications, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

## Support and Community

*   **[Discord](https://discord.gg/autogpt):** Get help and connect with the community.
*   **[GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose):** Report bugs or suggest features.

## License

*   **Polyform Shield License:** All code and content within the `autogpt_platform` folder.
*   **MIT License:** All other parts of the AutoGPT repository (e.g., `AutoGPT Classic`).

## Resources

*   **📖 [Documentation](https://docs.agpt.co)**
*   **🚀 [Contributing](CONTRIBUTING.md)**

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