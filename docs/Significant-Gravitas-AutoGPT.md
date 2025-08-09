# AutoGPT: Automate Complex Workflows with AI Agents

**Unleash the power of AI by building, deploying, and running autonomous AI agents with AutoGPT, revolutionizing automation.**

[Visit the original repository on GitHub](https://github.com/Significant-Gravitas/AutoGPT)

*   [Join the AutoGPT Discord](https://discord.gg/autogpt)
*   [Follow on Twitter](https://twitter.com/Auto_GPT)

## Key Features of AutoGPT

*   **Agent Builder:** Design and configure custom AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents for immediate use.
*   **Agent Interaction:** Run and interact with your own or pre-configured agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for process improvement.

## Hosting Options
*   **Self-Host:** Download and host the platform for free.
*   **Cloud-Hosted Beta:** Join the waitlist for a cloud-hosted version. ([Join the Waitlist](https://bit.ly/3ZDijAI) - Closed Beta - Public release Coming Soon!)

## Getting Started with Self-Hosting

### System Requirements
*   **Hardware:** CPU (4+ cores recommended), RAM (8GB minimum, 16GB recommended), Storage (10GB+ free)
*   **Software:**
    *   Linux (Ubuntu 20.04+ recommended), macOS (10.15+), or Windows 10/11 with WSL2
    *   Docker Engine (20.10.0+), Docker Compose (2.0.0+), Git (2.30+), Node.js (16.x+), npm (8.x+), VSCode (1.60+)
*   **Network:** Stable internet, required port access, outbound HTTPS connections.

### Quick Setup with One-Line Script (Recommended)

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script installs dependencies, configures Docker, and launches a local instance.

### AutoGPT Platform Components

*   **AutoGPT Frontend:** User interface for building, interacting with, and managing AI agents.
*   **AutoGPT Server:** The core engine where your agents run, including source code, infrastructure, and a marketplace.

### Example Agents

1.  **Generate Viral Videos:** Create short-form videos from trending topics on Reddit.
2.  **Identify Top Quotes:** Transcribe and summarize YouTube videos to generate social media posts.

## AutoGPT Classic

### Forge
**Forge your own agent!** Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

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

## AutoGPT Classic
> Below is information about the classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

## Licensing

*   **AutoGPT Platform:** Polyform Shield License (within `autogpt_platform` folder).
*   **Other Components (Classic Agent, etc.):** MIT License.

## Mission

Our mission is to empower you to:

*   🏗️ **Build:** Lay the foundation for something amazing.
*   🧪 **Test:** Fine-tune your agent to perfection.
*   🤝 **Delegate:** Let AI work for you, and bring your ideas to life.

Join the AI revolution!

## Resources

*   📖 [Documentation](https://docs.agpt.co)
*   🚀 [Contributing](CONTRIBUTING.md)

## Get Help

*   [Discord 💬](https://discord.gg/autogpt)
*   [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Sister Projects

*   **Agent Protocol:** Standardizes communication for seamless integration ([Agent Protocol](https://agentprotocol.ai/)).

## Stats
<p align="center">
<a href="https://star-history.com/#Significant-Gravitas/AutoGPT">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
  </picture>
</a>
</p>

## Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>