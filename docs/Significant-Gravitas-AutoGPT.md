# AutoGPT: Automate Anything with AI Agents

**Unleash the power of AI automation with AutoGPT, a platform to build, deploy, and run AI agents that handle complex tasks from start to finish.**  Check out the original repo [here](https://github.com/Significant-Gravitas/AutoGPT).

[![Discord](https://img.shields.io/discord/1098949141480289842?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features

*   **AI Agent Creation:** Design and configure AI agents with a low-code, user-friendly interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage your agents' lifecycles, from testing to production.
*   **Pre-built Agents:** Utilize a library of ready-to-use, pre-configured agents.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for optimization.

## Hosting Options

*   **Self-Host (Free):** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon):** Join the waitlist for a cloud-hosted experience.  [Join the Waitlist](https://bit.ly/3ZDijAI)

## Getting Started - Self-Hosting

**Self-hosting AutoGPT allows you to have full control, but requires some technical expertise.**

### System Requirements

*   **Hardware:**
    *   CPU: 4+ cores recommended
    *   RAM: Minimum 8GB, 16GB recommended
    *   Storage: At least 10GB of free space
*   **Software:**
    *   Operating Systems: Linux (Ubuntu 20.04+ recommended), macOS (10.15+), Windows 10/11 with WSL2
    *   Docker Engine (20.10.0+)
    *   Docker Compose (2.0.0+)
    *   Git (2.30+)
    *   Node.js (16.x+)
    *   npm (8.x+)
    *   VSCode (1.60+) or any modern code editor
*   **Network:**
    *   Stable internet connection
    *   Access to required ports (configured in Docker)
    *   Outbound HTTPS connections

### Setup

**For detailed installation instructions, refer to the official documentation:** [https://docs.agpt.co/platform/getting-started/](https://docs.agpt.co/platform/getting-started/)

#### Quick Setup (Recommended for Local Hosting)

Simplify the setup process with our one-line script:

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automates dependency installation, Docker configuration, and local instance launching.

## AutoGPT Components

### AutoGPT Frontend

The user interface for interacting with AI agents.

*   **Agent Builder:** Build and customize AI agents.
*   **Workflow Management:** Design and optimize automation workflows.
*   **Deployment:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-Configured Agents:** Leverage ready-to-use agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track and improve automation processes.

### AutoGPT Server

The engine that powers your agents.

*   **Source Code:** The core logic behind agents and automation.
*   **Infrastructure:** Systems for reliable and scalable performance.
*   **Marketplace:** A hub for discovering and deploying agents.

## Example Agents

*   **Generate Viral Videos from Trending Topics:** Automates the creation of short-form videos based on trending content.
*   **Identify Top Quotes from Videos for Social Media:** Transcribes videos, identifies impactful quotes, and generates social media posts.

## AutoGPT Classic

> Below is information about the classic version of AutoGPT.

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

## License

*   **Polyform Shield License:** All code and content within the `autogpt_platform` folder.
*   **MIT License:**  Remaining portions of the repository, including the original AutoGPT Agent and related projects.

## Mission

Our mission is to empower you to:

*   🏗️ **Build:** Construct the foundation for your AI-driven projects.
*   🧪 **Test:** Fine-tune your agents for optimal performance.
*   🤝 **Delegate:** Let AI handle tasks, bringing your ideas to life.

## Get Involved

*   **Documentation:** [https://docs.agpt.co](https://docs.agpt.co)
*   **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

## Questions & Support

*   **Discord:** [Join the Discord Server](https://discord.gg/autogpt)
*   **GitHub Issues:** Report bugs and request features on [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Sister Projects

### Agent Protocol

AutoGPT adheres to the [Agent Protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation for seamless compatibility.

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
Key improvements and SEO considerations:

*   **Concise Title:**  Uses a clear, keyword-rich title: "AutoGPT: Automate Anything with AI Agents".
*   **One-Sentence Hook:**  A strong opening that immediately explains what AutoGPT does.
*   **Headings & Structure:**  Well-organized with clear headings and subheadings for readability and SEO.
*   **Bulleted Key Features:**  Highlights the most important aspects.
*   **Keywords:**  Includes relevant keywords throughout the text (AI agents, automation, build, deploy, etc.).
*   **Links:**  Includes direct links to the original repository and documentation.
*   **Concise Language:** Streamlined text for better clarity.
*   **Call to Action:** Encourages the reader to get involved.
*   **Focus on Benefits:** Emphasizes the value proposition of AutoGPT.
*   **Updated Discord Link:** Added the correct discord invite link.
*   **Corrected Classic Section Titles:** Corrected the formatting for the "Classic" section titles.