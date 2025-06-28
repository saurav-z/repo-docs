# AutoGPT: Automate Complex Workflows with AI Agents

**AutoGPT** empowers you to build, deploy, and run autonomous AI agents that revolutionize automation. ([Original Repository](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

*   **Automated Workflows:** Create and manage AI agents to automate complex tasks.
*   **Customizable Agents:** Design and configure agents tailored to your specific needs.
*   **Pre-built Agents:** Utilize a library of ready-to-use agents for instant automation.
*   **Scalable Infrastructure:** Leverage a robust server infrastructure for reliable agent performance.
*   **Performance Monitoring:** Track agent performance and gain insights for continuous improvement.

## Hosting Options

*   **Self-Hosting:** Download and install AutoGPT for full control.
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for a simplified, cloud-based experience.

## Self-Hosting Setup

**Important Note:** Self-hosting requires technical expertise. Consider the cloud-hosted beta for an easier setup.

### System Requirements

*   **Hardware:**
    *   CPU: 4+ cores recommended
    *   RAM: Minimum 8GB, 16GB recommended
    *   Storage: At least 10GB of free space
*   **Software:**
    *   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), Windows 10/11 with WSL2
    *   Docker Engine (20.10.0 or newer)
    *   Docker Compose (2.0.0 or newer)
    *   Git (2.30 or newer)
    *   Node.js (16.x or newer)
    *   npm (8.x or newer)
    *   VSCode (1.60 or newer) or any modern code editor
*   **Network:**
    *   Stable internet connection
    *   Access to required ports (configured in Docker)
    *   Outbound HTTPS connections

### Installation

Follow the official self-hosting guide for detailed instructions:
👉 [Official Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

## AutoGPT Frontend

The user interface for interacting with and leveraging your AI automation.

*   **Agent Builder:** Design and customize AI agents using a low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Choose from a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for optimization.

Learn how to build custom blocks: [Building Blocks Guide](https://docs.agpt.co/platform/new_blocks/)

## AutoGPT Server

The powerful backend that drives your agents.

*   **Source Code:** The core logic of your agents and automation processes.
*   **Infrastructure:** Robust systems for reliable and scalable performance.
*   **Marketplace:** A comprehensive marketplace to discover and deploy pre-built agents.

## Example Agents

*   **Generate Viral Videos:** Creates short-form videos from trending topics.
*   **Identify Top Quotes:** Extracts and summarizes impactful quotes from videos for social media.

## Classic AutoGPT

Information about the classic version of AutoGPT.

### Forge

A toolkit for building your own agent application.

*   **Forge Your Agent:** [Forge your own agent!](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)
*   **Learn More:** [Learn More about Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### Benchmark

Measure your agent's performance with the `agbenchmark`.

*   **agbenchmark on Pypi:** [`agbenchmark`](https://pypi.org/project/agbenchmark/)
*   **Learn More:** [Learn More about the Benchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### UI

Easy-to-use frontend for managing your agents.

*   **Learn More:** [Learn More about the Frontend](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### CLI

A CLI to make it as easy as possible to use all of the tools offered by the repository.

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

## Get Help

*   **Discord:** [Discord](https://discord.gg/autogpt)
*   **Report a Bug/Feature Request:** [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation.

---

## License

MIT License: The majority of the AutoGPT repository is under the MIT License.

Polyform Shield License: This license applies to the autogpt_platform folder. 

For more information, see https://agpt.co/blog/introducing-the-autogpt-platform