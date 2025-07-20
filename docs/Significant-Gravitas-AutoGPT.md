# AutoGPT: Build, Deploy, and Run AI Agents to Automate Complex Workflows

**AutoGPT** empowers you to create, deploy, and manage powerful AI agents, revolutionizing how you automate tasks and workflows. [Learn more on the original GitHub repository](https://github.com/Significant-Gravitas/AutoGPT).

[![Discord](https://img.shields.io/discord/1076838801765357166?label=Discord&logo=discord&style=social)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Easily build, modify, and optimize your automation workflows through a visual interface.
*   **Deployment & Management:** Seamlessly manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agent Library:** Access a marketplace of ready-to-use agents for immediate deployment.
*   **Agent Interaction:** Interact with both custom and pre-configured agents through a user-friendly interface.
*   **Monitoring & Analytics:** Track agent performance and gain insights for continuous improvement.

## Getting Started

### Self-Hosting (Recommended for Local Use)

#### System Requirements

*   **Hardware:** 4+ CPU cores, minimum 8GB RAM (16GB recommended), and 10GB+ storage.
*   **Software:** Docker, Docker Compose, Git, Node.js, npm, and a modern code editor.
*   **Operating Systems:** Linux (Ubuntu 20.04+ recommended), macOS (10.15+), or Windows 10/11 with WSL2.
*   **Network:** Stable internet connection and ability to make HTTPS connections.

#### Setup

Follow the [official self-hosting guide](https://docs.agpt.co/platform/getting-started/) for detailed instructions.

**Quick Setup (One-Line Script)**

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

### Cloud-Hosted Beta (Coming Soon)

Join the [waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta to skip the technical setup and get started quickly.

## AutoGPT Components

### AutoGPT Frontend

The user interface where you'll build, interact with, and manage your AI agents.

*   **Agent Builder:** Build and customize your AI agents visually.
*   **Workflow Management:** Build and optimize your automation workflows with ease.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Start using pre-configured agents immediately.
*   **Agent Interaction:** Interact with your agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track your agents' performance.

### AutoGPT Server

The engine that powers your AI agents, enabling continuous operation and external triggers.

*   **Source Code:** The core logic behind the agents and automation.
*   **Infrastructure:** Reliable systems ensuring scalable performance.
*   **Marketplace:** A hub for finding and deploying pre-built agents.

## Example Agents

See the potential of AutoGPT with these examples:

1.  **Viral Video Generation:** Automate the creation of short-form videos from trending topics.
2.  **Social Media Quote Extraction:** Extract and post impactful quotes from your videos.

## AutoGPT Classic

The classic version of AutoGPT offers a range of tools for agent development and testing.

### Forge

A toolkit for building your own agent applications.

*   **Getting Started:** [Tutorial](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)
*   **Learn More:** [Forge Documentation](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### Benchmark

Measure your agent's performance with the `agbenchmark`.

*   **Install:** [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
*   **Learn More:** [Benchmark Documentation](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### UI

A user-friendly interface for controlling and monitoring your agents.

*   **Learn More:** [Frontend Documentation](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### CLI

The Command Line Interface, simplifies interacting with the tools offered.

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

## Mission and Licensing

*   **Our mission:**  To provide you with the tools for building, testing, and delegating to AI agents.

*   **Licensing:**
    *   MIT License: The majority of the AutoGPT repository.
    *   Polyform Shield License: Applies to the `autogpt_platform` folder.

    For more information, see https://agpt.co/blog/introducing-the-autogpt-platform

## Get Help

*   **Discord:** [Join our Discord](https://discord.gg/autogpt)
*   **Issues:** [Create a GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## Additional Resources

*   **Documentation:** [Documentation](https://docs.agpt.co)
*   **Contributing:** [Contributing Guide](CONTRIBUTING.md)

## Sister Projects

### Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) for seamless compatibility and standardization.

## Contributors

[View Contributors](https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors)