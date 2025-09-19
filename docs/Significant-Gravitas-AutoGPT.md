# AutoGPT: Build, Deploy, and Automate with AI Agents

**Unlock the power of AI automation with AutoGPT, the platform empowering you to create, deploy, and manage AI agents that streamline complex workflows.** ([Back to the repository](https://github.com/Significant-Gravitas/AutoGPT))

## Key Features

*   **AI Agent Creation:** Design and configure AI agents using a low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   **Agent Deployment:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Access a library of ready-to-use AI agents for immediate deployment.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Performance Monitoring:** Track agent performance and gain insights for continuous improvement.

## Getting Started

### Hosting Options

*   **Self-Hosting (Free!):** Download the platform and host it yourself.
*   **Cloud-Hosted Beta (Coming Soon):** Join the [waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta and experience AutoGPT with ease.

### Self-Hosting Guide

> **Note:** Setting up AutoGPT requires technical expertise. For a simpler experience, consider joining the cloud-hosted beta.

#### System Requirements

Ensure your system meets these requirements before installation:

*   **Hardware:** CPU (4+ cores recommended), RAM (8GB minimum, 16GB recommended), Storage (10GB+ free)
*   **Software:**
    *   Operating Systems: Linux (Ubuntu 20.04+ recommended), macOS (10.15+), Windows 10/11 with WSL2
    *   Docker Engine (20.10.0+), Docker Compose (2.0.0+), Git (2.30+), Node.js (16.x+), npm (8.x+), VSCode (1.60+) or any modern code editor
    *   Stable internet connection
    *   Access to required ports and outbound HTTPS connections

#### Updated Setup Instructions

For a fully maintained and regularly updated guide, visit the official documentation site:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

#### Quick Setup

Simplify installation with our one-line script.

For macOS/Linux:

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automatically installs dependencies, configures Docker, and launches your local instance.

## AutoGPT Platform Components

### AutoGPT Frontend

The user interface for interacting with and leveraging AI automation.

*   **Agent Builder:** Low-code interface for designing custom AI agents.
*   **Workflow Management:** Build and optimize automation workflows.
*   **Deployment Controls:** Manage agent lifecycles.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with agents via a user-friendly interface.
*   **Monitoring and Analytics:** Track performance and optimize processes.

[Learn more about building custom blocks.](https://docs.agpt.co/platform/new_blocks/)

### AutoGPT Server

The core of the platform where agents run.

*   **Source Code:** Drives agents and automation.
*   **Infrastructure:** Ensures reliable and scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

## Example Agents

1.  **Viral Video Generator:** Reads trending topics and creates short-form videos.
2.  **Social Media Quote Identifier:** Transcribes videos, identifies impactful quotes, and generates social media posts.

## License

*   **Polyform Shield License:** Code and content within the `autogpt_platform` folder.
*   **MIT License:** All other parts of the repository, including the original AutoGPT Agent, Forge, agbenchmark, and the AutoGPT Classic GUI.

## Mission

Empowering users with the tools to:

*   🏗️ **Build:** Create powerful AI agents.
*   🧪 **Test:** Fine-tune agents for optimal performance.
*   🤝 **Delegate:** Automate tasks and bring ideas to life.

## Additional Resources

*   [Documentation](https://docs.agpt.co)
*   [Contributing](CONTRIBUTING.md)
*   **AutoGPT Classic**

    *   [Forge - Build Your Own Agent - Quickstart](classic/FORGE-QUICKSTART.md)
    *   **Forge:** A toolkit for building custom agent applications.
        *   🚀 [Getting Started with Forge](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)
        *   📘 [Learn More about Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)
    *   **Benchmark:** Measure your agent's performance.
        *   📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
        *   📘 [Learn More about the Benchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)
    *   **UI:** User-friendly interface to control and monitor your agents.
        *   📘 [Learn More about the Frontend](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)
    *   **CLI:** Command-line interface for easy tool use.

### Questions, Problems, Suggestions?

*   [Discord](https://discord.gg/autogpt)
*   [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

### Sister Projects

*   **Agent Protocol:** Standardizes agent communication.

## Stars Stats

[<img src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" alt="Star History Chart"/>](https://star-history.com/#Significant-Gravitas/AutoGPT)

## Contributors

<img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />