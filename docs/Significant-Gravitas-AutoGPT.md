# AutoGPT: Unleash the Power of AI with Automated Agents

[Original Repo](https://github.com/Significant-Gravitas/AutoGPT)

AutoGPT empowers you to **build, deploy, and manage autonomous AI agents that revolutionize how you automate tasks and achieve goals.**

## Key Features:

*   **Automated AI Agents:** Create agents that can perform tasks autonomously.
*   **Agent Builder:** Design and configure custom AI agents with an intuitive interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows.
*   **Pre-Built Agents:** Leverage a library of ready-to-use agents for instant productivity.
*   **Agent Interaction:** Easily run and interact with agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights for continuous improvement.

## Quick Start

*   **Self-Hosting:** AutoGPT can be self-hosted. See the "How to Self-Host the AutoGPT Platform" section below.
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Public release coming soon!).

## How to Self-Host the AutoGPT Platform

> [!NOTE]
> Self-hosting requires some technical knowledge. For an easier experience, consider joining the cloud-hosted beta.

### System Requirements

Ensure your system meets these requirements before installation:

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

### Setup Instructions

For detailed self-hosting instructions, visit the official documentation:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

### Quick Setup (Recommended)

For local hosting, use the following one-line script:

**For macOS/Linux:**

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

**For Windows (PowerShell):**

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script installs dependencies, configures Docker, and launches your local instance.

## AutoGPT Frontend

The frontend is your gateway to interacting with your AI agents. It provides the following features:

*   **Agent Builder:** Customize agents with a low-code interface.
*   **Workflow Management:** Build and refine automation workflows.
*   **Deployment Controls:** Manage agent lifecycles, from testing to production.
*   **Ready-to-Use Agents:** Access a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with your agents.
*   **Monitoring and Analytics:** Track performance and optimize processes.

## AutoGPT Server

The server is the engine that powers your AI agents. It includes:

*   **Source Code:** The core logic driving agents and automations.
*   **Infrastructure:** Systems that ensure reliable and scalable performance.
*   **Marketplace:** A marketplace for finding and deploying pre-built agents.

## Example Agents

Here are a couple of examples of what AutoGPT can do:

1.  **Generate Viral Videos:** Automatically create short-form videos from trending topics on Reddit.
2.  **Extract Quotes for Social Media:** Transcribe YouTube videos, identify impactful quotes, and generate social media posts.

## AutoGPT Classic

Explore the classic version of AutoGPT, including its components.

*   **Forge:** Ready-to-use toolkit for building agent applications.
*   **Benchmark:** Measure agent performance with a stringent testing environment.
*   **UI:** User-friendly interface for agent control and monitoring.
*   **CLI:** Command-line interface for easy tool usage.

## License Overview

*   **Polyform Shield License:** All code and content within the `autogpt_platform` folder.
*   **MIT License:** All other parts of the AutoGPT repository.

## Resources

*   📖 [Documentation](https://docs.agpt.co)
*   🚀 [Contributing](CONTRIBUTING.md)
*   🤔 [Questions? Problems? Suggestions?](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)
*   💬 [Discord](https://discord.gg/autogpt)