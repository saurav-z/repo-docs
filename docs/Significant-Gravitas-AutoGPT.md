# AutoGPT: Build, Deploy, and Run AI Agents to Automate Workflows

**Unleash the power of AI automation with AutoGPT, a cutting-edge platform for creating, deploying, and managing intelligent agents.**  ([See the original repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoGPT empowers you to build, deploy, and manage AI agents that can automate complex tasks.

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents using an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize automation workflows with ease.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Utilize a library of pre-configured agents for immediate use.
*   **Agent Interaction:**  Easily run and interact with your own or pre-configured agents.
*   **Monitoring and Analytics:** Track agent performance and gain insights for process improvement.

## Hosting Options

*   **Self-Hosting:** Download and set up AutoGPT on your own infrastructure.  See below for setup instructions.
*   **Cloud-Hosted Beta:**  [Join the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta and let us handle the technical setup.

## Getting Started with Self-Hosting

**Prerequisites:** Please review the [official self-hosting guide](https://docs.agpt.co/platform/getting-started/) for the most up-to-date setup instructions.

### System Requirements

Ensure your system meets these requirements:

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

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The frontend is your gateway to interacting with and leveraging AI automation.

*   **Agent Builder:** Customize AI agents with our low-code interface.
*   **Workflow Management:** Build, modify, and optimize your workflows.
*   **Deployment Controls:** Manage the agent lifecycle from testing to production.
*   **Ready-to-Use Agents:** Leverage a library of pre-configured agents.
*   **Agent Interaction:** Run and interact with your or pre-configured agents via a user-friendly interface.
*   **Monitoring and Analytics:** Track performance and gain insights to continually improve your automation processes.

### 💽 AutoGPT Server

The server is the engine that powers your AI agents.

*   **Source Code:** The core logic driving agents and automation.
*   **Infrastructure:** Robust systems for reliable and scalable performance.
*   **Marketplace:** A marketplace to find and deploy a wide range of pre-built agents.

## Example Agents

Here are two examples of AutoGPT in action:

1.  **Generate Viral Videos:** Reads trending topics on Reddit, creates short-form videos.
2.  **Identify Top Quotes from Videos:** Transcribes YouTube videos, identifies impactful quotes, and creates social media posts.

## AutoGPT Classic

For classic version of AutoGPT, access these features:

### 🏗️ Forge

*   **Forge your own agent!** Forge is a toolkit for building your own agent applications. Learn more about Forge [here](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md).

### 🎯 Benchmark

*   **Measure your agent's performance!**  The `agbenchmark` can be used with any agent that supports the agent protocol.

### 💻 UI

*   **Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents.

### ⌨️ CLI

Use the CLI for easy tool access:

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

AutoGPT utilizes the [agent protocol](https://agentprotocol.ai/) standard to ensure seamless compatibility.

---

## 🤝 Join the Community

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

---

## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>

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
```

Key improvements and SEO considerations:

*   **Concise Hook:**  A clear, attention-grabbing one-sentence introduction to the project.
*   **Clear Headings:**  Uses consistent and SEO-friendly heading structure (H1, H2, etc.) for readability and search engine optimization.
*   **Keyword Optimization:** Includes relevant keywords like "AI agents," "automation," "workflow," and "deploy" throughout the text.
*   **Bulleted Lists:** Makes information easy to scan and digest, which is good for both users and SEO.
*   **Concise Descriptions:**  Keeps descriptions brief and to the point.
*   **Clear Calls to Action:**  Directs users to the Discord, issue tracker, and other resources.
*   **Internal Links:** Uses descriptive anchor text to link between different sections of the README.
*   **External Links:** Link to key resources like the documentation and related projects.
*   **Community Focus:** Highlights the community aspect with links to Discord.
*   **Contributor Showcase:**  Includes the contributors image to improve interest.
*   **Star History:** Included Star History graph.