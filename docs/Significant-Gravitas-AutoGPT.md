# AutoGPT: Unleash the Power of AI Agents for Automated Workflows

**Automate complex tasks and streamline your processes with AutoGPT, the leading platform for building, deploying, and running AI agents.** ([Original Repo](https://github.com/Significant-Gravitas/AutoGPT))

[![Discord](https://img.shields.io/discord/1095702848108922920?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt)
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://zdoc.app/de/Significant-Gravitas/AutoGPT) | 
[Español](https://zdoc.app/es/Significant-Gravitas/AutoGPT) | 
[français](https://zdoc.app/fr/Significant-Gravitas/AutoGPT) | 
[日本語](https://zdoc.app/ja/Significant-Gravitas/AutoGPT) | 
[한국어](https://zdoc.app/ko/Significant-Gravitas/AutoGPT) | 
[Português](https://zdoc.app/pt/Significant-Gravitas/AutoGPT) | 
[Русский](https://zdoc.app/ru/Significant-Gravitas/AutoGPT) | 
[中文](https://zdoc.app/zh/Significant-Gravitas/AutoGPT)

## Key Features of AutoGPT

*   **AI Agent Creation:** Design and customize AI agents with an intuitive, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Easily manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Get started immediately with a library of ready-to-use, pre-configured agents.
*   **Agent Interaction:** Run and interact with your custom or pre-configured agents through a user-friendly interface.
*   **Monitoring and Analytics:** Track agent performance and gain insights to improve your automation processes.

## Hosting Options

*   **Self-Hosting:** Download and self-host the platform for free. (See requirements below)
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Coming Soon!).

## Getting Started: Self-Hosting AutoGPT

> [!NOTE]
> If you'd prefer an easy-to-use, fully managed experience, [join the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

### System Requirements

Ensure your system meets these requirements before installation:

#### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: Minimum 8GB, 16GB recommended
- Storage: At least 10GB of free space

#### Software Requirements
- Operating Systems:
  - Linux (Ubuntu 20.04 or newer recommended)
  - macOS (10.15 or newer)
  - Windows 10/11 with WSL2
- Required Software (with minimum versions):
  - Docker Engine (20.10.0 or newer)
  - Docker Compose (2.0.0 or newer)
  - Git (2.30 or newer)
  - Node.js (16.x or newer)
  - npm (8.x or newer)
  - VSCode (1.60 or newer) or any modern code editor

#### Network Requirements
- Stable internet connection
- Access to required ports (will be configured in Docker)
- Ability to make outbound HTTPS connections

### Setup Instructions

For the latest and most up-to-date instructions, please refer to the official documentation:

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)

This tutorial assumes you have Docker, VSCode, git and npm installed.

---

#### ⚡ Quick Setup with One-Line Script (Recommended for Local Hosting)

Simplify the setup process with our automated script:

For macOS/Linux:
```
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script handles dependencies, Docker configuration, and launches your local instance automatically.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The AutoGPT frontend provides the user interface for interacting with your AI automation.

*   **Agent Builder:** Design and configure custom AI agents with a user-friendly, low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows.
*   **Deployment Controls:** Manage the lifecycle of your agents.
*   **Ready-to-Use Agents:** Select and deploy pre-configured agents.
*   **Agent Interaction:** Run and interact with agents.
*   **Monitoring and Analytics:** Track performance and improve processes.

[Learn how to build your own custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The AutoGPT Server is the core of the platform, running your agents.

*   **Source Code:** The logic that drives your agents.
*   **Infrastructure:** Reliable, scalable performance.
*   **Marketplace:** Find and deploy pre-built agents.

### 🐙 Example Agents

Real-world examples of what you can achieve with AutoGPT:

1.  **Generate Viral Videos from Trending Topics:** Creates short-form videos based on trending topics.
2.  **Identify Top Quotes from Videos for Social Media:** Identifies impactful quotes from videos and generates social media posts.

## Licensing

*   🛡️ **Polyform Shield License:** Code and content within the `autogpt_platform` folder. [Read more](https://agpt.co/blog/introducing-the-autogpt-platform).
*   🦉 **MIT License:** All other parts of the repository, including AutoGPT Classic, Forge, agbenchmark, and the AutoGPT Classic GUI.

## Mission

Our mission is to empower you to:

*   🏗️ **Build**: Lay the foundation for something amazing.
*   🧪 **Test**: Fine-tune your agent to perfection.
*   🤝 **Delegate**: Let AI work for you, bringing your ideas to life.

Be part of the AI revolution! **AutoGPT** is at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---
## 🤖 AutoGPT Classic

Explore the classic version of AutoGPT below.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forge your own agent!** Forge is a toolkit to build your own agent application. It handles most of the boilerplate code. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec).

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash; This guide walks you through creating and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend) about the Frontend

### ⌨️ CLI

[CLI]: #-cli

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

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) for standardization.

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

*   **Strong Hook:** Starts with a compelling one-sentence introduction optimized for keywords like "AI agents" and "automated workflows."
*   **Clear Headings:** Uses descriptive and SEO-friendly headings (e.g., "Key Features," "Getting Started").
*   **Bulleted Lists:**  Employs bullet points to break up text and improve readability.
*   **Keyword Integration:** Naturally integrates relevant keywords like "AI agents," "automation," "workflows," and "self-hosting" throughout the text.
*   **Emphasis on Benefits:**  Highlights the benefits of using AutoGPT, rather than just listing features.
*   **Concise Language:** Uses clear and concise language to improve comprehension.
*   **Call to Action:** Encourages users to "Join the Waitlist" and explore different components.
*   **Internal Linking:** Includes links to relevant sections within the README.
*   **SEO-Friendly Formatting:**  Uses bold text for emphasis.
*   **Backlink:**  The link back to the repo is clear.
*   **Structured Content:** Organizes content logically, making it easy for both users and search engines to understand.
*   **ALT text for Images:** Added ALT text to images.
*   **Up-to-date:** Updated the requirements and installation to reflect the new documentation and installation scripts.

This improved README is more informative, user-friendly, and optimized for search engines, making it more likely to attract and engage users interested in AI agent technology.