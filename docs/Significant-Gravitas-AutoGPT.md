# AutoGPT: Unleash the Power of AI Agents to Automate Anything

**AutoGPT empowers you to build, deploy, and manage AI agents that automate complex tasks and transform workflows, offering a new era of AI-driven efficiency.** ([Back to Top](#autogpt-unleash-the-power-of-ai-agents-to-automate-anything))

[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://zdoc.app/de/Significant-Gravitas/AutoGPT) | 
[Español](https://zdoc.app/es/Significant-Gravitas/AutoGPT) | 
[français](https://zdoc.app/fr/Significant-Gravitas/AutoGPT) | 
[日本語](https://zdoc.app/ja/Significant-Gravitas/AutoGPT) | 
[한국어](https://zdoc.app/ko/Significant-Gravitas/AutoGPT) | 
[Português](https://zdoc.app/pt/Significant-Gravitas/AutoGPT) | 
[Русский](https://zdoc.app/ru/Significant-Gravitas/AutoGPT) | 
[中文](https://zdoc.app/zh/Significant-Gravitas/AutoGPT)

## Key Features

*   **AI Agent Creation:** Design and configure custom AI agents using a user-friendly, low-code interface.
*   **Workflow Automation:** Build, modify, and optimize automation workflows easily.
*   **Agent Management:** Deploy, monitor, and manage the entire lifecycle of your AI agents.
*   **Pre-built Agents:** Leverage a library of ready-to-use, pre-configured agents for immediate automation.
*   **Agent Interaction:** Seamlessly run and interact with your agents through an intuitive interface.
*   **Performance Analytics:** Track agent performance and gain valuable insights to continually improve your automation processes.

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own infrastructure.  [See System Requirements](#system-requirements)
*   **Cloud-Hosted Beta:** [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta. (Closed Beta - Public release Coming Soon!)

## Getting Started: Self-Hosting

> [!NOTE]
> Self-hosting AutoGPT requires technical expertise. For an easier experience, consider joining the cloud-hosted beta.

### System Requirements

Ensure your system meets these prerequisites before installation:

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

### Installation

To get started, follow these installation steps:

1.  **Visit the official documentation:** [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/) for detailed instructions.

#### ⚡ Quick Setup (Recommended)

Simplify the setup process with our automatic script.

For macOS/Linux:
```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script automates dependency installation, Docker configuration, and local instance launching.

---

## 🧱 AutoGPT Platform: Frontend & Server

### 🧱 AutoGPT Frontend

The AutoGPT frontend offers a user-friendly interface for interacting with and managing AI agents:

*   **Agent Builder:** Easily design and configure custom AI agents with our low-code interface.
*   **Workflow Management:** Build, modify, and optimize your automation workflows effortlessly.
*   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
*   **Ready-to-Use Agents:** Quickly deploy and utilize pre-configured agents from our library.
*   **Agent Interaction:** Run and interact with your agents through our intuitive interface.
*   **Monitoring and Analytics:** Track agent performance and gain valuable insights to improve automation processes.

[Learn how to build your own custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The AutoGPT Server provides the core infrastructure for running your AI agents:

*   **Source Code:** The underlying logic driving our agents and automation processes.
*   **Infrastructure:** Robust systems ensuring reliable and scalable performance.
*   **Marketplace:** Find and deploy a wide range of pre-built agents.

---

## 🐙 Example Agents

AutoGPT enables a wide range of automation possibilities:

1.  **Generate Viral Videos from Trending Topics:** Automatically creates short-form videos based on trending topics from Reddit.

2.  **Identify Top Quotes from Videos for Social Media:** Transcribes YouTube videos, identifies impactful quotes, generates summaries, and automatically posts to social media.

---

## 🛡️ Licensing

*   **Polyform Shield License:** The `autogpt_platform` folder is licensed under the Polyform Shield License. [Read more](https://agpt.co/blog/introducing-the-autogpt-platform)
*   **MIT License:** All other parts of the repository, including the original AutoGPT agent and related projects, are licensed under the MIT License.

---

## Mission

Our mission is to empower you to:

*   🏗️ **Build**: Lay the foundation for amazing automation.
*   🧪 **Test**: Fine-tune your agents for optimal performance.
*   🤝 **Delegate**: Leverage AI to bring your ideas to life.

Be at the forefront of AI innovation!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic (Legacy Version)

> Information about the classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forge your own agent!**  Forge is a ready-to-go toolkit to build your own agent application.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### 🎯 Benchmark

**Measure your agent's performance!**  The `agbenchmark` can be used with any agent that supports the agent protocol.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

**Makes agents easy to use!**

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

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

Just clone the repo, install dependencies with `./run setup`, and you should be good to go!

---

## 🤔 Questions? Problems? Suggestions?

### Get Help

[Discord 💬](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

---

## 🤝 Sister Projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) for seamless communication with agents.

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

[Back to Top](#autogpt-unleash-the-power-of-ai-agents-to-automate-anything)
```
Key improvements and SEO optimizations:

*   **Concise Hook:** A strong opening sentence that grabs attention and clearly states the value proposition.
*   **Clear Headings:**  Uses H2 and H3 headings to organize content logically, improving readability and SEO.  Includes a Back to Top link.
*   **Keyword Optimization:** Includes relevant keywords like "AI agents," "automation," "workflows," "deploy," and "manage" throughout the text.
*   **Bulleted Lists:** Uses bullet points to highlight key features and benefits, making information easy to scan.
*   **Action-Oriented Language:**  Uses verbs like "build," "deploy," "manage," "automate" to engage the reader.
*   **Structured Content:**  Organizes information into logical sections with clear subheadings.
*   **SEO-Friendly Formatting:** Uses bolding and italics for emphasis, and links for internal and external navigation.
*   **Clear Calls to Action:** Encourages users to join the waitlist, read the documentation, and contribute.
*   **Mobile-Friendly Design:**  The Markdown formatting translates well across different devices.
*   **Comprehensive:** Includes all the original content but reformats it to be more user-friendly and search-engine optimized.
*   **Relevance:** Includes all the information to make the documentation usable.
*   **Focus on Value:** Highlights the benefits and value the project offers to users.