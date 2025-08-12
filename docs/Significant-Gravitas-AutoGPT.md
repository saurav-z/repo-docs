# AutoGPT: Automate and Innovate with AI Agents 🚀

**Unlock the power of AI with AutoGPT, the leading platform for building, deploying, and managing autonomous AI agents.**  [Explore the original repository](https://github.com/Significant-Gravitas/AutoGPT).

[![Discord](https://img.shields.io/discord/1098755955990097950?label=Discord&logo=discord&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

## Key Features

*   **Build & Customize:** Design your own AI agents with an intuitive, low-code Agent Builder.
*   **Automated Workflows:** Easily build, modify, and optimize automation workflows.
*   **Deployment & Management:** Manage the lifecycle of your agents, from testing to production.
*   **Pre-built Agents:** Utilize a library of ready-to-use agents to get started instantly.
*   **Agent Interaction:** Run and interact with your agents through a user-friendly interface.
*   **Performance Insights:** Monitor and analyze your agents' performance to continuously improve.

## Hosting Options

*   **Self-Host (Free!):** Download and run AutoGPT on your own infrastructure.
*   **Cloud-Hosted Beta (Coming Soon!):**  Join the [waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta and experience AutoGPT in the cloud.

## Getting Started (Self-Hosting)

Self-hosting AutoGPT allows for complete control over your AI agents.

### System Requirements

Ensure your system meets these requirements for optimal performance:

#### Hardware:
*   CPU: 4+ cores recommended
*   RAM: Minimum 8GB, 16GB recommended
*   Storage: At least 10GB of free space

#### Software:
*   Operating Systems: Linux (Ubuntu 20.04 or newer recommended), macOS (10.15 or newer), Windows 10/11 with WSL2
*   Required Software (with minimum versions): Docker Engine (20.10.0 or newer), Docker Compose (2.0.0 or newer), Git (2.30 or newer), Node.js (16.x or newer), npm (8.x or newer), VSCode (1.60 or newer) or any modern code editor

#### Network:
*   Stable internet connection
*   Access to required ports (configured in Docker)
*   Ability to make outbound HTTPS connections

### Setup Instructions

For the most up-to-date instructions, please refer to the official documentation:

👉  [Self-Hosting Guide](https://docs.agpt.co/platform/getting-started/)

#### Quick Setup with One-Line Script (Local Hosting)

Automate the setup process with our convenient script:

For macOS/Linux:
```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This script installs dependencies, configures Docker, and launches your local instance.

## AutoGPT Platform Components

### 🧱 AutoGPT Frontend

The user-friendly interface for interacting with your AI agents:

*   **Agent Builder:** Design and configure custom agents.
*   **Workflow Management:** Build and optimize automation workflows.
*   **Deployment Controls:** Manage the agent lifecycle.
*   **Ready-to-Use Agents:** Deploy pre-configured agents.
*   **Agent Interaction:** Run and interact with your agents.
*   **Monitoring & Analytics:** Track performance and improve automation.

[Learn to build custom blocks](https://docs.agpt.co/platform/new_blocks/).

### 💽 AutoGPT Server

The engine that powers your AI agents:

*   **Source Code:** The core logic for agents and automation.
*   **Infrastructure:** Robust systems for reliable performance.
*   **Marketplace:** A marketplace to find and deploy pre-built agents.

## 🐙 Example Agents

Discover the possibilities:

1.  **Generate Viral Videos from Trending Topics:**  Automatically create short-form videos based on trending topics identified on Reddit.
2.  **Identify Top Quotes from Videos for Social Media:**  Transcribe your YouTube videos, identify impactful quotes, and generate social media posts.

Create custom workflows and build agents for any use case!

---

## Licensing

*   🛡️ **Polyform Shield License:** Applied to code and content within the `autogpt_platform` folder. [Read more](https://agpt.co/blog/introducing-the-autogpt-platform)
*   🦉 **MIT License:** All other parts of the repository, including [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge), [agbenchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark), [AutoGPT Classic GUI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend), and additional projects like [GravitasML](https://github.com/Significant-Gravitas/gravitasml) and [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability).

---

## Mission

*   🏗️ **Building:** Lay the foundation for amazing applications.
*   🧪 **Testing:** Fine-tune your agents to perfection.
*   🤝 **Delegating:** Let AI work for you and bring your ideas to life.

Join the AI revolution!

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---

## 🤖 AutoGPT Classic

Explore the original AutoGPT tools:

**🛠️ [Build Your Own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

Forge is a ready-to-go toolkit for building your own agent applications. Tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec).

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md)

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge)

### 🎯 Benchmark

The `agbenchmark` offers a stringent testing environment for your agents.

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark)

### 💻 UI

The frontend provides a user-friendly interface to control and monitor your agents.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend)

### ⌨️ CLI

Use the CLI to easily manage tools.

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

Report issues via [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose).

## 🤝 Sister projects

### 🔄 Agent Protocol

AutoGPT uses the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation to standardize communication pathways.

---

## Stars Stats

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

*   **Concise Hook:**  A clear, compelling sentence immediately introduces AutoGPT.
*   **Keyword Optimization:** Uses relevant keywords like "AI agents," "automation," "build," "deploy," "manage."
*   **Clear Headings:**  Organizes content logically with descriptive headings.
*   **Bulleted Lists:** Makes key features and information easy to scan.
*   **Strong Calls to Action:** Encourages users to explore the documentation, contribute, and join the community.
*   **Internal Linking:** Links to different sections within the README (e.g., "Getting Started") and external resources.
*   **Visual Appeal:** Includes Discord and Twitter badges.
*   **Focus on Benefits:** Highlights the *value* to the user (e.g., automate workflows, build custom agents).
*   **SEO-friendly Language:** Uses language that people might search for (e.g., "AI agent platform," "build AI agents").
*   **Mobile-Friendly:** Markdown formatting adapts well to different screen sizes.
*   **Concise Descriptions:** Avoids overly long paragraphs, making it easier for users to quickly understand the key points.
*   **Comprehensive Coverage:** Includes information about both the AutoGPT Platform and the Classic version.
*   **Clear Separation of Concerns:** Differentiates between the core platform and the classic tools.
*   **Updated Documentation Links:**  Directs users to the most current documentation.
*   **Star History Graph:** A visual element for demonstrating project popularity.