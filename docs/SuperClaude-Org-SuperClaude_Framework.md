# SuperClaude Framework: Revolutionize Your Claude Code Development

**Transform your Claude code into a structured development platform with the SuperClaude Framework, a powerful tool for automation and intelligent agent orchestration.**  [Explore the original repository](https://github.com/SuperClaude-Org/SuperClaude_Framework).

<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.9-blue" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

<p align="center">
  <a href="https://superclaude.netlify.app/">
    <img src="https://img.shields.io/badge/🌐_Visit_Website-blue" alt="Website">
  </a>
  <a href="https://pypi.org/project/SuperClaude/">
    <img src="https://img.shields.io/pypi/v/SuperClaude.svg?" alt="PyPI">
  </a>
  <a href="https://www.npmjs.com/package/@bifrost_inc/superclaude">
    <img src="https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg" alt="npm">
  </a>
</p>

<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/🇺🇸_English-blue" alt="English">
  </a>
  <a href="README-zh.md">
    <img src="https://img.shields.io/badge/🇨🇳_中文-red" alt="中文">
  </a>
  <a href="README-ja.md">
    <img src="https://img.shields.io/badge/🇯🇵_日本語-green" alt="日本語">
  </a>
</p>

<p align="center">
  <a href="#quick-installation">Quick Start</a> •
  <a href="#support-the-project">Support</a> •
  <a href="#whats-new-in-v4">Features</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Key Features

*   **Intelligent Agents:** Leverage 14 specialized AI agents for tasks like security, frontend architecture, and more.
*   **Command-Line Interface:** Utilize 22 slash commands (prefixed with `/sc:`) for a streamlined development workflow, from brainstorming to deployment.
*   **Powerful MCP Server Integrations:** Seamlessly connect to 6 server integrations (Context7, Sequential, Magic, Playwright, Morphllm, Serena) for diverse functionalities.
*   **Adaptive Behavioral Modes:** Choose from 6 modes (Brainstorming, Business Panel, Orchestration, Token-Efficiency, Task Management, Introspection) to optimize for different project stages.
*   **Optimized Performance:** Experience a reduced framework footprint, enabling more context for your code and longer, complex operations.
*   **Comprehensive Documentation:** Benefit from a complete documentation rewrite, including real-world examples, common pitfalls, and practical workflows.

---

## 📊 Framework Statistics

| **Commands** | **Agents** | **Modes** | **MCP Servers** |
|:------------:|:----------:|:---------:|:---------------:|
| **22** | **14** | **6** | **6** |
| Slash Commands | Specialized AI | Behavioral | Integrations |

---

## ⚡ Quick Installation

Choose the installation method that best suits your needs:

| Method         | Command                                                       | Best For                    |
| :------------- | :------------------------------------------------------------ | :-------------------------- |
| **🐍 pipx**      | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **✅ Recommended** - Linux/macOS |
| **📦 pip**       | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install` | Traditional Python environments |
| **🌐 npm**       | `npm install -g @bifrost_inc/superclaude && superclaude install` | Cross-platform, Node.js users |

<details>
<summary><b>⚠️ IMPORTANT: Upgrading from SuperClaude V3</b></summary>

**If you have SuperClaude V3 installed, you SHOULD uninstall it before installing V4:**

```bash
# Uninstall V3 first
Remove all related files and directories :
*.md *.json and commands/

# Then install V4
pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install
```

**✅ What gets preserved during upgrade:**
- ✓ Your custom slash commands (outside `commands/sc/`)
- ✓ Your custom content in `CLAUDE.md` 
- ✓ Claude Code's `.claude.json`, `.credentials.json`, `settings.json` and `settings.local.json`
- ✓ Any custom agents and files you've added

**⚠️ Note:** Other SuperClaude-related `.json` files from V3 may cause conflicts and should be removed.

</details>

<details>
<summary><b>💡 Troubleshooting PEP 668 Errors</b></summary>

```bash
# Option 1: Use pipx (Recommended)
pipx install SuperClaude

# Option 2: User installation
pip install --user SuperClaude

# Option 3: Force installation (use with caution)
pip install --break-system-packages SuperClaude
```
</details>

---

## 💖 Support the Project

Your support helps keep SuperClaude alive and thriving! Consider contributing:

*   **Ko-fi:** [Ko-fi Link](https://ko-fi.com/superclaude) - One-time contributions
*   **Patreon:** [Patreon Link](https://patreon.com/superclaude) - Monthly support
*   **GitHub Sponsors:** [GitHub Sponsors Link](https://github.com/sponsors/SuperClaude-Org) - Flexible tiers

**Your contributions support:**

*   Claude Max Testing
*   Feature Development
*   Documentation
*   Community Support
*   MCP Integration Testing
*   Infrastructure

---

## 🎉 What's New in V4

*   **Smarter Agent System:** 14 specialized agents with domain expertise
*   **Improved Namespace:** `/sc:` prefix for all commands
*   **MCP Server Integration:** 6 powerful servers working together
*   **Behavioral Modes:** 6 adaptive modes for different contexts
*   **Optimized Performance:** Smaller framework, bigger projects
*   **Documentation Overhaul:** Complete rewrite for developers

---

## 📚 Documentation

Explore comprehensive guides and resources:

| **Getting Started**                                                                                                              | **User Guides**                                                                                                                     | **Developer Resources**                                                                                                           | **Reference**                                                                                            |
| :---------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| 📝 [Quick Start Guide](Docs/Getting-Started/quick-start.md)  Get up and running fast  <br> 💾 [Installation Guide](Docs/Getting-Started/installation.md)  Detailed setup instructions                                                                                              | 🎯 [Commands Reference](Docs/User-Guide/commands.md)  All 22 slash commands  <br> 🤖 [Agents Guide](Docs/User-Guide/agents.md)  14 specialized agents  <br> 🎨 [Behavioral Modes](Docs/User-Guide/modes.md)  5 adaptive modes <br> 🚩 [Flags Guide](Docs/User-Guide/flags.md)  Control behaviors <br> 🔧 [MCP Servers](Docs/User-Guide/mcp-servers.md)  6 server integrations <br> 💼 [Session Management](Docs/User-Guide/session-management.md)  Save & restore state | 🏗️ [Technical Architecture](Docs/Developer-Guide/technical-architecture.md)  System design details  <br> 💻 [Contributing Code](Docs/Developer-Guide/contributing-code.md)  Development workflow <br> 🧪 [Testing & Debugging](Docs/Developer-Guide/testing-debugging.md)  Quality assurance | ✨ [Best Practices](Docs/Reference/quick-start-practices.md)  Pro tips & patterns  <br> 📓 [Examples Cookbook](Docs/Reference/examples-cookbook.md)  Real-world recipes  <br> 🔍 [Troubleshooting](Docs/Reference/troubleshooting.md)  Common issues & fixes |

---

## 🤝 Contributing

Join the SuperClaude community!

*   **High Priority:** Documentation, MCP integration,
*   **Medium Priority:** Workflows, testing
*   **Low Priority:** i18n

<p align="center">
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/📖_Read-Contributing_Guide-blue" alt="Contributing Guide">
  </a>
  <a href="https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors">
    <img src="https://img.shields.io/badge/👥_View-All_Contributors-green" alt="Contributors">
  </a>
</p>

---

## ⚖️ License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?" alt="MIT License">
</p>

---

## ⭐ Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
 </picture>
</a>

---

### **🚀 Built with passion by the SuperClaude community**

<p align="center">
  <sub>Made with ❤️ for developers who push boundaries</sub>
</p>

<p align="center">
  <a href="#superclaude-framework">Back to Top ↑</a>
</p>