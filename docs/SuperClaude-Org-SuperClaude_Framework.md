# SuperClaude Framework: Transform Your Claude Code into a Powerful Development Platform

**Supercharge your development workflow with SuperClaude, a meta-programming framework that unlocks the full potential of Claude AI.** Explore the [SuperClaude Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework) for a new way to code.

<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.8-blue" alt="Version">
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

## Key Features

*   **Advanced Agents:** Leverage 14 specialized AI agents for tasks like security, UI design, and more.
*   **Organized Commands:** Utilize a clear `/sc:` prefix for 22 commands, streamlining your workflow from start to finish.
*   **Powerful Server Integrations:** Connect with 6 powerful MCP servers for documentation, complex analysis, UI generation, testing, and more.
*   **Adaptive Behavioral Modes:** Choose from 6 modes tailored to brainstorming, strategic analysis, and efficient task management.
*   **Optimized Performance:** Enjoy a smaller framework footprint, allowing for more context and longer conversations.
*   **Comprehensive Documentation:** Access a complete rewrite with practical examples and improved navigation.

## Installation

### Quick Start: Choose Your Installation Method

| Method    | Command                                                        | Best For                     |
| :-------- | :------------------------------------------------------------- | :--------------------------- |
| **🐍 pipx** | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **✅ Recommended** - Linux/macOS |
| **📦 pip**  | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install` | Traditional Python environments |
| **🌐 npm**  | `npm install -g @bifrost_inc/superclaude && superclaude install` | Cross-platform, Node.js users |

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

## What's New in V4

*   **Smarter Agent System:** 14 specialized agents for domain-specific expertise.
*   **Improved Namespace:** `/sc:` prefix for organized command structure.
*   **MCP Server Integration:** 6 powerful servers working together.
*   **Behavioral Modes:** 6 adaptive modes for different contexts.
*   **Optimized Performance:** Smaller framework footprint, enabling longer conversations.
*   **Documentation Overhaul:** Complete rewrite with real examples and improved navigation.

## Documentation

Comprehensive documentation to guide you through SuperClaude.

| Getting Started                               | User Guides                                  | Developer Resources                        | Reference                                    |
| :-------------------------------------------- | :------------------------------------------- | :----------------------------------------- | :------------------------------------------- |
| - [Quick Start Guide](Docs/Getting-Started/quick-start.md)  | - [Commands Reference](Docs/User-Guide/commands.md)  | - [Technical Architecture](Docs/Developer-Guide/technical-architecture.md)  | - [Best Practices](Docs/Reference/quick-start-practices.md)  |
| - [Installation Guide](Docs/Getting-Started/installation.md)  | - [Agents Guide](Docs/User-Guide/agents.md)  | - [Contributing Code](Docs/Developer-Guide/contributing-code.md)  | - [Examples Cookbook](Docs/Reference/examples-cookbook.md)  |
|                                               | - [Behavioral Modes](Docs/User-Guide/modes.md)  | - [Testing & Debugging](Docs/Developer-Guide/testing-debugging.md)  | - [Troubleshooting](Docs/Reference/troubleshooting.md)  |
|                                               | - [Flags Guide](Docs/User-Guide/flags.md)  |                                            |                                              |
|                                               | - [MCP Servers](Docs/User-Guide/mcp-servers.md)  |                                            |                                              |
|                                               | - [Session Management](Docs/User-Guide/session-management.md)  |                                            |                                              |

## Support the Project

Your support helps maintain and improve SuperClaude. Consider contributing via:

*   **Ko-fi:** [Support on Ko-fi](https://ko-fi.com/superclaude) (One-time contributions)
*   **Patreon:** [Become a Patron](https://patreon.com/superclaude) (Monthly support)
*   **GitHub Sponsors:** [GitHub Sponsor](https://github.com/sponsors/SuperClaude-Org) (Flexible tiers)

## Contributing

Join the SuperClaude community and contribute to the project.

| Priority | Area          | Description                        |
| :------- | :------------ | :--------------------------------- |
| 📝 High   | Documentation | Improve guides, add examples, fix typos |
| 🔧 High   | MCP Integration | Add server configs, test integrations |
| 🎯 Medium | Workflows     | Create command patterns & recipes |
| 🧪 Medium | Testing       | Add tests, validate features      |
| 🌐 Low    | i18n          | Translate docs to other languages |

*   **Read the Contributing Guide:**  [Contributing Guide](CONTRIBUTING.md)
*   **View All Contributors:**  [Contributors](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

## License

This project is licensed under the **MIT License**.  See the [LICENSE](LICENSE) file for details.

## Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
 </picture>
</a>

---

**[Back to Top ↑](#superclaude-framework)**