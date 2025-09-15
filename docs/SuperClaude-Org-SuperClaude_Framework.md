# 🚀 SuperClaude Framework: Unleash the Power of Claude Code

**Supercharge your development workflow with SuperClaude, the ultimate meta-programming framework that transforms Claude Code into a structured and powerful platform.** Explore the original repository at [https://github.com/SuperClaude-Org/SuperClaude_Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework).

<p align="center">
  <a href="https://github.com/hesreallyhim/awesome-claude-code/">
  <img src="https://awesome.re/mentioned-badge-flat.svg" alt="Mentioned in Awesome Claude Code">
  </a>
  <img src="https://img.shields.io/badge/version-4.1.0-blue" alt="Version"> 
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

*   **Powerful Agents:** Utilize 14 specialized AI agents for domain expertise, from security to frontend architecture.
*   **Structured Commands:** Execute 23 commands using the `/sc:` prefix, providing a clean and organized command structure.
*   **Advanced Server Integrations:** Leverage 6 MCP servers for documentation, analysis, UI generation, testing, and more.
*   **Adaptive Modes:** Choose from 6 behavioral modes for brainstorming, strategic analysis, orchestration, and token efficiency.
*   **Optimized Performance:** Experience a reduced framework footprint for more context and longer conversations.
*   **Comprehensive Documentation:** Access a complete guide with examples, troubleshooting, and practical workflows.

## Quick Installation

Choose your preferred method:

| Method      | Command                                                                   | Best For                                   |
| ----------- | ------------------------------------------------------------------------- | ------------------------------------------ |
| **🐍 pipx**  | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **✅ Recommended** - Linux/macOS           |
| **📦 pip**   | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install`  | Traditional Python environments            |
| **🌐 npm**   | `npm install -g @bifrost_inc/superclaude && superclaude install`            | Cross-platform, Node.js users             |

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

*   **Smarter Agent System:** 14 specialized agents with domain expertise.
*   **Improved Namespace:** `/sc:` prefix for all commands, preventing conflicts.
*   **MCP Server Integration:** 6 powerful servers working together for advanced features.
*   **Behavioral Modes:** 6 adaptive modes for different contexts.
*   **Optimized Performance:** Smaller footprint, enabling larger projects.
*   **Documentation Overhaul:** Complete rewrite with real examples and practical workflows.

## Documentation

Explore the comprehensive SuperClaude documentation:

### Getting Started

*   📝 [**Quick Start Guide**](Docs/Getting-Started/quick-start.md)
*   💾 [**Installation Guide**](Docs/Getting-Started/installation.md)

### User Guides

*   🎯 [**Commands Reference**](Docs/User-Guide/commands.md)
*   🤖 [**Agents Guide**](Docs/User-Guide/agents.md)
*   🎨 [**Behavioral Modes**](Docs/User-Guide/modes.md)
*   🚩 [**Flags Guide**](Docs/User-Guide/flags.md)
*   🔧 [**MCP Servers**](Docs/User-Guide/mcp-servers.md)
*   💼 [**Session Management**](Docs/User-Guide/session-management.md)

### Developer Resources

*   🏗️ [**Technical Architecture**](Docs/Developer-Guide/technical-architecture.md)
*   💻 [**Contributing Code**](Docs/Developer-Guide/contributing-code.md)
*   🧪 [**Testing & Debugging**](Docs/Developer-Guide/testing-debugging.md)

### Reference

*   📓 [**Examples Cookbook**](Docs/Reference/examples-cookbook.md)
*   🔍 [**Troubleshooting**](Docs/Reference/troubleshooting.md)

## Support the Project

Support SuperClaude and its community:

*   ☕ [**Ko-fi**](https://ko-fi.com/superclaude) - One-time contributions
*   🎯 [**Patreon**](https://patreon.com/superclaude) - Monthly support
*   💜 [**GitHub Sponsors**](https://github.com/sponsors/SuperClaude-Org) - Flexible tiers

## Contributing

We welcome contributions! Here's how you can help:

| Priority | Area        | Description                      |
| :------- | :---------- | :------------------------------- |
| 📝 **High**  | Documentation | Improve guides, add examples. |
| 🔧 **High**  | MCP Integration | Add server configs, test.    |
| 🎯 **Medium**| Workflows     | Create command patterns.      |
| 🧪 **Medium**| Testing       | Add tests, validate.           |
| 🌐 **Low**  | i18n        | Translate docs.                |

<p align="center">
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/📖_Read-Contributing_Guide-blue" alt="Contributing Guide">
  </a>
  <a href="https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors">
    <img src="https://img.shields.io/badge/👥_View-All_Contributors-green" alt="Contributors">
  </a>
</p>

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
 </picture>
</a>

### Built with passion by the SuperClaude community.