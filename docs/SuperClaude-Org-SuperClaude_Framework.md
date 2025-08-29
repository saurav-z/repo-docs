# SuperClaude Framework: Transform Your Claude Code into a Powerful Development Platform

Supercharge your Claude code with SuperClaude, a meta-programming framework that unlocks structured development, workflow automation, and intelligent agents.  [Explore the SuperClaude Framework on GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Version](https://img.shields.io/badge/version-4.0.8-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/pulls)
[![Website](https://img.shields.io/badge/%F0%9F%8C%90_Visit_Website-blue)](https://superclaude.netlify.app/)
[![PyPI](https://img.shields.io/pypi/v/SuperClaude.svg?)](https://pypi.org/project/SuperClaude/)
[![npm](https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg)](https://www.npmjs.com/package/@bifrost_inc/superclaude)
[![English](https://img.shields.io/badge/%F0%9F%87%BA%F0%9F%87%B8_English-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README.md)
[![中文](https://img.shields.io/badge/%F0%9F%87%A8%F0%9F%87%B3_中文-red)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README-zh.md)
[![日本語](https://img.shields.io/badge/%F0%9F%87%AF%F0%9F%87%B5_日本語-green)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README-ja.md)

## Key Features

*   **Intelligent Agents:** Leverage 14 specialized agents with domain expertise.
*   **Slash Commands:**  Utilize 21 commands for a complete development lifecycle.
*   **Behavioral Modes:** Adapt to different contexts with 5 versatile modes.
*   **MCP Server Integration:** Integrate with 6 powerful servers for advanced functionality.
*   **Optimized Performance:** Experience a smaller framework footprint with greater context capacity.
*   **Comprehensive Documentation:**  Benefit from a complete rewrite with real-world examples.

## Framework Statistics

| Commands      | Agents      | Modes       | MCP Servers |
| :------------ | :---------- | :---------- | :---------- |
| **21**        | **14**      | **5**       | **6**       |
| Slash Commands | Specialized AI | Behavioral | Integrations |

## Quick Installation

Choose the installation method that best suits your environment:

| Method        | Command                                                      | Best For                    |
| :------------ | :----------------------------------------------------------- | :-------------------------- |
| **🐍 pipx**     | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | ✅ Recommended - Linux/macOS |
| **📦 pip**      | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install`  | Traditional Python environments |
| **🌐 npm**      | `npm install -g @bifrost_inc/superclaude && superclaude install`     | Cross-platform, Node.js users |

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

## Support the Project

Support SuperClaude's development and maintenance:

<table>
<tr>
<td align="center" width="33%">

### ☕ **Ko-fi**
[![Ko-fi](https://img.shields.io/badge/Support_on-Ko--fi-ff5e5b?logo=ko-fi)](https://ko-fi.com/superclaude)

*One-time contributions*

</td>
<td align="center" width="33%">

### 🎯 **Patreon**
[![Patreon](https://img.shields.io/badge/Become_a-Patron-f96854?logo=patreon)](https://patreon.com/superclaude)

*Monthly support*

</td>
<td align="center" width="33%">

### 💜 **GitHub**
[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsor-30363D?logo=github-sponsors)](https://github.com/sponsors/SuperClaude-Org)

*Flexible tiers*

</td>
</tr>
</table>

### Your Support Enables:

| Item                      | Cost/Impact                      |
| :------------------------ | :------------------------------- |
| 🔬 **Claude Max Testing**   | $100/month for validation & testing  |
| ⚡ **Feature Development** | New capabilities & improvements |
| 📚 **Documentation**       | Comprehensive guides & examples  |
| 🤝 **Community Support**   | Quick issue responses & help    |
| 🔧 **MCP Integration**     | Testing new server connections   |
| 🌐 **Infrastructure**      | Hosting & deployment costs       |

## What's New in V4

Explore the latest enhancements in SuperClaude V4:

*   **Smarter Agent System:** 14 specialized agents for domain-specific expertise.
*   **Improved Namespace:** `/sc:` prefix for all commands.
*   **MCP Server Integration:** 6 powerful servers working together for advanced tasks.
*   **Behavioral Modes:** 5 adaptive modes for different development contexts.
*   **Optimized Performance:** Reduced framework footprint for enhanced capacity.
*   **Documentation Overhaul:** Complete rewrite with real examples and practical workflows.

## Documentation

Comprehensive documentation to guide you through SuperClaude:

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

*   ✨ [**Best Practices**](Docs/Reference/quick-start-practices.md)
*   📓 [**Examples Cookbook**](Docs/Reference/examples-cookbook.md)
*   🔍 [**Troubleshooting**](Docs/Reference/troubleshooting.md)

## Contributing

Join the SuperClaude community and contribute to its growth:

| Priority | Area          | Description                       |
| :------- | :------------ | :-------------------------------- |
| 📝 **High**   | Documentation   | Improve guides, add examples, fix typos |
| 🔧 **High**   | MCP Integration | Add server configs, test integrations |
| 🎯 **Medium** | Workflows     | Create command patterns & recipes  |
| 🧪 **Medium** | Testing       | Add tests, validate features       |
| 🌐 **Low**    | i18n          | Translate docs to other languages |

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

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?" alt="MIT License">
</p>

## Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
 </picture>
</a>

---

### Built with passion by the SuperClaude community

<sub>Made with ❤️ for developers who push boundaries</sub>