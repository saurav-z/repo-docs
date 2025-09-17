# SuperClaude Framework: Transform Claude Code into a Structured Development Platform

**Supercharge your development workflow with SuperClaude, a powerful framework that structures and enhances Claude code for increased productivity.**  ([Original Repository](https://github.com/SuperClaude-Org/SuperClaude_Framework))

[![Mentioned in Awesome Claude Code](https://awesome.re/mentioned-badge-flat.svg)](https://github.com/hesreallyhim/awesome-claude-code/)
[![Version](https://img.shields.io/badge/version-4.1.1-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework/releases)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/pulls)
[![Website](https://img.shields.io/badge/🌐_Visit_Website-blue)](https://superclaude.netlify.app/)
[![PyPI](https://img.shields.io/pypi/v/SuperClaude.svg?)](https://pypi.org/project/SuperClaude/)
[![npm](https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg)](https://www.npmjs.com/package/@bifrost_inc/superclaude)
[![English](https://img.shields.io/badge/🇺🇸_English-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README.md)
[![中文](https://img.shields.io/badge/🇨🇳_中文-red)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README-zh.md)
[![日本語](https://img.shields.io/badge/🇯🇵_日本語-green)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README-ja.md)

---

## Key Features

*   **Slash Command Interface:** Execute 24 commands, including debugging, code generation, and deployment.
*   **Specialized AI Agents:** Leverage 14 expert agents with domain-specific expertise for enhanced problem-solving.
*   **Adaptive Behavioral Modes:** Utilize 6 modes to optimize for brainstorming, strategic analysis, and more.
*   **Powerful MCP Server Integrations:** Connect with 6 servers for advanced functionality like documentation, testing, and transformations.
*   **Optimized Performance:** Benefit from a smaller footprint, enabling more context and longer conversations.
*   **Comprehensive Documentation:** Explore detailed guides and real-world examples to maximize your workflow.

---

## Installation

### Choose Your Installation Method

| Method  | Command                                                                   | Best For                                    |
| :------ | :------------------------------------------------------------------------ | :------------------------------------------ |
| **🐍 pipx** | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **✅ Recommended** - Linux/macOS       |
| **📦 pip**  | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install`  | Traditional Python environments |
| **🌐 npm**  | `npm install -g @bifrost_inc/superclaude && superclaude install`    | Cross-platform, Node.js users               |

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

## Project Overview

SuperClaude is a meta-programming configuration framework that transforms Claude Code into a structured development platform via behavioral instruction injection and component orchestration, offering powerful tools and intelligent agents for systematic workflow automation.

---

## Support the Project

*Your support helps keep SuperClaude active and improves the framework.*

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

*   **Claude Max Testing:** $100/month for validation & testing
*   **Feature Development:** New capabilities & improvements
*   **Documentation:** Comprehensive guides & examples
*   **Community Support:** Quick issue responses & help
*   **MCP Integration:** Testing new server connections
*   **Infrastructure:** Hosting & deployment costs

---

## What's New in V4

*   **Smarter Agent System:** 14 Specialized agents with domain expertise.
*   **Improved Command Namespace:** `/sc:` prefix for all commands.
*   **MCP Server Integration:** 6 powerful servers working together.
*   **Behavioral Modes:** 6 Adaptive modes for different contexts.
*   **Optimized Performance:** Reduced framework footprint and improved context handling.
*   **Documentation Overhaul:** Complete rewrite with real examples and improved navigation.

---

## Documentation

### Comprehensive Documentation to guide you:

<table>
<tr>
<th align="center">🚀 Getting Started</th>
<th align="center">📖 User Guides</th>
<th align="center">🛠️ Developer Resources</th>
<th align="center">📋 Reference</th>
</tr>
<tr>
<td valign="top">

- 📝 [**Quick Start Guide**](Docs/Getting-Started/quick-start.md)  
  *Get up and running fast*

- 💾 [**Installation Guide**](Docs/Getting-Started/installation.md)  
  *Detailed setup instructions*

</td>
<td valign="top">

- 🎯 [**Commands Reference**](Docs/User-Guide/commands.md)  
  *All 23 slash commands*

- 🤖 [**Agents Guide**](Docs/User-Guide/agents.md)  
  *14 specialized agents*

- 🎨 [**Behavioral Modes**](Docs/User-Guide/modes.md)  
  *5 adaptive modes*

- 🚩 [**Flags Guide**](Docs/User-Guide/flags.md)  
  *Control behaviors*

- 🔧 [**MCP Servers**](Docs/User-Guide/mcp-servers.md)  
  *6 server integrations*

- 💼 [**Session Management**](Docs/User-Guide/session-management.md)  
  *Save & restore state*

</td>
<td valign="top">

- 🏗️ [**Technical Architecture**](Docs/Developer-Guide/technical-architecture.md)  
  *System design details*

- 💻 [**Contributing Code**](Docs/Developer-Guide/contributing-code.md)  
  *Development workflow*

- 🧪 [**Testing & Debugging**](Docs/Developer-Guide/testing-debugging.md)  
  *Quality assurance*

</td>
<td valign="top">
- 📓 [**Examples Cookbook**](Docs/Reference/examples-cookbook.md)  
  *Real-world recipes*

- 🔍 [**Troubleshooting**](Docs/Reference/troubleshooting.md)  
  *Common issues & fixes*

</td>
</tr>
</table>

---

## Contributing

We welcome contributions! Help us by:

| Priority | Area           | Description                                   |
| :------- | :------------- | :-------------------------------------------- |
| 📝 **High**   | Documentation  | Improve guides, add examples, fix typos      |
| 🔧 **High**   | MCP Integration | Add server configs, test integrations         |
| 🎯 **Medium** | Workflows      | Create command patterns & recipes            |
| 🧪 **Medium** | Testing        | Add tests, validate features                 |
| 🌐 **Low**    | i18n           | Translate docs to other languages             |

<p align="center">
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/📖_Read-Contributing_Guide-blue" alt="Contributing Guide">
  </a>
  <a href="https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors">
    <img src="https://img.shields.io/badge/👥_View-All_Contributors-green" alt="Contributors">
  </a>
</p>

---

## License

This project is licensed under the **MIT License**.

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

### Built with passion by the SuperClaude community

<p align="center">
  <sub>Made with ❤️ for developers who push boundaries</sub>
</p>

<p align="center">
  <a href="#-superclaude-framework">Back to Top ↑</a>
</p>
```
Key improvements and explanations:

*   **SEO-Optimized Title and Hook:** The title is clear and includes relevant keywords like "Claude Code" and "Development Platform." The hook sentence immediately tells the user what the framework does and its key benefit.
*   **Clear Headings:** Consistent use of headings (H2, H3) for better organization and readability. This is crucial for both human readers and search engine crawlers.
*   **Bulleted Key Features:**  Used bullet points to make the key features stand out and easy to scan.  Each feature is concise.
*   **Concise Descriptions:**  Descriptions are brief and focus on the benefits of each feature.
*   **Installation Section:**  The installation instructions are kept, as that is important to the user, but made a bit more readable with a table.  I added an extra heading before it.
*   **Support the Project:** The Support section is cleaned up for better readability and uses the badges/links.
*   **Documentation Organization:** The documentation section is kept and reorganized to make it more user-friendly
*   **Contributing Section:** Updated with a table to show contributors what types of contributions are most desired.
*   **License and Other Info:** Kept license, star history, and community info.
*   **Conciseness:** Removed unnecessary text while preserving all the important information.
*   **Markdown Formatting:** Proper markdown formatting for headings, lists, and links is used for better readability and SEO.
*   **Back to Top Link** Added a clear "Back to Top" link for easy navigation.
*   **Repo Link** Explicitly links back to the original repo at the beginning.

This revised README is much more user-friendly, easier to understand, and more likely to be found by users searching for a solution like SuperClaude.