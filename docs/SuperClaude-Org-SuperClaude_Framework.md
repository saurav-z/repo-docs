# SuperClaude Framework: Supercharge Your Claude Code for Enhanced Development (🚀)

[Visit the SuperClaude Framework on GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework)

SuperClaude Framework transforms your Claude code into a structured development platform, unlocking new levels of efficiency and power.

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
  <a href="#quick-installation">Quick Start</a> •
  <a href="#support-the-project">Support</a> •
  <a href="#whats-new-in-v4">Features</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Key Features & Benefits

*   **Structured Development Platform:**  Transforms Claude Code into a structured and organized development environment.
*   **Enhanced Efficiency:** Automates workflows and streamlines your development process.
*   **Intelligent Agents:**  Includes 14 specialized AI agents with domain expertise, accelerating your work.
*   **Modular Architecture:** Integrates with 6 powerful MCP servers for various tasks.
*   **Adaptive Modes:**  Provides 5 behavioral modes to adapt to different development contexts.
*   **Optimized Performance:**  A smaller framework that allows more context for your code, enabling longer conversations and complex operations.
*   **Comprehensive Documentation:** Offers real-world examples, common pitfalls, and practical workflows for developers.

---

## Quick Installation

Choose your preferred method to get started:

| Method | Command | Best For |
|---|---|---|
| **🐍 pipx** | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **✅ Recommended** - Linux/macOS |
| **📦 pip** | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install` | Traditional Python environments |
| **🌐 npm** | `npm install -g @bifrost_inc/superclaude && superclaude install` | Cross-platform, Node.js users |

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

## What's New in V4

Explore the latest enhancements in SuperClaude Framework V4:

*   **🤖 Smarter Agent System:** 14 domain-specific agents for improved capabilities.
*   **📝 Improved Namespace:** `/sc:` prefix for clear command structure.
*   **🔧 MCP Server Integration:** 6 powerful server integrations.
*   **🎯 Behavioral Modes:** 5 modes for adaptive and optimized operations.
*   **⚡ Optimized Performance:** Reduced framework footprint for efficient operations.
*   **📚 Documentation Overhaul:** Comprehensive documentation rewrite for developers.

---

## Documentation

Access the SuperClaude Framework documentation for detailed information and guidance:

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
  *All 21 slash commands*

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

- ✨ [**Best Practices**](Docs/Reference/quick-start-practices.md)
  *Pro tips & patterns*

- 📓 [**Examples Cookbook**](Docs/Reference/examples-cookbook.md)
  *Real-world recipes*

- 🔍 [**Troubleshooting**](Docs/Reference/troubleshooting.md)
  *Common issues & fixes*

</td>
</tr>
</table>

---

## Support the Project

Support the continued development and maintenance of SuperClaude Framework.

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

| Item | Cost/Impact |
|---|---|
| 🔬 **Claude Max Testing** | $100/month for validation & testing |
| ⚡ **Feature Development** | New capabilities & improvements |
| 📚 **Documentation** | Comprehensive guides & examples |
| 🤝 **Community Support** | Quick issue responses & help |
| 🔧 **MCP Integration** | Testing new server connections |
| 🌐 **Infrastructure** | Hosting & deployment costs |

---

## Contributing

Join the SuperClaude community and contribute!  Here's how:

| Priority | Area | Description |
|---|---|---|
| 📝 **High** | Documentation | Improve guides, add examples, fix typos |
| 🔧 **High** | MCP Integration | Add server configs, test integrations |
| 🎯 **Medium** | Workflows | Create command patterns & recipes |
| 🧪 **Medium** | Testing | Add tests, validate features |
| 🌐 **Low** | i18n | Translate docs to other languages |

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

This project is licensed under the **MIT License**.  See the [LICENSE](LICENSE) file for details.

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
```
Key improvements and explanations:

*   **SEO Optimization:**  Included keywords like "Claude code," "development platform," "AI agents," "workflow automation," and project-relevant terms in the headings and descriptions. This increases the chances of the project appearing in search results.
*   **Clear Structure:** Used clear headings and subheadings to improve readability and organization. This helps users quickly find the information they need.
*   **Concise Language:** Rewrote text to be more direct and easier to understand.  Removed unnecessary words.
*   **Bulleted Key Features:**  Provided a list of key features to highlight the main benefits of the framework. This helps users quickly grasp what the project offers.
*   **One-Sentence Hook:**  Provided a compelling one-sentence description to grab the reader's attention immediately and make the project more appealing.
*   **Call to Action:** Added links to contribute to the project.
*   **Improved Formatting:**  Maintained the original formatting while making it more visually appealing, using consistent alignment and spacing.
*   **Clear Upgrade Instructions:** Kept the upgrade instructions, making them very clear.
*   **Removed Redundancy:** Streamlined the content to avoid repetition.
*   **Enhanced Content:** Added a "Key Features" section to make the project more discoverable.
*   **Combined sections:** Some sections were consolidated for better flow.
*   **More descriptive text:** Replaced some of the shorter headings with more descriptive ones.
*   **Focused on benefits:** Emphasized the benefits of using the SuperClaude Framework for the user.