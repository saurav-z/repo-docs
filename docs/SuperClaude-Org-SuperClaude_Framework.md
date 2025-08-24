# Supercharge Your Development with SuperClaude: The AI-Powered Framework (🚀)

**SuperClaude** is a powerful framework that transforms Claude Code into a structured, intelligent development platform, streamlining your workflow and boosting productivity.  ([View on GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework))

<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.8-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome">
</p>

<p align="center">
  <a href="https://superclaude.netlify.app/">
    <img src="https://img.shields.io/badge/🌐_Visit_Website-blue?style=for-the-badge" alt="Website">
  </a>
  <a href="https://pypi.org/project/SuperClaude/">
    <img src="https://img.shields.io/pypi/v/SuperClaude.svg?style=for-the-badge&label=PyPI" alt="PyPI">
  </a>
  <a href="https://www.npmjs.com/package/@bifrost_inc/superclaude">
    <img src="https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg?style=for-the-badge&label=npm" alt="npm">
  </a>
</p>


## Key Features

*   **AI-Powered Agents:** Leverage 14 specialized AI agents, each with domain expertise, for tasks like security analysis and UI design.
*   **Enhanced Command Structure:**  Utilize a streamlined command structure with the `/sc:` prefix, providing a clear and organized way to manage your projects with 21 commands.
*   **Powerful Server Integrations:** Integrate with 6 advanced MCP servers for features like complex analysis, UI generation, and session persistence.
*   **Adaptive Behavioral Modes:** Choose from 5 behavioral modes optimized for different contexts, including brainstorming, orchestration, and token efficiency.
*   **Optimized Performance:** Experience a reduced framework footprint and improved context handling for more complex and longer conversations.
*   **Comprehensive Documentation:** Access a complete rewrite of the documentation with practical examples and workflow guides.

## Quick Installation

Install SuperClaude using your preferred method:

*   **pipx (Recommended):**  `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` (for Linux/macOS)
*   **pip:**  `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install` (for traditional Python environments)
*   **npm:** `npm install -g @bifrost_inc/superclaude && superclaude install` (for Node.js users)

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


## Framework Statistics

<div align="center">

| **Commands** | **Agents** | **Modes** | **MCP Servers** |
|:------------:|:----------:|:---------:|:---------------:|
| **21** | **14** | **5** | **6** |
| Slash Commands | Specialized AI | Behavioral | Integrations |

</div>


##  Support the Project

Support SuperClaude development through these options:

<table>
<tr>
<td align="center" width="33%">
  
### ☕ **Ko-fi**
[![Ko-fi](https://img.shields.io/badge/Support_on-Ko--fi-ff5e5b?style=for-the-badge&logo=ko-fi)](https://ko-fi.com/superclaude)

*One-time contributions*

</td>
<td align="center" width="33%">

### 🎯 **Patreon**
[![Patreon](https://img.shields.io/badge/Become_a-Patron-f96854?style=for-the-badge&logo=patreon)](https://patreon.com/superclaude)

*Monthly support*

</td>
<td align="center" width="33%">

### 💜 **GitHub**
[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsor-30363D?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/SuperClaude-Org)

*Flexible tiers*

</td>
</tr>
</table>

**Your Support Enables:**

*   🔬 **Claude Max Testing:**  Ensuring the framework is up-to-date.
*   ⚡ **Feature Development:** Adding new capabilities and improvements.
*   📚 **Documentation:** Creating comprehensive guides and examples.
*   🤝 **Community Support:** Providing quick responses to issues and assistance.
*   🔧 **MCP Integration:** Testing new server connections.
*   🌐 **Infrastructure:** Covering hosting and deployment costs.


## What's New in V4

V4 introduces substantial enhancements based on community feedback:

<div align="center">

<table>
<tr>
<td width="50%">

### 🤖 **Smarter Agent System**
**14 specialized agents** with domain expertise:
- Security engineer catches real vulnerabilities
- Frontend architect understands UI patterns
- Automatic coordination based on context
- Domain-specific expertise on demand

</td>
<td width="50%">

### 📝 **Improved Namespace**
**`/sc:` prefix** for all commands:
- No conflicts with custom commands
- 21 commands covering full lifecycle
- From brainstorming to deployment
- Clean, organized command structure

</td>
</tr>
<tr>
<td width="50%">

### 🔧 **MCP Server Integration**
**6 powerful servers** working together:
- **Context7** → Up-to-date documentation
- **Sequential** → Complex analysis
- **Magic** → UI component generation
- **Playwright** → Browser testing
- **Morphllm** → Bulk transformations
- **Serena** → Session persistence

</td>
<td width="50%">

### 🎯 **Behavioral Modes**
**5 adaptive modes** for different contexts:
- **Brainstorming** → Asks right questions
- **Orchestration** → Efficient tool coordination
- **Token-Efficiency** → 30-50% context savings
- **Task Management** → Systematic organization
- **Introspection** → Meta-cognitive analysis

</td>
</tr>
<tr>
<td width="50%">

### ⚡ **Optimized Performance**
**Smaller framework, bigger projects:**
- Reduced framework footprint
- More context for your code
- Longer conversations possible
- Complex operations enabled

</td>
<td width="50%">

### 📚 **Documentation Overhaul**
**Complete rewrite** for developers:
- Real examples & use cases
- Common pitfalls documented
- Practical workflows included
- Better navigation structure

</td>
</tr>
</table>

</div>

## Documentation

Comprehensive resources to get you started and help you master SuperClaude:

<div align="center">

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

</div>

## Contributing

Help build SuperClaude! We welcome contributions of all kinds.

| Priority | Area | Description |
|:--------:|------|-------------|
| 📝 **High** | Documentation | Improve guides, add examples, fix typos |
| 🔧 **High** | MCP Integration | Add server configs, test integrations |
| 🎯 **Medium** | Workflows | Create command patterns & recipes |
| 🧪 **Medium** | Testing | Add tests, validate features |
| 🌐 **Low** | i18n | Translate docs to other languages |

<div align="center">
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/📖_Read-Contributing_Guide-blue?style=for-the-badge" alt="Contributing Guide">
  </a>
  <a href="https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors">
    <img src="https://img.shields.io/badge/👥_View-All_Contributors-green?style=for-the-badge" alt="Contributors">
  </a>
</div>

## License

This project is licensed under the [MIT License](LICENSE).

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License">
</p>

## Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
 </picture>
</a>

## Built with passion by the SuperClaude community

<p align="center">
  <sub>Made with ❤️ for developers who push boundaries</sub>
</p>

<p align="center">
  <a href="#-superclaude-framework">Back to Top ↑</a>
</p>
```
Key improvements and reasons:

*   **SEO Optimization:**  Uses relevant keywords like "AI," "framework," "Claude," and "development" throughout.
*   **Clear Headings:**  Uses descriptive headings to break up the content and improve readability.
*   **Bulleted Key Features:**  Highlights key benefits in a concise format.  This is easier for users to scan and understand quickly.
*   **One-Sentence Hook:** The opening sentence grabs attention and summarizes the core value proposition.
*   **Conciseness:**  Removes redundant phrases and focuses on essential information.
*   **Action-Oriented Language:**  Uses phrases like "Supercharge Your Development" and "Quick Installation" to encourage engagement.
*   **Improved Formatting:**  Consistent use of bolding, code blocks, and tables makes the information easier to digest.
*   **Contextualized Troubleshooting:** Troubleshooting steps are kept and in line with best practices.
*   **Complete Coverage:** The most important parts of the original README are kept.
*   **Added Star History:** Enhances the README by allowing users to visualize repository popularity.