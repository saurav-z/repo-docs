# SuperClaude Framework: Supercharge Your Claude Code with Intelligent Automation

**SuperClaude Framework** transforms Claude code into a powerful development platform, enhancing your workflow with structured development and AI-powered tools. [Check out the original repository](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Version](https://img.shields.io/badge/version-4.0.8-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/pulls)
[![Website](https://img.shields.io/badge/🌐_Visit_Website-blue)](https://superclaude.netlify.app/)
[![PyPI](https://img.shields.io/pypi/v/SuperClaude.svg?)](https://pypi.org/project/SuperClaude/)
[![npm](https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg)](https://www.npmjs.com/package/@bifrost_inc/superclaude)

## Key Features

*   **Intelligent Agents:** 14 specialized AI agents for various development tasks.
*   **Enhanced Command Structure:**  /sc: prefix for a clean and organized command system with 21 commands.
*   **MCP Server Integration:** Integration with 6 powerful servers for extended capabilities like context, analysis, and UI generation.
*   **Adaptive Behavioral Modes:** 5 modes to optimize for different contexts and improve token efficiency.
*   **Optimized Performance:** Reduced framework footprint for more complex projects and longer conversations.
*   **Comprehensive Documentation:** Rewritten documentation with practical examples and clear workflows.

## Quick Installation

Choose your preferred method:

| Method      | Command                                                                  | Best For                     |
|-------------|--------------------------------------------------------------------------|------------------------------|
| **🐍 pipx** | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **Recommended** (Linux/macOS) |
| **📦 pip**  | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install`  | Traditional Python Environments |
| **🌐 npm**  | `npm install -g @bifrost_inc/superclaude && superclaude install`           | Cross-platform (Node.js)     |

## Important Upgrade Notes

**If upgrading from SuperClaude V3:**

Uninstall V3 before installing V4:

```bash
# Uninstall V3 first
Remove all related files and directories :
*.md *.json and commands/

# Then install V4
pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install
```

**Preserved:**

*   Custom slash commands (outside `commands/sc/`)
*   Custom content in `CLAUDE.md`
*   `.claude.json`, `.credentials.json`, `settings.json`, `settings.local.json`
*   Custom agents and files

**Remove:** Other SuperClaude-related `.json` files from V3 may cause conflicts and should be removed.

## Troubleshooting PEP 668 Errors

```bash
# Option 1: Use pipx (Recommended)
pipx install SuperClaude

# Option 2: User installation
pip install --user SuperClaude

# Option 3: Force installation (use with caution)
pip install --break-system-packages SuperClaude
```

## Support the Project

Your support helps maintain and develop SuperClaude. Contributions of any size are appreciated.

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

## What's New in V4

V4 introduces significant improvements based on user feedback:

<table>
<tr>
<td width="50%">

### 🤖 Smarter Agent System

*   **14 specialized agents** with domain expertise.
*   Security engineer catches real vulnerabilities
*   Frontend architect understands UI patterns
*   Automatic coordination based on context
*   Domain-specific expertise on demand

</td>
<td width="50%">

### 📝 Improved Namespace

*   **/sc:** prefix for all commands
*   No conflicts with custom commands
*   21 commands covering full lifecycle
*   From brainstorming to deployment
*   Clean, organized command structure

</td>
</tr>
<tr>
<td width="50%">

### 🔧 MCP Server Integration

*   **6 powerful servers** working together.
*   **Context7** → Up-to-date documentation
*   **Sequential** → Complex analysis
*   **Magic** → UI component generation
*   **Playwright** → Browser testing
*   **Morphllm** → Bulk transformations
*   **Serena** → Session persistence

</td>
<td width="50%">

### 🎯 Behavioral Modes

*   **5 adaptive modes** for different contexts.
*   **Brainstorming** → Asks right questions
*   **Orchestration** → Efficient tool coordination
*   **Token-Efficiency** → 30-50% context savings
*   **Task Management** → Systematic organization
*   **Introspection** → Meta-cognitive analysis

</td>
</tr>
<tr>
<td width="50%">

### ⚡ Optimized Performance

*   **Smaller framework, bigger projects.**
*   Reduced framework footprint
*   More context for your code
*   Longer conversations possible
*   Complex operations enabled

</td>
<td width="50%">

### 📚 Documentation Overhaul

*   **Complete rewrite** for developers.
*   Real examples & use cases
*   Common pitfalls documented
*   Practical workflows included
*   Better navigation structure

</td>
</tr>
</table>

## Documentation

Comprehensive documentation to get you started and help you master SuperClaude.

### 🚀 Getting Started

*   📝 [**Quick Start Guide**](Docs/Getting-Started/quick-start.md)  
    *Get up and running fast*
*   💾 [**Installation Guide**](Docs/Getting-Started/installation.md)  
    *Detailed setup instructions*

### 📖 User Guides

*   🎯 [**Commands Reference**](Docs/User-Guide/commands.md)  
    *All 21 slash commands*
*   🤖 [**Agents Guide**](Docs/User-Guide/agents.md)  
    *14 specialized agents*
*   🎨 [**Behavioral Modes**](Docs/User-Guide/modes.md)  
    *5 adaptive modes*
*   🚩 [**Flags Guide**](Docs/User-Guide/flags.md)  
    *Control behaviors*
*   🔧 [**MCP Servers**](Docs/User-Guide/mcp-servers.md)  
    *6 server integrations*
*   💼 [**Session Management**](Docs/User-Guide/session-management.md)  
    *Save & restore state*

### 🛠️ Developer Resources

*   🏗️ [**Technical Architecture**](Docs/Developer-Guide/technical-architecture.md)  
    *System design details*
*   💻 [**Contributing Code**](Docs/Developer-Guide/contributing-code.md)  
    *Development workflow*
*   🧪 [**Testing & Debugging**](Docs/Developer-Guide/testing-debugging.md)  
    *Quality assurance*

### 📋 Reference

*   ✨ [**Best Practices**](Docs/Reference/quick-start-practices.md)  
    *Pro tips & patterns*
*   📓 [**Examples Cookbook**](Docs/Reference/examples-cookbook.md)  
    *Real-world recipes*
*   🔍 [**Troubleshooting**](Docs/Reference/troubleshooting.md)  
    *Common issues & fixes*

## Contributing

We welcome contributions!

| Priority | Area          | Description                   |
|:--------:|---------------|-------------------------------|
| 📝 **High** | Documentation | Improve guides, add examples, fix typos |
| 🔧 **High** | MCP Integration | Add server configs, test integrations |
| 🎯 **Medium** | Workflows     | Create command patterns & recipes |
| 🧪 **Medium** | Testing       | Add tests, validate features    |
| 🌐 **Low**  | i18n          | Translate docs to other languages |

<p align="center">
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/📖_Read-Contributing_Guide-blue" alt="Contributing Guide">
  </a>
  <a href="https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors">
    <img src="https://img.shields.io/badge/👥_View-All_Contributors-green" alt="Contributors">
  </a>
</p>

## License

This project is licensed under the **MIT License**.  See the [LICENSE](LICENSE) file for details.

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

## Built with Passion

This project is built by the SuperClaude community, made with ❤️ for developers.

[Back to Top ↑](#-superclaude-framework)