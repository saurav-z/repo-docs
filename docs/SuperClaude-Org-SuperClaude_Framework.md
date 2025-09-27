# 🤖 SuperClaude Framework: Transform Your Claude Code into a Powerful Development Platform

Supercharge your development workflow with the **SuperClaude Framework**, a meta-programming platform designed to enhance and structure your Claude code. [Explore the SuperClaude Framework on GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework).

<p align="center">
  <a href="https://github.com/hesreallyhim/awesome-claude-code/">
  <img src="https://awesome.re/mentioned-badge-flat.svg" alt="Mentioned in Awesome Claude Code">
  </a>
<a href="https://github.com/SuperClaude-Org/SuperGemini_Framework" target="_blank">
  <img src="https://img.shields.io/badge/Try-SuperGemini_Framework-blue" alt="Try SuperGemini Framework"/>
</a>
<a href="https://github.com/SuperClaude-Org/SuperQwen_Framework" target="_blank">
  <img src="https://img.shields.io/badge/Try-SuperQwen_Framework-orange" alt="Try SuperQwen Framework"/>
</a>
  <img src="https://img.shields.io/badge/version-4.2.0-blue" alt="Version">
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
  <a href="#-quick-installation">Quick Start</a> •
  <a href="#-support-the-project">Support</a> •
  <a href="#-whats-new-in-v4">Features</a> •
  <a href="#-documentation">Docs</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## Key Features

*   **25+ Commands:** Access a comprehensive suite of commands for streamlined development.
*   **15+ Specialized Agents:** Leverage AI-powered agents with domain expertise.
*   **7+ Adaptive Modes:** Tailor your workflow with intelligent behavioral modes.
*   **8+ MCP Server Integrations:** Connect to powerful servers for enhanced functionality.
*   **Deep Research Capabilities:** Advanced features for autonomous web research.

---

## Quick Installation

Choose your preferred method:

| Method | Command | Best For |
|:------:|---------|----------|
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

V4 brings significant improvements including:

*   **Smarter Agent System:** 15 specialized agents for various domains.
*   **Improved Namespace:** `/sc:` prefix for all commands.
*   **MCP Server Integration:** Integration with 8 powerful servers.
*   **Behavioral Modes:** 7 adaptive modes for different contexts.
*   **Optimized Performance:** Reduced framework footprint.
*   **Documentation Overhaul:** Comprehensive documentation for developers.

---

## Deep Research Capabilities

### Autonomous Web Research Features:

*   **Adaptive Planning:** Intelligent strategies for efficient research.
*   **Multi-Hop Reasoning:** Up to 5 iterative searches.
*   **Quality Scoring:** Confidence-based validation.
*   **Case-Based Learning:** Cross-session intelligence.

### Research Command Usage

```bash
# Basic research with automatic depth
/sc:research "latest AI developments 2024"

# Controlled research depth
/sc:research "quantum computing breakthroughs" --depth exhaustive

# Specific strategy selection
/sc:research "market analysis" --strategy planning-only

# Domain-filtered research
/sc:research "React patterns" --domains "reactjs.org,github.com"
```

### Research Depth Levels

| Depth | Sources | Hops | Time | Best For |
|:-----:|:-------:|:----:|:----:|----------|
| **Quick** | 5-10 | 1 | ~2min | Quick facts, simple queries |
| **Standard** | 10-20 | 3 | ~5min | General research (default) |
| **Deep** | 20-40 | 4 | ~8min | Comprehensive analysis |
| **Exhaustive** | 40+ | 5 | ~10min | Academic-level research |

### Integrated Tool Orchestration

The Deep Research system intelligently coordinates multiple tools:
- **Tavily MCP**: Primary web search and discovery
- **Playwright MCP**: Complex content extraction
- **Sequential MCP**: Multi-step reasoning and synthesis
- **Serena MCP**: Memory and learning persistence
- **Context7 MCP**: Technical documentation lookup

---

## Documentation

Comprehensive guides to help you get started:

### Complete Guide to SuperClaude

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
  *All 25 slash commands*

- 🤖 [**Agents Guide**](Docs/User-Guide/agents.md)  
  *15 specialized agents*

- 🎨 [**Behavioral Modes**](Docs/User-Guide/modes.md)  
  *7 adaptive modes*

- 🚩 [**Flags Guide**](Docs/User-Guide/flags.md)  
  *Control behaviors*

- 🔧 [**MCP Servers**](Docs/User-Guide/mcp-servers.md)  
  *7 server integrations*

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

## Support the Project

> Maintaining SuperClaude requires time and resources. Your support helps to cover costs and keep development active.

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

### **Your Support Enables:**

| Item | Cost/Impact |
|------|-------------|
| 🔬 **Claude Max Testing** | $100/month for validation & testing |
| ⚡ **Feature Development** | New capabilities & improvements |
| 📚 **Documentation** | Comprehensive guides & examples |
| 🤝 **Community Support** | Quick issue responses & help |
| 🔧 **MCP Integration** | Testing new server connections |
| 🌐 **Infrastructure** | Hosting & deployment costs |

> **Note:** Contributing code, documentation, or spreading the word also helps! 🙏

---

## Contributing

We welcome contributions! Find out how you can contribute by reading the [Contributing Guide](CONTRIBUTING.md).

| Priority | Area | Description |
|:--------:|------|-------------|
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

### Built with passion by the SuperClaude community

<p align="center">
  <sub>Made with ❤️ for developers who push boundaries</sub>
</p>

<p align="center">
  <a href="#-superclaude-framework">Back to Top ↑</a>
</p>