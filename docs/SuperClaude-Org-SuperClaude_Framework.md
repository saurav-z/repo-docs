# SuperClaude Framework: Revolutionizing Claude Code Development

**Supercharge your Claude code with the SuperClaude Framework, transforming it into a structured, powerful development platform for streamlined workflows.**  [View the original repository on GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework)

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

## Key Features

*   **Structured Development Platform:** Transforms Claude code into a structured development environment.
*   **Intelligent Agents:** Features 14 specialized AI agents for various development tasks.
*   **Command-Line Tools:** Offers 22 powerful slash commands for a complete development lifecycle.
*   **Behavioral Modes:** Includes 6 adaptive modes for different development contexts.
*   **MCP Server Integration:** Leverages 6 powerful server integrations for enhanced functionality.
*   **Streamlined Workflows:** Automates workflows and improves efficiency with powerful tools.
*   **Comprehensive Documentation:** Features extensive guides, examples, and troubleshooting resources.
*   **Open Source and Community Driven:** Welcomes contributions and supports a thriving community.

## Quick Installation

Choose your preferred installation method:

| Method    | Command                                                                    | Best For                               |
| --------- | -------------------------------------------------------------------------- | -------------------------------------- |
| **pipx**  | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **✅ Recommended** - Linux/macOS        |
| **pip**   | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install` | Traditional Python environments        |
| **npm**   | `npm install -g @bifrost_inc/superclaude && superclaude install`          | Cross-platform, Node.js users          |

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
*   **Improved Namespace:** `/sc:` prefix for all commands for cleaner organization.
*   **MCP Server Integration:** 6 powerful servers for enhanced functionality.
*   **Behavioral Modes:** 6 adaptive modes for diverse development needs.
*   **Optimized Performance:** Reduced footprint for larger projects and longer conversations.
*   **Documentation Overhaul:** Comprehensive rewrite with practical examples and workflows.

## Documentation

Find detailed guides and references to get you started:

### Getting Started

*   📝 [Quick Start Guide](Docs/Getting-Started/quick-start.md)
*   💾 [Installation Guide](Docs/Getting-Started/installation.md)

### User Guides

*   🎯 [Commands Reference](Docs/User-Guide/commands.md)
*   🤖 [Agents Guide](Docs/User-Guide/agents.md)
*   🎨 [Behavioral Modes](Docs/User-Guide/modes.md)
*   🚩 [Flags Guide](Docs/User-Guide/flags.md)
*   🔧 [MCP Servers](Docs/User-Guide/mcp-servers.md)
*   💼 [Session Management](Docs/User-Guide/session-management.md)

### Developer Resources

*   🏗️ [Technical Architecture](Docs/Developer-Guide/technical-architecture.md)
*   💻 [Contributing Code](Docs/Developer-Guide/contributing-code.md)
*   🧪 [Testing & Debugging](Docs/Developer-Guide/testing-debugging.md)

### Reference

*   ✨ [Best Practices](Docs/Reference/quick-start-practices.md)
*   📓 [Examples Cookbook](Docs/Reference/examples-cookbook.md)
*   🔍 [Troubleshooting](Docs/Reference/troubleshooting.md)

## Support the Project

[Insert support information and options here, but rephrase the tone]

SuperClaude is a community-driven project that requires ongoing maintenance and development.  If you find value in SuperClaude, please consider supporting the project. Your contributions help cover operational costs and facilitate continuous improvement.

Support options include:

*   ☕ **Ko-fi:** [Ko-fi link](https://ko-fi.com/superclaude) (one-time contributions)
*   🎯 **Patreon:** [Patreon link](https://patreon.com/superclaude) (monthly support)
*   💜 **GitHub Sponsors:** [GitHub Sponsors link](https://github.com/sponsors/SuperClaude-Org) (flexible tiers)

## Contributing

Contribute to SuperClaude and help improve the project!

| Priority | Area          | Description                                      |
| :------- | :------------ | :----------------------------------------------- |
| 📝 High   | Documentation | Improve guides, add examples, fix typos          |
| 🔧 High   | MCP Integration | Add server configs, test integrations          |
| 🎯 Medium | Workflows     | Create command patterns & recipes             |
| 🧪 Medium | Testing       | Add tests, validate features                    |
| 🌐 Low    | i18n          | Translate docs to other languages                |

*   📖 [Contributing Guide](CONTRIBUTING.md)
*   👥 [View All Contributors](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

## License

This project is licensed under the [MIT License](LICENSE).

## Star History

[Insert Star History chart here]