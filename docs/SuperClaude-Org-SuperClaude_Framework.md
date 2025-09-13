# SuperClaude Framework: Supercharge Your Development with AI 🤖

**Transform your Claude code into a structured, AI-powered development platform** with the SuperClaude Framework.  [Explore the original repository](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Version](https://img.shields.io/badge/version-4.0.9-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework/releases)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/pulls)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg?label=PyPI)](https://pypi.org/project/SuperClaude/)
[![NPM version](https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg?label=NPM)](https://www.npmjs.com/package/@bifrost_inc/superclaude)

## Key Features

*   **AI-Powered Agents:** Leverage 14 specialized agents with domain expertise (security, frontend, etc.) for automated and context-aware assistance.
*   **Intuitive Command Structure:** Utilize `/sc:` prefixed commands for a clean, organized lifecycle management (brainstorming, deployment, and more).
*   **Advanced Server Integrations:** Connect to 6 powerful MCP servers for up-to-date documentation, complex analysis, UI generation, browser testing, and session persistence.
*   **Adaptive Behavioral Modes:** Select from 6 modes (Brainstorming, Business Panel, Orchestration, etc.) to tailor the framework's behavior to your needs.
*   **Optimized Performance:** Experience a smaller framework footprint, enabling more context for your code and supporting longer conversations.
*   **Comprehensive Documentation:** Benefit from a complete documentation overhaul with examples, practical workflows, and an improved navigation structure.

## Quick Installation

### Installation Methods

Choose the installation method that best suits your needs:

| Method      | Command                                                                     | Best For                                  |
| :---------- | :-------------------------------------------------------------------------- | :---------------------------------------- |
| **pipx**    | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **Recommended** (Linux/macOS)             |
| **pip**     | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install` | Traditional Python environments         |
| **npm**     | `npm install -g @bifrost_inc/superclaude && superclaude install`             | Cross-platform, Node.js users            |

<details>
<summary><b>⚠️ Upgrading from SuperClaude V3</b></summary>

**If you have SuperClaude V3 installed, UNINSTALL IT BEFORE installing V4:**

```bash
# Uninstall V3 first
# Remove all related files and directories :
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

> Maintaining SuperClaude requires time and resources. Your support helps keep development active and enables further improvements.

Consider supporting the project through:

*   [Ko-fi](https://ko-fi.com/superclaude) - One-time contributions
*   [Patreon](https://patreon.com/superclaude) - Monthly support
*   [GitHub Sponsors](https://github.com/sponsors/SuperClaude-Org) - Flexible tiers

Your support helps fund:

*   Claude Max Testing
*   Feature Development
*   Documentation
*   Community Support
*   MCP Integration
*   Infrastructure

## What's New in V4

*   **Smarter Agent System:** 14 specialized agents with domain expertise.
*   **Improved Namespace:** `/sc:` prefix for all commands.
*   **MCP Server Integration:** 6 powerful server integrations.
*   **Behavioral Modes:** 6 adaptive modes for different contexts.
*   **Optimized Performance:** Reduced framework footprint.
*   **Documentation Overhaul:** Complete rewrite for developers.

## Documentation

Comprehensive guides and resources to help you get started and utilize SuperClaude effectively.

### Getting Started

*   [Quick Start Guide](Docs/Getting-Started/quick-start.md)
*   [Installation Guide](Docs/Getting-Started/installation.md)

### User Guides

*   [Commands Reference](Docs/User-Guide/commands.md)
*   [Agents Guide](Docs/User-Guide/agents.md)
*   [Behavioral Modes](Docs/User-Guide/modes.md)
*   [Flags Guide](Docs/User-Guide/flags.md)
*   [MCP Servers](Docs/User-Guide/mcp-servers.md)
*   [Session Management](Docs/User-Guide/session-management.md)

### Developer Resources

*   [Technical Architecture](Docs/Developer-Guide/technical-architecture.md)
*   [Contributing Code](Docs/Developer-Guide/contributing-code.md)
*   [Testing & Debugging](Docs/Developer-Guide/testing-debugging.md)

### Reference

*   [Examples Cookbook](Docs/Reference/examples-cookbook.md)
*   [Troubleshooting](Docs/Reference/troubleshooting.md)

## Contributing

Join the SuperClaude community and help us build a better framework!

*   **High Priority:** Documentation improvements, MCP server integrations.
*   **Medium Priority:** Workflow creation, testing.
*   **Low Priority:** i18n (translation).

[Read the Contributing Guide](CONTRIBUTING.md)
[View all Contributors](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

## License

This project is licensed under the [MIT License](LICENSE).

## Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
 </picture>
</a>

## Built with ❤️ by the SuperClaude community