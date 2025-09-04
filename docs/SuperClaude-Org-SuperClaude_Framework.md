# SuperClaude Framework: Supercharge Your Claude Code with Intelligent Automation

**Transform your Claude code into a structured development powerhouse with the SuperClaude Framework, offering powerful tools, specialized AI agents, and seamless workflow automation.**  For the original repository, visit [SuperClaude Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework).

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

*   **Intelligent Agents:** Utilize 14 specialized AI agents with domain expertise for tasks like security, UI design, and more.
*   **Enhanced Command Structure:** Simplify your workflow with a unified `/sc:` prefix for all commands, offering a clean and organized structure.
*   **Robust MCP Server Integration:** Leverage 6 powerful server integrations (Context7, Sequential, Magic, Playwright, Morphllm, and Serena) for diverse functionalities.
*   **Adaptive Behavioral Modes:** Optimize your development process with 6 adaptive modes for brainstorming, strategic analysis, orchestration, and more.
*   **Performance Optimization:** Experience a smaller framework footprint for enhanced code context and extended conversations.
*   **Comprehensive Documentation:** Explore a complete documentation overhaul with real-world examples, practical workflows, and improved navigation.

## Quick Installation

Choose your preferred installation method:

| Method   | Command                                                                    | Best For                               |
| :------- | :------------------------------------------------------------------------- | :------------------------------------- |
| **pipx** | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **Recommended** (Linux/macOS)          |
| **pip**  | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install`  | Traditional Python environments       |
| **npm**  | `npm install -g @bifrost_inc/superclaude && superclaude install`            | Cross-platform, Node.js users       |

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

Your support helps maintain and improve SuperClaude:

*   [Ko-fi](https://ko-fi.com/superclaude) - One-time contributions
*   [Patreon](https://patreon.com/superclaude) - Monthly support
*   [GitHub Sponsors](https://github.com/sponsors/SuperClaude-Org) - Flexible tiers

## What's New in V4

*   **Smarter Agent System:** 14 domain-expert agents.
*   **Improved Namespace:**  `/sc:` prefix for organized commands.
*   **MCP Server Integration:** Integration with 6 powerful servers.
*   **Behavioral Modes:** 6 adaptive modes for various contexts.
*   **Optimized Performance:** Reduced footprint for greater context.
*   **Documentation Overhaul:** Comprehensive rewrite with practical examples.

## Documentation

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

*   [Best Practices](Docs/Reference/quick-start-practices.md)
*   [Examples Cookbook](Docs/Reference/examples-cookbook.md)
*   [Troubleshooting](Docs/Reference/troubleshooting.md)

## Contributing

Join the SuperClaude community and contribute:

*   **High Priority:** Documentation, MCP Integration.
*   **Medium Priority:** Workflows, Testing.
*   **Low Priority:** i18n

*   [Contributing Guide](CONTRIBUTING.md)
*   [View All Contributors](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.

## Star History

<!--  Star History Chart -->
<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
 </picture>
</a>
<!--  End Star History Chart -->